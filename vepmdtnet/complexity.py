"""Parameter count and FLOP estimate that understands the custom operators of the model."""

from __future__ import annotations

import time
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn

from .deformable import ModulatedDeformConvPack
from .msdeform_transformer import MultiScaleDeformableAttention
from .upsampling import CARAFE


def profile_model_complexity(
    model: nn.Module,
    input_channels: Sequence[int],
    input_size: Tuple[int, int],
    device: torch.device,
) -> Dict[str, object]:
    """Estimate one-sample FLOPs with forward hooks.

    ``input_channels`` lists the channels of each positional model input, e.g. ``(3, 1)``
    for RGB + DEM.  A multiply and an addition count as two FLOPs.  Convolution, linear,
    DCNv2 resampling, deformable-attention sampling, multi-head attention and CARAFE
    reassembly are included; normalisation, activations, softmax and ordinary
    interpolation are omitted, as is common in segmentation papers.
    """
    flop_counts = {
        "conv_linear": 0,
        "dcn_main_and_sampling": 0,
        "deformable_attention_sampling": 0,
        "multihead_attention": 0,
        "carafe_reassembly": 0,
    }
    handles = []

    def conv_hook(module: nn.Conv2d, inputs, output: Tensor) -> None:
        kernel_ops = (
            module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        )
        flop_counts["conv_linear"] += int(2 * output.numel() * kernel_ops)

    def linear_hook(module: nn.Linear, inputs, output: Tensor) -> None:
        flop_counts["conv_linear"] += int(2 * output.numel() * module.in_features)

    def dcn_hook(module: ModulatedDeformConvPack, inputs, output: Tensor) -> None:
        points = module.kernel_size[0] * module.kernel_size[1]
        kernel_ops = module.weight.size(1) * points
        convolution = 2 * output.numel() * kernel_ops
        # Four-neighbour bilinear interpolation plus modulation/accumulation.
        resampling = 8 * output.numel() * points
        flop_counts["dcn_main_and_sampling"] += int(convolution + resampling)

    def msdeform_hook(module: MultiScaleDeformableAttention, inputs, output: Tensor) -> None:
        batch, queries, channels = output.shape
        samples = module.num_levels * module.num_points
        flop_counts["deformable_attention_sampling"] += int(9 * batch * queries * channels * samples)

    def mha_hook(module: nn.MultiheadAttention, inputs, output) -> None:
        query, key = inputs[0], inputs[1]
        if module.batch_first:
            batch, query_length, embedding = query.shape
            key_length = key.size(1)
        else:
            query_length, batch, embedding = query.shape
            key_length = key.size(0)
        projections = 2 * batch * (
            query_length * embedding * embedding * 2 + key_length * embedding * embedding * 2
        )
        attention = 4 * batch * query_length * key_length * embedding
        flop_counts["multihead_attention"] += int(projections + attention)

    def carafe_hook(module: CARAFE, inputs, output: Tensor) -> None:
        points = module.up_kernel * module.up_kernel
        flop_counts["carafe_reassembly"] += int(2 * output.numel() * points)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, ModulatedDeformConvPack):
            handles.append(module.register_forward_hook(dcn_hook))
        elif isinstance(module, MultiScaleDeformableAttention):
            handles.append(module.register_forward_hook(msdeform_hook))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(mha_hook))
        elif isinstance(module, CARAFE):
            handles.append(module.register_forward_hook(carafe_hook))

    was_training = model.training
    model.eval()
    inputs = [
        torch.zeros(1, channels, *input_size, device=device, dtype=torch.float32)
        for channels in input_channels
    ]
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
        del inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_flops = int(sum(flop_counts.values()))
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "input_size": tuple(input_size),
        "batch_size": 1,
        "total_parameters": total_params,
        "trainable_parameters": trainable,
        "estimated_flops": total_flops,
        "estimated_gflops": total_flops / 1.0e9,
        "estimated_gmacs": total_flops / 2.0e9,
        "profiling_seconds": seconds,
        "flop_breakdown": flop_counts,
        "convention": "multiply and add are two FLOPs",
    }
