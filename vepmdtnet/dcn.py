from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
from torchvision.ops import deform_conv2d


class DCNv2(nn.Module):
    """Modulated deformable convolution (DCN) replacing a pretrained nn.Conv2d.

    Offsets start at zero. The modulation mask starts close to one so replacing
    a pretrained ResNet conv does not halve its activation at initialization.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        initial_mask: float = 0.90,
        force_fp32: bool = True,
    ) -> None:
        super().__init__()
        if conv.padding_mode != "zeros":
            raise ValueError("Only zero-padded Conv2d can be converted to DCNv2")
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.kernel_size = conv.kernel_size
        self.force_fp32 = force_fp32

        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = (
            nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        )
        points = self.kernel_size[0] * self.kernel_size[1]
        self.offset_mask = nn.Conv2d(
            conv.in_channels,
            3 * points,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        mask_logit = math.log(initial_mask / (1.0 - initial_mask))
        with torch.no_grad():
            self.offset_mask.bias[2 * points :].fill_(mask_logit)

    def _deform(self, x: Tensor) -> Tensor:
        offset_mask = self.offset_mask(x)
        points = self.kernel_size[0] * self.kernel_size[1]
        offset = offset_mask[:, : 2 * points]
        mask = torch.sigmoid(offset_mask[:, 2 * points :])
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self.force_fp32 or x.dtype == torch.float32:
            return self._deform(x)
        # torchvision deform_conv2d is the most numerically sensitive part of
        # this model under AMP.  Compute offsets, masks and resampling in FP32,
        # then return to the surrounding ResNet activation dtype.
        original_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._deform(x.float())
        return output.to(dtype=original_dtype)


def replace_bottleneck_conv2(
    stage: nn.Sequential,
    mode: str,
    initial_mask: float,
    force_fp32: bool = True,
) -> int:
    indices: Iterable[int]
    if mode == "last":
        indices = (len(stage) - 1,)
    elif mode == "all":
        indices = range(len(stage))
    else:
        raise ValueError(f"Unsupported DCN replacement mode: {mode}")

    replaced = 0
    for index in indices:
        block = stage[index]
        conv2 = getattr(block, "conv2", None)
        if isinstance(conv2, DCNv2):
            continue
        if not isinstance(conv2, nn.Conv2d):
            raise TypeError(f"Expected Bottleneck.conv2 to be Conv2d, got {type(conv2)}")
        block.conv2 = DCNv2(
            conv2,
            initial_mask=initial_mask,
            force_fp32=force_fp32,
        )
        replaced += 1
    return replaced
