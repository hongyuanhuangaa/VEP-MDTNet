from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .blocks import ConvGNAct
from .config import VFEMRTConfig
from .msdeform_transformer import (
    MSDeformDecoderLayer,
    MSDeformEncoderLayer,
    encoder_reference_points,
    query_reference_points,
    sine_position_encoding,
)


class MultiscaleDeformableBridge(nn.Module):
    """V1/V3/V5 EMRT refinement with a V3 query grid.

    The ordinary convolutional decoder still starts from a projected V5 at
    output stride 32. In parallel, projected V1/V3/V5 features are processed
    by the multiscale deformable encoder-decoder. Its V3-resolution output is
    injected later, after the decoder has fused the V3 skip at output stride 8.
    """

    def __init__(self, input_channels: Sequence[int], config: VFEMRTConfig) -> None:
        super().__init__()
        if len(input_channels) != 3:
            raise ValueError("The EMRT module requires V1, V3 and V5 channels")
        self.config = config
        dim = config.bridge_hidden_dim

        # This path is independent of EMRT and always starts the CNN decoder
        # from V5. Consequently, disabling EMRT is a clean one-factor ablation.
        self.simple_output = ConvGNAct(
            input_channels[-1], config.bridge_channels, 1
        )

        self.projections = nn.ModuleList(
            [ConvGNAct(channels, dim, 1) for channels in input_channels]
            if config.use_msdeform_bridge
            else []
        )
        self.v1_downsample = (
            ConvGNAct(
                dim,
                dim,
                3,
                stride=config.emrt_v1_encoder_stride,
            )
            if config.use_msdeform_bridge
            and config.emrt_v1_encoder_stride > 1
            else nn.Identity()
        )

        if config.use_msdeform_bridge and config.use_msdeform_encoder:
            self.level_embedding = nn.Parameter(torch.empty(3, dim))
            nn.init.normal_(self.level_embedding, std=0.02)
        else:
            self.register_parameter("level_embedding", None)

        self.encoder_layers = nn.ModuleList(
            [
                MSDeformEncoderLayer(
                    dim=dim,
                    heads=config.bridge_heads,
                    levels=3,
                    points=config.bridge_points,
                    ffn_dim=config.bridge_ffn_dim,
                    dropout=config.bridge_dropout,
                    use_local_conv=config.use_bridge_local_conv,
                )
                for _ in range(
                    config.bridge_encoder_layers
                    if config.use_msdeform_bridge
                    and config.use_msdeform_encoder
                    else 0
                )
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                MSDeformDecoderLayer(
                    dim=dim,
                    heads=config.bridge_heads,
                    levels=3,
                    points=config.bridge_points,
                    ffn_dim=config.bridge_ffn_dim,
                    dropout=config.bridge_dropout,
                )
                for _ in range(
                    config.bridge_decoder_layers
                    if config.use_msdeform_bridge
                    and config.use_msdeform_decoder
                    else 0
                )
            ]
        )
        self.refine_output = (
            ConvGNAct(dim, config.emrt_refine_channels, 3)
            if config.use_msdeform_bridge
            and config.use_emrt_decoder_refinement
            else None
        )

    def _positions(self, levels: Sequence[Tensor]) -> Tensor:
        if self.level_embedding is None:
            raise RuntimeError("Level embeddings exist only when the encoder is enabled")
        encoded: List[Tensor] = []
        for index, level in enumerate(levels):
            position = sine_position_encoding(
                level, self.config.bridge_hidden_dim
            )
            position = position + self.level_embedding[index][None, None, :]
            encoded.append(position)
        return torch.cat(encoded, dim=1)

    def forward(
        self,
        v1: Tensor,
        v3: Tensor,
        v5: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, object]]:
        bridge = self.simple_output(v5)
        if not self.config.use_msdeform_bridge:
            return bridge, None, {
                "mode": "simple_v5_bridge",
                "memory_shapes": [],
                "query_shape": None,
                "sampling_points_per_head_per_level": 0,
            }

        levels = [
            projection(value)
            for projection, value in zip(self.projections, (v1, v3, v5))
        ]
        raw_shapes = [tuple(level.shape) for level in levels]
        levels[0] = self.v1_downsample(levels[0])

        if self.config.use_msdeform_encoder:
            positions = self._positions(levels)
            references = encoder_reference_points(levels)
            for layer in self.encoder_layers:
                levels = layer(levels, positions, references)

        # V3 (the middle level) provides 32x32 queries. Each query/head samples
        # K learned points from every V1/V3/V5 memory level.
        query_level = levels[1]
        query = query_level.flatten(2).transpose(1, 2)
        if self.config.use_msdeform_decoder:
            query_position = sine_position_encoding(
                query_level, self.config.bridge_hidden_dim
            )
            query_references = query_reference_points(
                query_level, len(levels)
            )
            for layer in self.decoder_layers:
                query = layer(
                    query,
                    query_position,
                    query_references,
                    levels,
                )

        emrt_2d = query.transpose(1, 2).reshape_as(query_level)
        refinement = (
            self.refine_output(emrt_2d)
            if self.refine_output is not None
            else None
        )
        return bridge, refinement, {
            "mode": "v135_q3_multiscale_deformable_refinement",
            "input_shapes": raw_shapes,
            "memory_shapes": [tuple(item.shape) for item in levels],
            "query_shape": tuple(query.shape),
            "query_stage": 3,
            "injection_stage": self.config.emrt_inject_stage,
            "sampling_points_per_head_per_level": self.config.bridge_points,
            "refinement_shape": (
                tuple(refinement.shape) if refinement is not None else None
            ),
        }
