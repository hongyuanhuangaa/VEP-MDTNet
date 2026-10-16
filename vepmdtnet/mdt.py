"""MDT: multi-scale deformable Transformer over the virtual features V1, V3 and V5."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .blocks import ConvGNAct
from .config import VEPMDTNetConfig
from .msda import (
    MDTDecoderLayer,
    MDTEncoderLayer,
    encoder_reference_points,
    query_reference_points,
    sine_position_encoding,
)


class MDT(nn.Module):
    """V1/V3/V5 are projected to ``mdt_dim`` channels and processed by the MDT encoder;
    the MDT decoder uses the encoded V3 (V3') as queries and aggregates context from
    the three levels with MSDA cross-attention.  ``output_conv`` produces F_MDT at the
    resolution of decoder stage D3, where the decoder fuses it.

    Independently of the MDT, ``v5_projection`` maps V5 to the input of the
    convolutional decoder, so disabling the MDT (``use_mdt=False``) is a clean
    one-factor ablation.
    """

    def __init__(self, input_channels: Sequence[int], config: VEPMDTNetConfig) -> None:
        super().__init__()
        if len(input_channels) != 3:
            raise ValueError("The MDT requires V1, V3 and V5 channels")
        self.config = config
        dim = config.mdt_dim

        self.v5_projection = ConvGNAct(input_channels[-1], config.decoder_in_channels, 1)

        self.input_projections = nn.ModuleList(
            [ConvGNAct(channels, dim, 1) for channels in input_channels]
            if config.use_mdt
            else []
        )
        self.v1_downsample = (
            ConvGNAct(dim, dim, 3, stride=config.mdt_v1_stride)
            if config.use_mdt and config.mdt_v1_stride > 1
            else nn.Identity()
        )

        if config.use_mdt and config.use_mdt_encoder:
            self.level_embedding = nn.Parameter(torch.empty(3, dim))
            nn.init.normal_(self.level_embedding, std=0.02)
        else:
            self.register_parameter("level_embedding", None)

        self.encoder_layers = nn.ModuleList(
            [
                MDTEncoderLayer(
                    dim=dim,
                    heads=config.mdt_heads,
                    levels=3,
                    points=config.mdt_points,
                    ffn_dim=config.mdt_ffn_dim,
                    dropout=config.mdt_dropout,
                    use_local_conv=config.use_mdt_local_conv,
                )
                for _ in range(
                    config.mdt_encoder_layers if config.use_mdt and config.use_mdt_encoder else 0
                )
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                MDTDecoderLayer(
                    dim=dim,
                    heads=config.mdt_heads,
                    levels=3,
                    points=config.mdt_points,
                    ffn_dim=config.mdt_ffn_dim,
                    dropout=config.mdt_dropout,
                )
                for _ in range(
                    config.mdt_decoder_layers if config.use_mdt and config.use_mdt_decoder else 0
                )
            ]
        )
        self.output_conv = (
            ConvGNAct(dim, config.mdt_out_channels, 3)
            if config.use_mdt and config.use_mdt_injection
            else None
        )

    def _positions(self, levels: Sequence[Tensor]) -> Tensor:
        if self.level_embedding is None:
            raise RuntimeError("Level embeddings exist only when the MDT encoder is enabled")
        encoded: List[Tensor] = []
        for index, level in enumerate(levels):
            position = sine_position_encoding(level, self.config.mdt_dim)
            position = position + self.level_embedding[index][None, None, :]
            encoded.append(position)
        return torch.cat(encoded, dim=1)

    def forward(
        self,
        v1: Tensor,
        v3: Tensor,
        v5: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, object]]:
        """Return (decoder input from V5, F_MDT or None, details)."""
        decoder_input = self.v5_projection(v5)
        if not self.config.use_mdt:
            return decoder_input, None, {
                "mode": "no_mdt",
                "memory_shapes": [],
                "query_shape": None,
                "sampling_points_per_head_per_level": 0,
            }

        levels = [
            projection(value)
            for projection, value in zip(self.input_projections, (v1, v3, v5))
        ]
        raw_shapes = [tuple(level.shape) for level in levels]
        levels[0] = self.v1_downsample(levels[0])

        if self.config.use_mdt_encoder:
            positions = self._positions(levels)
            references = encoder_reference_points(levels)
            for layer in self.encoder_layers:
                levels = layer(levels, positions, references)

        # V3' (the middle level) provides the queries.  Each query/head samples
        # K learned points from every V1/V3/V5 memory level.
        query_level = levels[1]
        query = query_level.flatten(2).transpose(1, 2)
        if self.config.use_mdt_decoder:
            query_position = sine_position_encoding(query_level, self.config.mdt_dim)
            query_references = query_reference_points(query_level, len(levels))
            for layer in self.decoder_layers:
                query = layer(query, query_position, query_references, levels)

        mdt_2d = query.transpose(1, 2).reshape_as(query_level)
        f_mdt = self.output_conv(mdt_2d) if self.output_conv is not None else None
        return decoder_input, f_mdt, {
            "mode": "mdt_v135_query_v3",
            "input_shapes": raw_shapes,
            "memory_shapes": [tuple(item.shape) for item in levels],
            "query_shape": tuple(query.shape),
            "query_stage": 3,
            "injection_stage": self.config.mdt_inject_stage,
            "sampling_points_per_head_per_level": self.config.mdt_points,
            "f_mdt_shape": tuple(f_mdt.shape) if f_mdt is not None else None,
        }
