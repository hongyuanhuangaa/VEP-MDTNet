"""Multi-scale deformable attention (MSDA, paper Eq. msda) and the MDT encoder / decoder layers."""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvGNAct


def flatten_levels(levels: Sequence[Tensor]) -> Tensor:
    return torch.cat([item.flatten(2).transpose(1, 2) for item in levels], dim=1)


def split_levels(tokens: Tensor, shapes: Sequence[tuple[int, int]]) -> List[Tensor]:
    lengths = [height * width for height, width in shapes]
    pieces = tokens.split(lengths, dim=1)
    return [
        piece.transpose(1, 2).reshape(tokens.size(0), tokens.size(2), height, width)
        for piece, (height, width) in zip(pieces, shapes)
    ]


def normalized_grid(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def encoder_reference_points(levels: Sequence[Tensor]) -> Tensor:
    batch = levels[0].size(0)
    grids = [
        normalized_grid(x.size(2), x.size(3), x.device, x.dtype) for x in levels
    ]
    base = torch.cat(grids, dim=0)
    return base[None, :, None, :].expand(batch, -1, len(levels), -1)


def query_reference_points(query_level: Tensor, num_levels: int) -> Tensor:
    base = normalized_grid(
        query_level.size(2), query_level.size(3), query_level.device, query_level.dtype
    )
    return base[None, :, None, :].expand(query_level.size(0), -1, num_levels, -1)


def sine_position_encoding(level: Tensor, hidden_dim: int) -> Tensor:
    if hidden_dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by four")
    height, width = level.shape[-2:]
    grid = normalized_grid(height, width, level.device, level.dtype)
    x = grid[:, 0] * (2.0 * math.pi)
    y = grid[:, 1] * (2.0 * math.pi)
    quarter = hidden_dim // 4
    omega = torch.arange(quarter, device=level.device, dtype=level.dtype)
    omega = 1.0 / (10000.0 ** (omega / max(1, quarter)))
    x = x[:, None] * omega[None]
    y = y[:, None] * omega[None]
    position = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    return position[None].expand(level.size(0), -1, -1)


class MSDA(nn.Module):
    """Multi-scale deformable attention in pure PyTorch (grid_sample).

    Each of the M heads samples K points around the reference point on each of
    the L levels (offsets Delta p_mlqk, weights A_mlqk normalised over L*K).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        total = num_heads * num_levels * num_points
        self.sampling_offsets = nn.Linear(embed_dim, total * 2)
        self.attention_weights = nn.Linear(embed_dim, total)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        theta = torch.arange(self.num_heads, dtype=torch.float32)
        theta = theta * (2.0 * math.pi / self.num_heads)
        directions = torch.stack([theta.cos(), theta.sin()], dim=-1)
        directions = directions / directions.abs().amax(dim=-1, keepdim=True)
        directions = directions[:, None, None, :].repeat(
            1, self.num_levels, self.num_points, 1
        )
        radii = torch.arange(1, self.num_points + 1, dtype=torch.float32)
        directions = directions * radii[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(directions.reshape(-1))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value_levels: Sequence[Tensor],
    ) -> Tensor:
        if len(value_levels) != self.num_levels:
            raise ValueError("value_levels does not match configured num_levels")
        batch, query_length, _ = query.shape
        shapes = [(x.size(2), x.size(3)) for x in value_levels]
        lengths = [height * width for height, width in shapes]
        values = flatten_levels(value_levels)
        values = self.value_proj(values)
        value_pieces = values.split(lengths, dim=1)

        offsets = self.sampling_offsets(query).view(
            batch,
            query_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        weights = self.attention_weights(query).view(
            batch,
            query_length,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        weights = weights.softmax(dim=-1).view(
            batch,
            query_length,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        output = query.new_zeros(
            batch, query_length, self.num_heads, self.head_dim
        )
        for level_index, ((height, width), value) in enumerate(
            zip(shapes, value_pieces)
        ):
            value = value.reshape(
                batch, height * width, self.num_heads, self.head_dim
            )
            value = value.permute(0, 2, 3, 1).reshape(
                batch * self.num_heads, self.head_dim, height, width
            )
            normalizer = query.new_tensor([width, height])
            locations = (
                reference_points[:, :, None, level_index, None, :]
                + offsets[:, :, :, level_index] / normalizer
            )
            grid = locations.permute(0, 2, 1, 3, 4).reshape(
                batch * self.num_heads, query_length, self.num_points, 2
            )
            sampled = F.grid_sample(
                value,
                grid * 2.0 - 1.0,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.reshape(
                batch,
                self.num_heads,
                self.head_dim,
                query_length,
                self.num_points,
            ).permute(0, 3, 1, 4, 2)
            level_weights = weights[:, :, :, level_index, :, None]
            output = output + (sampled * level_weights).sum(dim=3)

        output = output.reshape(batch, query_length, self.embed_dim)
        return self.output_proj(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MDTEncoderLayer(nn.Module):
    """MDT encoder layer: optional local 3x3 conv per level, MSDA self-attention, FFN."""

    def __init__(
        self,
        dim: int,
        heads: int,
        levels: int,
        points: int,
        ffn_dim: int,
        dropout: float,
        use_local_conv: bool,
    ) -> None:
        super().__init__()
        self.use_local_conv = use_local_conv
        self.local = nn.ModuleList(
            [ConvGNAct(dim, dim, 3) for _ in range(levels)]
            if use_local_conv
            else [nn.Identity() for _ in range(levels)]
        )
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MSDA(dim, heads, levels, points)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout)

    def forward(
        self,
        levels: Sequence[Tensor],
        positions: Tensor,
        reference_points: Tensor,
    ) -> List[Tensor]:
        local_levels = [
            level + conv(level) if self.use_local_conv else level
            for level, conv in zip(levels, self.local)
        ]
        source = flatten_levels(local_levels)
        attended = self.attention(
            self.norm1(source + positions), reference_points, local_levels
        )
        source = source + self.dropout(attended)
        source = source + self.ffn(self.norm2(source))
        shapes = [(x.size(2), x.size(3)) for x in local_levels]
        return split_levels(source, shapes)


class MDTDecoderLayer(nn.Module):
    """MDT decoder layer: query self-attention, MSDA cross-attention to the V1/V3/V5 memory, FFN."""

    def __init__(
        self,
        dim: int,
        heads: int,
        levels: int,
        points: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attention = MSDA(dim, heads, levels, points)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        query_position: Tensor,
        reference_points: Tensor,
        memory_levels: Sequence[Tensor],
    ) -> Tensor:
        q = self.norm1(query + query_position)
        attended, _ = self.self_attention(q, q, q, need_weights=False)
        query = query + self.dropout(attended)
        cross_query = self.norm2(query + query_position)
        query = query + self.dropout(
            self.cross_attention(cross_query, reference_points, memory_levels)
        )
        query = query + self.ffn(self.norm3(query))
        return query
