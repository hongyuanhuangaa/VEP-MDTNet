from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn

from .blocks import ConvGNAct


class StructureGuidanceHead(nn.Module):
    """Learn a one-channel structural hint E from a virtual feature."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden = max(16, min(128, in_channels // 4))
        self.head = nn.Sequential(
            ConvGNAct(in_channels, hidden, 3),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class RegionGuidanceHead(nn.Module):
    """Learn a one-channel region-confidence hint P from a virtual feature."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class VirtualSkipGuidance(nn.Module):
    """Generate E/P only from V4,V3,V2,V1 in decoder order."""

    def __init__(
        self,
        skip_channels: Sequence[int],
        use_e: bool,
        use_p: bool,
    ) -> None:
        super().__init__()
        self.use_e = use_e
        self.use_p = use_p
        self.e_heads = nn.ModuleList(
            [StructureGuidanceHead(channels) for channels in skip_channels]
        ) if use_e else nn.ModuleList()
        self.p_heads = nn.ModuleList(
            [RegionGuidanceHead(channels) for channels in skip_channels]
        ) if use_p else nn.ModuleList()

    def forward(self, skips: Sequence[Tensor]) -> Dict[str, List[Tensor]]:
        if self.use_e:
            e_maps = [head(value) for head, value in zip(self.e_heads, skips)]
        else:
            e_maps = [value.new_zeros(value.size(0), 1, *value.shape[-2:]) for value in skips]
        if self.use_p:
            p_maps = [head(value) for head, value in zip(self.p_heads, skips)]
        else:
            p_maps = [value.new_zeros(value.size(0), 1, *value.shape[-2:]) for value in skips]
        return {"E": e_maps, "P": p_maps}

