"""E-P Skip Gate: edge guidance response E, patch region response P and the gated skip connection.

    E_n   = sigmoid(Conv1x1(SiLU(GN(Conv3x3(V_n)))))            edge guidance response
    P_n   = sigmoid(Conv1x1(V_n))                               patch region response
    eta_n = sigmoid(Conv1x1(Concat[V_n, U_n, E_n, P_n]))        gating coefficient
    V~_n  = eta_n * V_n
    F_n^skip = Conv3x3(Conv3x3(Concat[U_n, V~_n, E_n, P_n]))
where U_n is the bilinearly upsampled feature of the previous decoder stage.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvGNAct, DoubleConv


class EdgeGuidanceHead(nn.Module):
    """E_n: one-channel edge guidance response of a virtual feature."""

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


class PatchRegionHead(nn.Module):
    """P_n: one-channel patch region response of a virtual feature."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class EPGuidance(nn.Module):
    """E and P for the skip features V4, V3, V2, V1 (decoder order).

    A disabled response (use_e / use_p = False) is replaced by a zero map.
    """

    def __init__(self, skip_channels: Sequence[int], use_e: bool, use_p: bool) -> None:
        super().__init__()
        self.use_e = use_e
        self.use_p = use_p
        self.edge_heads = nn.ModuleList(
            [EdgeGuidanceHead(channels) for channels in skip_channels]
        ) if use_e else nn.ModuleList()
        self.patch_heads = nn.ModuleList(
            [PatchRegionHead(channels) for channels in skip_channels]
        ) if use_p else nn.ModuleList()

    def forward(self, skips: Sequence[Tensor]) -> Dict[str, List[Tensor]]:
        if self.use_e:
            e_maps = [head(value) for head, value in zip(self.edge_heads, skips)]
        else:
            e_maps = [value.new_zeros(value.size(0), 1, *value.shape[-2:]) for value in skips]
        if self.use_p:
            p_maps = [head(value) for head, value in zip(self.patch_heads, skips)]
        else:
            p_maps = [value.new_zeros(value.size(0), 1, *value.shape[-2:]) for value in skips]
        return {"E": e_maps, "P": p_maps}


class EPSkipGate(nn.Module):
    """One decoder stage: upsample to U_n, gate the skip feature with eta_n, fuse to F_n^skip.

    With ``use_gate=False`` the skip feature is passed unchanged (ordinary U-Net skip).
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_gate: bool) -> None:
        super().__init__()
        self.use_gate = use_gate
        self.gate = (
            nn.Sequential(nn.Conv2d(skip_channels + in_channels + 2, skip_channels, 1), nn.Sigmoid())
            if use_gate
            else None
        )
        self.fuse = DoubleConv(in_channels + skip_channels + 2, out_channels)

    @staticmethod
    def _resize(value: Tensor, size) -> Tensor:
        if value.shape[-2:] == tuple(size):
            return value
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: Tensor, skip: Tensor, e: Tensor, p: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        size = skip.shape[-2:]
        u = self._resize(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), size)
        e = self._resize(e, size)
        p = self._resize(p, size)
        if self.gate is not None:
            eta = self.gate(torch.cat([skip, u, e, p], dim=1))
            skip_used = eta * skip
        else:
            eta = torch.ones_like(skip)
            skip_used = skip
        output = self.fuse(torch.cat([u, skip_used, e, p], dim=1))
        return output, {"E": e, "P": p, "eta": eta}
