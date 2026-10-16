"""U-Net decoder D4..D1 with an E-P Skip Gate per stage and F_MDT fused at D3."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import DoubleConv
from .ep_skip_gate import EPSkipGate


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        decoder_channels: Sequence[int],
        use_gate: bool,
        mdt_channels: Optional[int] = None,
        mdt_inject_index: int = 1,
    ) -> None:
        super().__init__()
        if len(skip_channels) != 4 or len(decoder_channels) != 4:
            raise ValueError("The decoder requires four skip stages")
        blocks: List[nn.Module] = []
        current = in_channels
        for skip, output in zip(skip_channels, decoder_channels):
            blocks.append(EPSkipGate(current, skip, output, use_gate=use_gate))
            current = output
        self.blocks = nn.ModuleList(blocks)
        if not 0 <= mdt_inject_index < len(decoder_channels):
            raise ValueError("mdt_inject_index is outside the decoder")
        self.mdt_inject_index = mdt_inject_index
        # Concat[D3, F_MDT] -> two 3x3 Conv-GN-SiLU
        self.mdt_fusion = (
            DoubleConv(decoder_channels[mdt_inject_index] + mdt_channels,
                       decoder_channels[mdt_inject_index])
            if mdt_channels is not None
            else None
        )
        self.out_channels = current

    def forward(
        self,
        x: Tensor,
        skips: Sequence[Tensor],
        e_maps: Sequence[Tensor],
        p_maps: Sequence[Tensor],
        f_mdt: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor], List[Dict[str, Tensor]]]:
        if not (len(skips) == len(e_maps) == len(p_maps) == len(self.blocks)):
            raise ValueError("Each decoder stage requires skip, E and P")
        stages: List[Tensor] = []
        auxiliary: List[Dict[str, Tensor]] = []
        for index, (block, skip, e, p) in enumerate(zip(self.blocks, skips, e_maps, p_maps)):
            x, details = block(x, skip, e, p)
            if index == self.mdt_inject_index and self.mdt_fusion is not None:
                if f_mdt is None:
                    raise ValueError("The configured decoder requires F_MDT")
                if f_mdt.shape[-2:] != x.shape[-2:]:
                    f_mdt = F.interpolate(f_mdt, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = self.mdt_fusion(torch.cat([x, f_mdt], dim=1))
                details["F_MDT"] = f_mdt
            stages.append(x)
            auxiliary.append(details)
        return x, stages, auxiliary
