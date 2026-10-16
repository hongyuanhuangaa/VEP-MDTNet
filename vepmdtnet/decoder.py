from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import DoubleConv
from .upsampling import UpsampleLayer


class GuidedSkipDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_carafe: bool,
        use_skip_gate: bool,
    ) -> None:
        super().__init__()
        self.use_skip_gate = use_skip_gate
        self.upsample = UpsampleLayer(in_channels, use_carafe)
        self.skip_gate = (
            nn.Sequential(
                nn.Conv2d(skip_channels + in_channels + 2, skip_channels, 1),
                nn.Sigmoid(),
            )
            if use_skip_gate
            else None
        )
        self.fuse = DoubleConv(in_channels + skip_channels + 2, out_channels)

    @staticmethod
    def _align(value: Tensor, target: Tensor) -> Tensor:
        if value.shape[-2:] == target.shape[-2:]:
            return value
        return F.interpolate(
            value, size=target.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self, x: Tensor, skip: Tensor, e: Tensor, p: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        x_up = self.upsample(x, skip.shape[-2:])
        e = self._align(e, skip)
        p = self._align(p, skip)
        if self.skip_gate is not None:
            eta = self.skip_gate(torch.cat([skip, x_up, e, p], dim=1))
            skip_used = eta * skip
        else:
            eta = torch.ones_like(skip)
            skip_used = skip
        output = self.fuse(torch.cat([x_up, skip_used, e, p], dim=1))
        return output, {"E": e, "P": p, "eta": eta}


class VirtualGuidedDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        decoder_channels: Sequence[int],
        use_carafe: bool,
        use_skip_gate: bool,
        emrt_refinement_channels: Optional[int] = None,
        emrt_inject_index: int = 1,
    ) -> None:
        super().__init__()
        if len(skip_channels) != 4 or len(decoder_channels) != 4:
            raise ValueError("The decoder requires four skip stages")
        blocks: List[nn.Module] = []
        current = in_channels
        for skip, output in zip(skip_channels, decoder_channels):
            blocks.append(
                GuidedSkipDecoderBlock(
                    current,
                    skip,
                    output,
                    use_carafe=use_carafe,
                    use_skip_gate=use_skip_gate,
                )
            )
            current = output
        self.blocks = nn.ModuleList(blocks)
        if not 0 <= emrt_inject_index < len(decoder_channels):
            raise ValueError("emrt_inject_index is outside the decoder")
        self.emrt_inject_index = emrt_inject_index
        self.emrt_fusion = (
            DoubleConv(
                decoder_channels[emrt_inject_index]
                + emrt_refinement_channels,
                decoder_channels[emrt_inject_index],
            )
            if emrt_refinement_channels is not None
            else None
        )
        self.out_channels = current

    @staticmethod
    def _align_refinement(value: Tensor, target: Tensor) -> Tensor:
        if value.shape[-2:] == target.shape[-2:]:
            return value
        return F.interpolate(
            value,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        x: Tensor,
        skips: Sequence[Tensor],
        e_maps: Sequence[Tensor],
        p_maps: Sequence[Tensor],
        emrt_refinement: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor], List[Dict[str, Tensor]]]:
        if not (len(skips) == len(e_maps) == len(p_maps) == len(self.blocks)):
            raise ValueError("Each decoder stage requires skip, E and P")
        stages: List[Tensor] = []
        auxiliary: List[Dict[str, Tensor]] = []
        for index, (block, skip, e, p) in enumerate(
            zip(self.blocks, skips, e_maps, p_maps)
        ):
            x, details = block(x, skip, e, p)
            if index == self.emrt_inject_index and self.emrt_fusion is not None:
                if emrt_refinement is None:
                    raise ValueError(
                        "The configured decoder requires an EMRT refinement"
                    )
                aligned = self._align_refinement(emrt_refinement, x)
                x = self.emrt_fusion(torch.cat([x, aligned], dim=1))
                details["emrt_refinement"] = aligned
            stages.append(x)
            auxiliary.append(details)
        return x, stages, auxiliary
