from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvGNAct


class RecursiveVirtualStage(nn.Module):
    """One strictly feed-forward virtual stage.

    The real RGB and DEM tensors are never modified.  At every stage the
    current RGB/DEM projections form both the stable average baseline and the
    modality-aware part of the residual input.  From stage two onward, the
    previous virtual feature is resized to the current resolution and appended
    to that residual input.  Thus the implemented recurrence is

        V_n = 0.5 * (R_n_hat + D_n_hat)
              + f_n([R_n_hat, D_n_hat, resize(V_{n-1})]).

    ``previous_rgb_channels`` and ``previous_dem_channels`` remain in the
    constructor only for compatibility with older callers and component tests;
    previous real-stream features are intentionally not used.
    """

    def __init__(
        self,
        stage_index: int,
        current_rgb_channels: int,
        current_dem_channels: int,
        virtual_channels: int,
        previous_rgb_channels: Optional[int] = None,
        previous_dem_channels: Optional[int] = None,
        previous_virtual_channels: Optional[int] = None,
        use_previous_virtual: bool = True,
        use_direct_concat_fusion: bool = False,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.stage_index = stage_index
        self.use_previous_virtual = use_previous_virtual and stage_index > 1
        self.use_direct_concat_fusion = use_direct_concat_fusion
        self.current_rgb_projection = ConvGNAct(
            current_rgb_channels, virtual_channels, 1
        )
        self.current_dem_projection = ConvGNAct(
            current_dem_channels, virtual_channels, 1
        )
        self.direct_concat_fusion = (
            ConvGNAct(2 * virtual_channels, virtual_channels, 1)
            if use_direct_concat_fusion
            else None
        )

        # Kept as legacy constructor arguments so older callers do not break.
        # The revised residual is conditioned on current RGB/DEM features,
        # rather than redundantly re-reading previous real-stream features.
        _ = previous_rgb_channels, previous_dem_channels

        if stage_index == 1:
            residual_sources = 2
            self.previous_rgb_projection = None
            self.previous_dem_projection = None
            self.previous_virtual_projection = None
        else:
            self.previous_rgb_projection = None
            self.previous_dem_projection = None
            self.previous_virtual_projection = (
                ConvGNAct(previous_virtual_channels, virtual_channels, 1)
                if self.use_previous_virtual and previous_virtual_channels is not None
                else None
            )
            residual_sources = 2 + int(self.previous_virtual_projection is not None)

        self.residual_conv1 = ConvGNAct(
            residual_sources * virtual_channels,
            virtual_channels,
            3,
            stride=1,
        )
        self.residual_conv2 = ConvGNAct(
            virtual_channels, virtual_channels, 3, activate=False
        )
        if zero_init_residual:
            nn.init.zeros_(self.residual_conv2.norm.weight)
            nn.init.zeros_(self.residual_conv2.norm.bias)

    @staticmethod
    def _align(x: Tensor, target: Tensor) -> Tensor:
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(
            x, size=target.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self,
        current_rgb: Tensor,
        current_dem: Tensor,
        previous_rgb: Optional[Tensor] = None,
        previous_dem: Optional[Tensor] = None,
        previous_virtual: Optional[Tensor] = None,
        enable_residual: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        current_dem = self._align(current_dem, current_rgb)
        rgb_base = self.current_rgb_projection(current_rgb)
        dem_base = self.current_dem_projection(current_dem)
        average = 0.5 * (rgb_base + dem_base)

        if not enable_residual:
            residual = torch.zeros_like(average)
            virtual = (
                self.direct_concat_fusion(
                    torch.cat([rgb_base, dem_base], dim=1)
                )
                if self.direct_concat_fusion is not None
                else average
            )
        else:
            if self.stage_index == 1:
                residual_input = torch.cat([rgb_base, dem_base], dim=1)
            else:
                # Current real features describe the present semantic level;
                # V_{n-1} contributes only the recursively fused history.
                sources = [rgb_base, dem_base]
                if self.previous_virtual_projection is not None:
                    if previous_virtual is None:
                        raise ValueError("previous_virtual is required by this stage")
                    previous_virtual_aligned = self.previous_virtual_projection(
                        previous_virtual
                    )
                    previous_virtual_aligned = self._align(
                        previous_virtual_aligned, average
                    )
                    sources.append(previous_virtual_aligned)
                residual_input = torch.cat(sources, dim=1)

            residual = self.residual_conv2(self.residual_conv1(residual_input))
            residual = self._align(residual, average)
            virtual = average + residual
        return virtual, {"average": average, "residual": residual}
