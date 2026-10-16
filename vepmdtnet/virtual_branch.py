from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvGNAct


class VirtualBranchStage(nn.Module):
    """One stage n of the virtual branch (paper Eq. vb).

    The RGB and DEM encoder features are never modified.  phi_R and phi_D project
    R_n and D_n to the virtual width; their average is the baseline, and the
    residual mapping F_n (two 3x3 convolutions) is computed from
    Concat[phi_R(R_n), phi_D(D_n), B_n(phi_V(V_{n-1}))]:

        V_n = 0.5 * (phi_R(R_n) + phi_D(D_n)) + F_n([...]),

    where the V_{n-1} term is absent at stage 1.  When the virtual branch is
    disabled, V_n is the projected average or, with ``use_concat_fusion``, a 1x1
    Conv-GN-SiLU of the concatenated projections.
    """

    def __init__(
        self,
        stage_index: int,
        rgb_channels: int,
        dem_channels: int,
        virtual_channels: int,
        prev_virtual_channels: Optional[int] = None,
        use_previous_virtual: bool = True,
        use_concat_fusion: bool = False,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.stage_index = stage_index
        self.use_previous_virtual = use_previous_virtual and stage_index > 1
        self.rgb_projection = ConvGNAct(rgb_channels, virtual_channels, 1)        # phi_R
        self.dem_projection = ConvGNAct(dem_channels, virtual_channels, 1)        # phi_D
        self.concat_fusion = (
            ConvGNAct(2 * virtual_channels, virtual_channels, 1)
            if use_concat_fusion
            else None
        )
        self.prev_virtual_projection = (                                          # phi_V
            ConvGNAct(prev_virtual_channels, virtual_channels, 1)
            if self.use_previous_virtual and prev_virtual_channels is not None
            else None
        )
        residual_sources = 2 + int(self.prev_virtual_projection is not None)

        # F_n: 3x3 Conv-GN-SiLU, then 3x3 Conv-GN; the last GN starts at zero so
        # that V_n initially equals the average of the projections.
        self.residual_conv1 = ConvGNAct(residual_sources * virtual_channels, virtual_channels, 3)
        self.residual_conv2 = ConvGNAct(virtual_channels, virtual_channels, 3, activate=False)
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
        rgb: Tensor,
        dem: Tensor,
        prev_virtual: Optional[Tensor] = None,
        use_residual: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        dem = self._align(dem, rgb)
        rgb_proj = self.rgb_projection(rgb)
        dem_proj = self.dem_projection(dem)
        average = 0.5 * (rgb_proj + dem_proj)

        if not use_residual:
            residual = torch.zeros_like(average)
            virtual = (
                self.concat_fusion(torch.cat([rgb_proj, dem_proj], dim=1))
                if self.concat_fusion is not None
                else average
            )
        else:
            sources = [rgb_proj, dem_proj]
            if self.prev_virtual_projection is not None:
                if prev_virtual is None:
                    raise ValueError("V_{n-1} is required by this stage")
                sources.append(self._align(self.prev_virtual_projection(prev_virtual), average))
            residual = self.residual_conv2(self.residual_conv1(torch.cat(sources, dim=1)))
            residual = self._align(residual, average)
            virtual = average + residual
        return virtual, {"average": average, "residual": residual}
