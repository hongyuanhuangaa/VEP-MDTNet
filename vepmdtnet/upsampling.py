from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CARAFE(nn.Module):
    """Content-aware 2x upsampling retained from the current decoder."""

    def __init__(
        self,
        channels: int,
        scale_factor: int = 2,
        up_kernel: int = 5,
        compressed_channels: int = 64,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.compressor = nn.Conv2d(channels, compressed_channels, 1)
        self.kernel_encoder = nn.Conv2d(
            compressed_channels,
            scale_factor * scale_factor * up_kernel * up_kernel,
            3,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, _, _ = x.shape
        kernels = F.pixel_shuffle(
            self.kernel_encoder(self.compressor(x)), self.scale_factor
        )
        kernels = F.softmax(kernels, dim=1)
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        height, width = x_up.shape[-2:]
        patches = F.unfold(
            x_up, kernel_size=self.up_kernel, padding=self.up_kernel // 2
        ).reshape(
            batch,
            channels,
            self.up_kernel * self.up_kernel,
            height,
            width,
        )
        return torch.sum(patches * kernels.unsqueeze(1), dim=2)


class UpsampleLayer(nn.Module):
    def __init__(self, channels: int, use_carafe: bool) -> None:
        super().__init__()
        self.carafe = CARAFE(channels) if use_carafe else None

    def forward(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
        if self.carafe is not None:
            x = self.carafe(x)
        else:
            x = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x, size=target_size, mode="bilinear", align_corners=False
            )
        return x

