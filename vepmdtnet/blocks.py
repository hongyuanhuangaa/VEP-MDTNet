from __future__ import annotations

from torch import Tensor, nn


def group_count(channels: int, requested: int = 8) -> int:
    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(group_count(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_channels, out_channels, 3),
            ConvGNAct(out_channels, out_channels, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

