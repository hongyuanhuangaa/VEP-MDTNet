from __future__ import annotations

from torch import nn


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(nn.Conv2d(in_channels, num_classes, 1))

