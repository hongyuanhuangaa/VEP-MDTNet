from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from .config import VEPMDTNetConfig
from .dcn import replace_bottleneck_conv2
from .virtual_branch import VirtualBranchStage


@dataclass
class PyramidFeatures:
    c1: Tensor
    c2: Tensor
    c3: Tensor
    c4: Tensor
    c5: Tensor

    def as_tuple(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.c1, self.c2, self.c3, self.c4, self.c5


class ResNet50Encoder(nn.Module):
    """A modality-specific ResNet-50 encoder, optionally with DCN in stages 3-5."""

    stage_channels: Tuple[int, int, int, int, int] = (64, 256, 512, 1024, 2048)

    def __init__(
        self,
        in_channels: int,
        pretrained: bool,
        dcn_stages: Tuple[int, ...] = (),
        dcn_mode: str = "all",
        dcn_initial_mask: float = 0.90,
        force_fp32_dcn: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        if in_channels != 3:
            self._replace_input_conv(backbone, in_channels, pretrained)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        stage_modules = {3: self.layer2, 4: self.layer3, 5: self.layer4}
        self.dcn_replacements: Dict[int, int] = {}
        for stage_index in dcn_stages:
            self.dcn_replacements[stage_index] = replace_bottleneck_conv2(
                stage_modules[stage_index],
                dcn_mode,
                dcn_initial_mask,
                force_fp32=force_fp32_dcn,
            )

    @staticmethod
    def _replace_input_conv(
        backbone: nn.Module, in_channels: int, use_pretrained: bool
    ) -> None:
        old = backbone.conv1
        new = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if use_pretrained:
            with torch.no_grad():
                mean_weight = old.weight.mean(dim=1, keepdim=True)
                if in_channels == 1:
                    new.weight.copy_(mean_weight)
                elif in_channels >= 3:
                    new.weight[:, :3].copy_(old.weight)
                    if in_channels > 3:
                        new.weight[:, 3:].copy_(
                            mean_weight.repeat(1, in_channels - 3, 1, 1)
                        )
                else:
                    new.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        else:
            nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        backbone.conv1 = new

    def forward(self, x: Tensor) -> PyramidFeatures:
        c1 = self.layer0(x)
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return PyramidFeatures(c1, c2, c3, c4, c5)


class DualEncoder(nn.Module):
    """Independent RGB and DEM ResNet-50 encoders plus the five-stage virtual branch.

    Returns the RGB pyramid R1..R5, the DEM pyramid D1..D5 and the virtual
    features V1..V5.  Only the RGB encoder uses DCN.
    """

    stage_channels: Tuple[int, int, int, int, int] = (64, 256, 512, 1024, 2048)

    def __init__(self, config: VEPMDTNetConfig) -> None:
        super().__init__()
        dcn_stages = config.rgb_dcn_stages if config.use_rgb_dcn else ()
        self.rgb_encoder = ResNet50Encoder(
            config.image_channels,
            config.pretrained,
            dcn_stages=dcn_stages,
            dcn_mode=config.rgb_dcn_mode,
            dcn_initial_mask=config.dcn_initial_mask,
            force_fp32_dcn=config.force_fp32_dcn,
        )
        self.dem_encoder = ResNet50Encoder(config.terrain_channels, config.pretrained, dcn_stages=())
        self.virtual_branch_stages = set(config.virtual_branch_stages)
        self.use_virtual_branch = config.use_virtual_branch

        stages: List[nn.Module] = []
        for index, (channels, virtual_channels) in enumerate(
            zip(self.stage_channels, config.virtual_channels), start=1
        ):
            stages.append(
                VirtualBranchStage(
                    stage_index=index,
                    rgb_channels=channels,
                    dem_channels=channels,
                    virtual_channels=virtual_channels,
                    prev_virtual_channels=config.virtual_channels[index - 2] if index > 1 else None,
                    use_previous_virtual=config.use_previous_virtual,
                    use_concat_fusion=config.use_concat_fusion,
                    zero_init_residual=config.zero_init_virtual_residual,
                )
            )
        self.virtual_branch = nn.ModuleList(stages)

    def forward(
        self, image: Tensor, terrain: Tensor
    ) -> Tuple[PyramidFeatures, PyramidFeatures, PyramidFeatures, List[Dict[str, Tensor]]]:
        # Both encoders run without reading virtual features.
        rgb = self.rgb_encoder(image)
        dem = self.dem_encoder(terrain)

        virtual_levels: List[Tensor] = []
        auxiliary: List[Dict[str, Tensor]] = []
        for index, (stage, r_n, d_n) in enumerate(
            zip(self.virtual_branch, rgb.as_tuple(), dem.as_tuple()), start=1
        ):
            v_n, stage_aux = stage(
                rgb=r_n,
                dem=d_n,
                prev_virtual=virtual_levels[-1] if virtual_levels else None,
                use_residual=self.use_virtual_branch and index in self.virtual_branch_stages,
            )
            virtual_levels.append(v_n)
            auxiliary.append(stage_aux)
        return rgb, dem, PyramidFeatures(*virtual_levels), auxiliary
