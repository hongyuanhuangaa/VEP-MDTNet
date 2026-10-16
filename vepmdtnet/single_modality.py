"""True single-modality (RGB-only / DEM-only) variants of VEP-MDTNet.

The unused encoder and the RGB-DEM virtual-fusion branch are removed rather
than fed zeros.  A single RGB or DEM ResNet-50 is projected to the same
V1..V5 channel pyramid used by the full model, after which the MDT bridge,
E/P-guided skip decoder and final segmentation head are kept unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvGNAct
from .bridge import MultiscaleDeformableBridge
from .config import VFEMRTConfig
from .decoder import VirtualGuidedDecoder
from .encoder import PyramidFeatures, ResNet50Stream
from .guidance import VirtualSkipGuidance
from .heads import SegmentationHead


Modality = Literal["rgb", "dem"]
ModelOutput = Union[Tensor, Tuple[Tensor, Dict[str, object]]]


class SingleModalityEncoder(nn.Module):
    """One modality-specific ResNet-50 plus five 1x1 pyramid projections."""

    real_channels = (64, 256, 512, 1024, 2048)

    def __init__(self, config: VFEMRTConfig, modality: Modality) -> None:
        super().__init__()
        if modality not in {"rgb", "dem"}:
            raise ValueError("modality must be 'rgb' or 'dem'")
        self.modality = modality
        in_channels = (
            config.image_channels if modality == "rgb" else config.terrain_channels
        )
        # Preserve the corresponding full-model branch: DCN is used only by
        # the RGB branch; the DEM branch remains a standard ResNet-50.
        dcn_stages = (
            config.rgb_dcn_stages
            if modality == "rgb" and config.use_rgb_dcn
            else ()
        )
        self.stream = ResNet50Stream(
            in_channels=in_channels,
            pretrained=config.pretrained,
            dcn_stages=dcn_stages,
            dcn_mode=config.rgb_dcn_mode,
            dcn_initial_mask=config.dcn_initial_mask,
            force_fp32_dcn=config.force_fp32_dcn,
        )
        self.projections = nn.ModuleList(
            [
                ConvGNAct(real_channels, projected_channels, 1)
                for real_channels, projected_channels in zip(
                    self.real_channels, config.virtual_channels
                )
            ]
        )

    def forward(self, inputs: Tensor) -> Tuple[PyramidFeatures, PyramidFeatures]:
        source = self.stream(inputs)
        projected = [
            projection(feature)
            for projection, feature in zip(
                self.projections, source.as_tuple()
            )
        ]
        return source, PyramidFeatures(*projected)


class SingleModalityEMRTNet(nn.Module):
    """RGB-only or DEM-only equivalent of the current main89 decoder path."""

    def __init__(self, config: VFEMRTConfig, modality: Modality) -> None:
        super().__init__()
        if modality not in {"rgb", "dem"}:
            raise ValueError("modality must be 'rgb' or 'dem'")
        self.config = config
        self.modality = modality
        self.input_channels = (
            config.image_channels if modality == "rgb" else config.terrain_channels
        )
        self.encoder = SingleModalityEncoder(config, modality)

        # Keep main89's V1/V3/V5 EMRT design: V3 is the query and the
        # refinement is injected after the convolutional decoder fuses V3.
        self.bridge = MultiscaleDeformableBridge(
            (
                config.virtual_channels[0],
                config.virtual_channels[2],
                config.virtual_channels[4],
            ),
            config,
        )
        skip_channels = tuple(reversed(config.virtual_channels[:4]))
        self.guidance = VirtualSkipGuidance(
            skip_channels,
            use_e=config.use_e_guidance,
            use_p=config.use_p_guidance,
        )
        decoder_skip_stages = (4, 3, 2, 1)
        emrt_inject_index = decoder_skip_stages.index(config.emrt_inject_stage)
        emrt_refinement_channels = (
            config.emrt_refine_channels
            if config.use_msdeform_bridge
            and config.use_emrt_decoder_refinement
            else None
        )
        self.decoder = VirtualGuidedDecoder(
            in_channels=config.bridge_channels,
            skip_channels=skip_channels,
            decoder_channels=config.decoder_channels,
            use_carafe=config.use_carafe,
            use_skip_gate=config.use_skip_gate,
            emrt_refinement_channels=emrt_refinement_channels,
            emrt_inject_index=emrt_inject_index,
        )

        final_channels = config.decoder_channels[-1]
        self.final_refine = (
            nn.Sequential(
                ConvGNAct(
                    final_channels,
                    final_channels,
                    3,
                    groups=final_channels,
                ),
                ConvGNAct(final_channels, final_channels, 1),
            )
            if config.use_final_refine
            else nn.Identity()
        )
        self.classifier = SegmentationHead(final_channels, config.num_classes)
        self.deep_supervision_heads = (
            nn.ModuleList(
                [
                    SegmentationHead(channels, config.num_classes)
                    for channels in config.decoder_channels
                ]
            )
            if config.use_deep_supervision
            else nn.ModuleList()
        )

    def _fp32_context(self, device_type: str):
        if not self.config.force_fp32_tail:
            return nullcontext()
        return torch.autocast(device_type=device_type, enabled=False)

    @staticmethod
    def _float_pyramid(pyramid: PyramidFeatures) -> PyramidFeatures:
        return PyramidFeatures(*(value.float() for value in pyramid.as_tuple()))

    def forward(self, inputs: Tensor) -> ModelOutput:
        if inputs.ndim != 4:
            raise ValueError("single-modality input must be a BCHW tensor")
        if inputs.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} {self.modality.upper()} channels, "
                f"received {inputs.size(1)}"
            )

        source, pyramid = self.encoder(inputs)
        output_size = inputs.shape[-2:]
        with self._fp32_context(inputs.device.type):
            tail = (
                self._float_pyramid(pyramid)
                if self.config.force_fp32_tail
                else pyramid
            )
            v1, v2, v3, v4, v5 = tail.as_tuple()
            bridge, emrt_refinement, bridge_details = self.bridge(v1, v3, v5)
            skips = (v4, v3, v2, v1)
            guidance = self.guidance(skips)
            decoded, decoder_stages, decoder_details = self.decoder(
                bridge,
                skips,
                guidance["E"],
                guidance["P"],
                emrt_refinement=emrt_refinement,
            )
            decoded = F.interpolate(
                decoded,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            logits = self.classifier(self.final_refine(decoded))

            deep_logits: List[Tensor] = []
            if self.config.use_deep_supervision:
                deep_logits = [
                    F.interpolate(
                        head(feature),
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    for head, feature in zip(
                        self.deep_supervision_heads, decoder_stages
                    )
                ]

        if not self.config.return_aux:
            return logits
        return logits, {
            "modality": self.modality,
            "source": source,
            "pyramid": pyramid,
            "bridge": bridge_details,
            "guidance": guidance,
            "decoder": decoder_details,
            "decoder_stages": decoder_stages,
            "deep_logits": deep_logits,
        }


def build_model(
    config: Optional[VFEMRTConfig] = None,
    modality: Modality = "rgb",
) -> SingleModalityEMRTNet:
    return SingleModalityEMRTNet(config or VFEMRTConfig(), modality)


def build_rgb_model(
    config: Optional[VFEMRTConfig] = None,
) -> SingleModalityEMRTNet:
    return build_model(config, modality="rgb")


def build_dem_model(
    config: Optional[VFEMRTConfig] = None,
) -> SingleModalityEMRTNet:
    return build_model(config, modality="dem")

