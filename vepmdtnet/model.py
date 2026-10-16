from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvGNAct
from .bridge import MultiscaleDeformableBridge
from .config import VFEMRTConfig
from .decoder import VirtualGuidedDecoder
from .encoder import DualIndependentRecursiveEncoder, PyramidFeatures
from .guidance import VirtualSkipGuidance
from .heads import SegmentationHead


ModelOutput = Union[Tensor, Tuple[Tensor, Dict[str, object]]]


class RecursiveVirtualEMRTNet(nn.Module):
    """Dual independent ResNets with a recursive virtual fusion pathway.

    RGB and DEM never receive a virtual feature.  V1..V5 are formed only in
    the third (virtual) pathway. V5 starts the convolutional decoder, V1/V3/V5
    feed the EMRT refinement, and V4..V1 provide the four skip connections.
    """

    def __init__(self, config: VFEMRTConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DualIndependentRecursiveEncoder(config)

        # EMRT input order is V1, V3, V5; its query is V3. The convolutional
        # decoder remains V5 -> V4 -> V3 -> V2 -> V1.
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
        emrt_inject_index = decoder_skip_stages.index(
            config.emrt_inject_stage
        )
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

        # These heads are inactive in the requested final-loss-only setup.
        # Turning them on only exposes scale logits in ``aux``; the caller must
        # explicitly define a deep-supervision loss before they affect training.
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

    def forward(self, image: Tensor, terrain: Tensor) -> ModelOutput:
        if image.dim() != 4 or terrain.dim() != 4:
            raise ValueError("image and terrain must both be BCHW tensors")
        if image.size(0) != terrain.size(0):
            raise ValueError("image and terrain batch sizes must match")
        if image.size(1) != self.config.image_channels:
            raise ValueError(
                f"Expected {self.config.image_channels} RGB channels, "
                f"received {image.size(1)}"
            )
        if terrain.size(1) != self.config.terrain_channels:
            raise ValueError(
                f"Expected {self.config.terrain_channels} terrain channels, "
                f"received {terrain.size(1)}"
            )
        if terrain.shape[-2:] != image.shape[-2:]:
            terrain = F.interpolate(
                terrain,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        rgb, dem, virtual, virtual_details = self.encoder(image, terrain)
        output_size = image.shape[-2:]

        with self._fp32_context(image.device.type):
            if self.config.force_fp32_tail:
                virtual_tail = self._float_pyramid(virtual)
            else:
                virtual_tail = virtual

            v1, v2, v3, v4, v5 = virtual_tail.as_tuple()
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

        auxiliary: Dict[str, object] = {
            "rgb": rgb,
            "dem": dem,
            "virtual": virtual,
            "virtual_details": virtual_details,
            "bridge": bridge_details,
            "guidance": guidance,
            "decoder": decoder_details,
            "decoder_stages": decoder_stages,
            "deep_logits": deep_logits,
        }
        return logits, auxiliary


def build_model(config: VFEMRTConfig | None = None) -> RecursiveVirtualEMRTNet:
    return RecursiveVirtualEMRTNet(config or VFEMRTConfig())
