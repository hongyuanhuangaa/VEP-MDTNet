from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvGNAct
from .config import VEPMDTNetConfig
from .decoder import Decoder
from .encoder import DualEncoder, PyramidFeatures
from .ep_skip_gate import EPGuidance
from .heads import SegmentationHead
from .mdt import MDT


ModelOutput = Union[Tensor, Tuple[Tensor, Dict[str, object]]]


class VEPMDTNet(nn.Module):
    """VEP-MDTNet: RGB and DEM encoders, virtual-branch fusion, E-P Skip Gate and MDT.

    V5 starts the convolutional decoder, V1/V3/V5 feed the MDT whose output F_MDT
    is fused at decoder stage D3, and V4..V1 provide the four gated skip connections.
    """

    def __init__(self, config: VEPMDTNetConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DualEncoder(config)
        self.mdt = MDT(
            (config.virtual_channels[0], config.virtual_channels[2], config.virtual_channels[4]),
            config,
        )
        skip_channels = tuple(reversed(config.virtual_channels[:4]))
        self.ep_guidance = EPGuidance(skip_channels, use_e=config.use_e_guidance, use_p=config.use_p_guidance)
        decoder_skip_stages = (4, 3, 2, 1)
        self.decoder = Decoder(
            in_channels=config.decoder_in_channels,
            skip_channels=skip_channels,
            decoder_channels=config.decoder_channels,
            use_gate=config.use_ep_skip_gate,
            mdt_channels=config.mdt_out_channels if config.use_mdt and config.use_mdt_injection else None,
            mdt_inject_index=decoder_skip_stages.index(config.mdt_inject_stage),
        )
        final_channels = config.decoder_channels[-1]
        self.final_refine = (
            nn.Sequential(
                ConvGNAct(final_channels, final_channels, 3, groups=final_channels),
                ConvGNAct(final_channels, final_channels, 1),
            )
            if config.use_final_refine
            else nn.Identity()
        )
        self.classifier = SegmentationHead(final_channels, config.num_classes)

    def _fp32_context(self, device_type: str):
        if not self.config.force_fp32_tail:
            return nullcontext()
        return torch.autocast(device_type=device_type, enabled=False)

    def forward(self, image: Tensor, terrain: Tensor) -> ModelOutput:
        if image.dim() != 4 or terrain.dim() != 4:
            raise ValueError("image and terrain must both be BCHW tensors")
        if image.size(0) != terrain.size(0):
            raise ValueError("image and terrain batch sizes must match")
        if image.size(1) != self.config.image_channels:
            raise ValueError(f"Expected {self.config.image_channels} RGB channels, received {image.size(1)}")
        if terrain.size(1) != self.config.terrain_channels:
            raise ValueError(f"Expected {self.config.terrain_channels} DEM channels, received {terrain.size(1)}")
        if terrain.shape[-2:] != image.shape[-2:]:
            terrain = F.interpolate(terrain, size=image.shape[-2:], mode="bilinear", align_corners=False)

        rgb, dem, virtual, virtual_details = self.encoder(image, terrain)
        output_size = image.shape[-2:]
        with self._fp32_context(image.device.type):
            if self.config.force_fp32_tail:
                virtual_tail = PyramidFeatures(*(v.float() for v in virtual.as_tuple()))
            else:
                virtual_tail = virtual
            v1, v2, v3, v4, v5 = virtual_tail.as_tuple()
            decoder_input, f_mdt, mdt_details = self.mdt(v1, v3, v5)
            skips = (v4, v3, v2, v1)
            ep = self.ep_guidance(skips)
            decoded, decoder_stages, decoder_details = self.decoder(
                decoder_input, skips, ep["E"], ep["P"], f_mdt=f_mdt
            )
            decoded = F.interpolate(decoded, size=output_size, mode="bilinear", align_corners=False)
            logits = self.classifier(self.final_refine(decoded))

        if not self.config.return_aux:
            return logits
        return logits, {
            "rgb": rgb,
            "dem": dem,
            "virtual": virtual,
            "virtual_details": virtual_details,
            "mdt": mdt_details,
            "ep": ep,
            "decoder": decoder_details,
            "decoder_stages": decoder_stages,
        }


def build_model(config: Optional[VEPMDTNetConfig] = None) -> VEPMDTNet:
    return VEPMDTNet(config or VEPMDTNetConfig())
