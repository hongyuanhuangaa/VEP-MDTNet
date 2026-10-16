"""True single-modality (RGB-only / DEM-only) variants of VEP-MDTNet.

The unused encoder and the virtual branch are removed rather than fed zeros.  A
single RGB or DEM ResNet-50 is projected to the same V1..V5 channel pyramid used
by the full model, after which the MDT, the E-P Skip Gate decoder and the
segmentation head are kept unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .blocks import ConvGNAct
from .config import VEPMDTNetConfig
from .decoder import Decoder
from .encoder import PyramidFeatures, ResNet50Encoder
from .ep_skip_gate import EPGuidance
from .heads import SegmentationHead
from .mdt import MDT


Modality = Literal["rgb", "dem"]
ModelOutput = Union[Tensor, Tuple[Tensor, Dict[str, object]]]


class SingleModalityEncoder(nn.Module):
    """One modality-specific ResNet-50 plus five 1x1 projections to V1..V5."""

    stage_channels = (64, 256, 512, 1024, 2048)

    def __init__(self, config: VEPMDTNetConfig, modality: Modality) -> None:
        super().__init__()
        if modality not in {"rgb", "dem"}:
            raise ValueError("modality must be 'rgb' or 'dem'")
        self.modality = modality
        in_channels = config.image_channels if modality == "rgb" else config.terrain_channels
        # As in the full model, DCN is used only by the RGB encoder.
        dcn_stages = config.rgb_dcn_stages if modality == "rgb" and config.use_rgb_dcn else ()
        self.resnet = ResNet50Encoder(
            in_channels=in_channels,
            pretrained=config.pretrained,
            dcn_stages=dcn_stages,
            dcn_mode=config.rgb_dcn_mode,
            dcn_initial_mask=config.dcn_initial_mask,
            force_fp32_dcn=config.force_fp32_dcn,
        )
        self.projections = nn.ModuleList(
            [ConvGNAct(c, v, 1) for c, v in zip(self.stage_channels, config.virtual_channels)]
        )

    def forward(self, inputs: Tensor) -> Tuple[PyramidFeatures, PyramidFeatures]:
        source = self.resnet(inputs)
        projected = [projection(f) for projection, f in zip(self.projections, source.as_tuple())]
        return source, PyramidFeatures(*projected)


class SingleModalityVEPMDTNet(nn.Module):
    """RGB-only or DEM-only network with the same MDT / E-P Skip Gate decoder path."""

    def __init__(self, config: VEPMDTNetConfig, modality: Modality) -> None:
        super().__init__()
        if modality not in {"rgb", "dem"}:
            raise ValueError("modality must be 'rgb' or 'dem'")
        self.config = config
        self.modality = modality
        self.input_channels = config.image_channels if modality == "rgb" else config.terrain_channels
        self.encoder = SingleModalityEncoder(config, modality)
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

    def forward(self, inputs: Tensor) -> ModelOutput:
        if inputs.ndim != 4:
            raise ValueError("single-modality input must be a BCHW tensor")
        if inputs.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} {self.modality.upper()} channels, received {inputs.size(1)}"
            )
        source, pyramid = self.encoder(inputs)
        output_size = inputs.shape[-2:]
        with self._fp32_context(inputs.device.type):
            tail = PyramidFeatures(*(v.float() for v in pyramid.as_tuple())) if self.config.force_fp32_tail else pyramid
            v1, v2, v3, v4, v5 = tail.as_tuple()
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
            "modality": self.modality,
            "source": source,
            "pyramid": pyramid,
            "mdt": mdt_details,
            "ep": ep,
            "decoder": decoder_details,
            "decoder_stages": decoder_stages,
        }


def build_model(config: Optional[VEPMDTNetConfig] = None, modality: Modality = "rgb") -> SingleModalityVEPMDTNet:
    return SingleModalityVEPMDTNet(config or VEPMDTNetConfig(), modality)
