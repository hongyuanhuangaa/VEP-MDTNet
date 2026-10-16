"""VEP-MDTNet: dual-modal (RGB + DEM) terrace extraction network.

Paper component -> code:

    VEP-MDTNet                        model.VEPMDTNet, config.VEPMDTNetConfig
    RGB / DEM encoders (ResNet50)     encoder.DualEncoder.rgb_encoder / .dem_encoder
    DCN in RGB stages 3-5             dcn.DCNv2
    virtual branch, V_n               virtual_branch.VirtualBranchStage
    edge guidance response E_n        ep_skip_gate.EdgeGuidanceHead
    patch region response P_n         ep_skip_gate.PatchRegionHead
    E-P Skip Gate, eta_n              ep_skip_gate.EPSkipGate
    MSDA                              msda.MSDA
    MDT (encoder / decoder), F_MDT    mdt.MDT (msda.MDTEncoderLayer / MDTDecoderLayer)
    decoder D4..D1 (F_MDT fused at D3) decoder.Decoder
"""

from .config import VEPMDTNetConfig
from .losses import MainSegmentationLoss
from .model import VEPMDTNet, build_model
from .single_modality import SingleModalityVEPMDTNet
from .single_modality import build_model as build_single_modality_model

__all__ = [
    "VEPMDTNetConfig",
    "MainSegmentationLoss",
    "VEPMDTNet",
    "SingleModalityVEPMDTNet",
    "build_model",
    "build_single_modality_model",
]
