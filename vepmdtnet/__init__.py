"""VEP-MDTNet: dual-modal (RGB + DEM) terrace extraction network.

Code names used inside the package (kept so that existing checkpoints load):

    paper term                         code
    ---------------------------------  -------------------------------------------
    VEP-MDTNet                         RecursiveVirtualEMRTNet / VFEMRTConfig
    virtual-branch fusion              virtual_branch.RecursiveVirtualStage
    DCN in RGB stages 3-5              deformable.ModulatedDeformConvPack
    edge (E) / patch (P) guidance      guidance.VirtualSkipGuidance
    E-P Skip Gate                      decoder.GuidedSkipDecoderBlock
    MDT (multi-scale deformable        bridge.MultiscaleDeformableBridge
         Transformer)                  ("EMRT" / "msdeform" in config fields)
"""

from .config import VFEMRTConfig
from .losses import MainSegmentationLoss
from .model import RecursiveVirtualEMRTNet, build_model
from .single_modality import SingleModalityEMRTNet
from .single_modality import build_model as build_single_modality_model

__all__ = [
    "VFEMRTConfig",
    "MainSegmentationLoss",
    "RecursiveVirtualEMRTNet",
    "SingleModalityEMRTNet",
    "build_model",
    "build_single_modality_model",
]
