from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VEPMDTNetConfig:
    """Architecture configuration and ablation switches of VEP-MDTNet.

    ResNet stages 1..5 denote c1..c5 with output strides 2, 4, 8, 16 and 32.
    The RGB and DEM encoders are always independent; only the virtual branch
    fuses RGB and DEM information.
    """

    num_classes: int = 2
    input_size: Tuple[int, int] = (256, 256)
    image_channels: int = 3
    terrain_channels: int = 1
    pretrained: bool = True

    # Virtual-branch fusion: V_n = (phi_R(R_n) + phi_D(D_n)) / 2 + F_n([...]).
    # With use_virtual_branch=False the stages fall back to the projected
    # average (use_concat_fusion=False) or a projected concatenation + 1x1
    # convolution (use_concat_fusion=True).
    virtual_channels: Tuple[int, int, int, int, int] = (64, 128, 256, 384, 512)
    use_virtual_branch: bool = True
    virtual_branch_stages: Tuple[int, ...] = (1, 2, 3, 4, 5)
    use_previous_virtual: bool = True
    use_concat_fusion: bool = False
    zero_init_virtual_residual: bool = True

    # DCN: replace the 3x3 conv (conv2) of the RGB ResNet bottlenecks with DCNv2.
    use_rgb_dcn: bool = True
    rgb_dcn_stages: Tuple[int, ...] = (3, 4, 5)
    rgb_dcn_mode: str = "all"  # "all" or "last"
    dcn_initial_mask: float = 0.90
    force_fp32_dcn: bool = True

    # MDT: multi-scale deformable Transformer over V1/V3/V5.  V3 provides the
    # decoder queries; V1 is reduced from H/2 to H/4 before the MDT encoder so it
    # does not dominate the token count.  F_MDT is fused into decoder stage D3.
    use_mdt: bool = True
    use_mdt_encoder: bool = True
    use_mdt_decoder: bool = True
    use_mdt_local_conv: bool = True
    use_mdt_injection: bool = True
    mdt_v1_stride: int = 2
    mdt_out_channels: int = 256
    mdt_inject_stage: int = 3
    mdt_dim: int = 256
    mdt_heads: int = 8
    mdt_points: int = 6
    mdt_encoder_layers: int = 4
    mdt_decoder_layers: int = 2
    mdt_ffn_dim: int = 1024
    mdt_dropout: float = 0.10
    # Width of the projected V5 that starts the convolutional decoder.
    decoder_in_channels: int = 512

    # E-P Skip Gate: edge guidance response E, patch region response P and the
    # gating coefficient eta of every skip connection.
    use_e_guidance: bool = True
    use_p_guidance: bool = True
    use_ep_skip_gate: bool = True
    use_final_refine: bool = True
    decoder_channels: Tuple[int, int, int, int] = (512, 256, 128, 64)

    return_aux: bool = False
    force_fp32_tail: bool = True

    def __post_init__(self) -> None:
        if self.image_channels <= 0 or self.terrain_channels <= 0:
            raise ValueError("image_channels and terrain_channels must be positive")
        if len(self.virtual_channels) != 5:
            raise ValueError("virtual_channels must contain five ResNet stages")
        if len(self.decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four decoder stages")
        if not set(self.virtual_branch_stages).issubset({1, 2, 3, 4, 5}):
            raise ValueError("virtual_branch_stages must be a subset of 1..5")
        if self.use_virtual_branch and self.use_concat_fusion:
            raise ValueError("use_virtual_branch and use_concat_fusion are mutually exclusive")
        if not set(self.rgb_dcn_stages).issubset({3, 4, 5}):
            raise ValueError("rgb_dcn_stages must be a subset of (3, 4, 5)")
        if self.rgb_dcn_mode not in {"all", "last"}:
            raise ValueError("rgb_dcn_mode must be 'all' or 'last'")
        if not 0.0 < self.dcn_initial_mask < 1.0:
            raise ValueError("dcn_initial_mask must lie strictly between 0 and 1")
        if self.mdt_dim % self.mdt_heads != 0:
            raise ValueError("mdt_dim must be divisible by mdt_heads")
        if self.mdt_dim % 4 != 0:
            raise ValueError("mdt_dim must be divisible by 4 for the 2D sine encoding")
        if self.mdt_points <= 0:
            raise ValueError("mdt_points must be positive")
        if self.mdt_encoder_layers < 0 or self.mdt_decoder_layers < 0:
            raise ValueError("MDT layer counts cannot be negative")
        if self.mdt_v1_stride <= 0:
            raise ValueError("mdt_v1_stride must be positive")
        if self.mdt_out_channels <= 0:
            raise ValueError("mdt_out_channels must be positive")
        if self.mdt_inject_stage not in {1, 2, 3, 4}:
            raise ValueError("mdt_inject_stage must be one of decoder skip stages 1..4")
