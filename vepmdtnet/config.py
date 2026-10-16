from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VFEMRTConfig:
    """Configuration and ablation switches for the recursive virtual model.

    ResNet stages 1..5 denote c1..c5 with output strides 2, 4, 8, 16 and 32.
    The two real streams are always independent; only the virtual stream fuses
    RGB and DEM information.
    """

    num_classes: int = 2
    input_size: Tuple[int, int] = (256, 256)
    image_channels: int = 3
    terrain_channels: int = 1
    pretrained: bool = True

    # Five-stage V-FuseNet-style recursive virtual stream.
    virtual_channels: Tuple[int, int, int, int, int] = (64, 128, 256, 384, 512)
    use_virtual_residual: bool = True
    virtual_residual_stages: Tuple[int, ...] = (1, 2, 3, 4, 5)
    use_previous_virtual: bool = True
    use_direct_concat_fusion: bool = False
    zero_init_virtual_residual: bool = True

    # Replace the 3x3 conv (conv2) in RGB ResNet bottlenecks with DCNv2 packs.
    use_rgb_dcn: bool = True
    rgb_dcn_stages: Tuple[int, ...] = (3, 4, 5)
    rgb_dcn_mode: str = "all"  # "all" or "last"
    dcn_initial_mask: float = 0.90
    force_fp32_dcn: bool = True

    # V1/V3/V5 EMRT refinement.  V3 is the 32x32 decoder query, while the
    # three projected levels provide shallow detail, middle structure and
    # deep semantics.  V1 is reduced from H/2 to H/4 before the transformer
    # encoder so it does not dominate the token count.
    use_msdeform_bridge: bool = True
    use_msdeform_encoder: bool = True
    use_msdeform_decoder: bool = True
    use_bridge_local_conv: bool = True
    use_emrt_decoder_refinement: bool = True
    emrt_v1_encoder_stride: int = 2
    emrt_refine_channels: int = 256
    emrt_inject_stage: int = 3
    # Retained only so legacy checkpoint dictionaries can be read and rejected
    # with a clear architecture message.  The V1/V3/V5 model does not use the
    # old gated V5 bridge residual.
    use_gated_emrt_residual: bool = False
    emrt_gate_init_bias: float = -2.0
    bridge_hidden_dim: int = 256
    bridge_channels: int = 512
    bridge_heads: int = 8
    bridge_points: int = 6
    bridge_encoder_layers: int = 4
    bridge_decoder_layers: int = 2
    bridge_ffn_dim: int = 1024
    bridge_dropout: float = 0.10

    # Virtual skip guidance and decoder switches.
    use_e_guidance: bool = True
    use_p_guidance: bool = True
    use_skip_gate: bool = True
    # The current final decoder uses bilinear 2x upsampling.
    use_carafe: bool = False
    use_final_refine: bool = True
    decoder_channels: Tuple[int, int, int, int] = (512, 256, 128, 64)

    # Kept explicit for ablations. The requested default uses one final loss.
    use_deep_supervision: bool = False
    return_aux: bool = False
    force_fp32_tail: bool = True

    def __post_init__(self) -> None:
        if self.image_channels <= 0 or self.terrain_channels <= 0:
            raise ValueError("image_channels and terrain_channels must be positive")
        if len(self.virtual_channels) != 5:
            raise ValueError("virtual_channels must contain five ResNet stages")
        if len(self.decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four decoder stages")
        valid_stages = {1, 2, 3, 4, 5}
        if not set(self.virtual_residual_stages).issubset(valid_stages):
            raise ValueError("virtual_residual_stages must be a subset of 1..5")
        if self.use_virtual_residual and self.use_direct_concat_fusion:
            raise ValueError(
                "use_virtual_residual and use_direct_concat_fusion are "
                "mutually exclusive"
            )
        if not set(self.rgb_dcn_stages).issubset({3, 4, 5}):
            raise ValueError("rgb_dcn_stages must be a subset of (3, 4, 5)")
        if self.rgb_dcn_mode not in {"all", "last"}:
            raise ValueError("rgb_dcn_mode must be 'all' or 'last'")
        if not 0.0 < self.dcn_initial_mask < 1.0:
            raise ValueError("dcn_initial_mask must lie strictly between 0 and 1")
        if self.bridge_hidden_dim % self.bridge_heads != 0:
            raise ValueError("bridge_hidden_dim must be divisible by bridge_heads")
        if self.bridge_hidden_dim % 4 != 0:
            raise ValueError("bridge_hidden_dim must be divisible by 4 for 2D sine encoding")
        if self.bridge_points <= 0:
            raise ValueError("bridge_points must be positive")
        if self.bridge_encoder_layers < 0 or self.bridge_decoder_layers < 0:
            raise ValueError("bridge layer counts cannot be negative")
        if self.emrt_v1_encoder_stride <= 0:
            raise ValueError("emrt_v1_encoder_stride must be positive")
        if self.emrt_refine_channels <= 0:
            raise ValueError("emrt_refine_channels must be positive")
        if self.emrt_inject_stage not in {1, 2, 3, 4}:
            raise ValueError("emrt_inject_stage must be one of decoder skip stages 1..4")
        if self.use_gated_emrt_residual:
            raise ValueError(
                "The legacy gated V5 bridge is incompatible with V1/V3/V5-Q3 EMRT"
            )
