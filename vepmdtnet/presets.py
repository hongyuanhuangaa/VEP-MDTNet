"""Architecture presets of every model variant reported in the paper.

Each preset reproduces the switches stored in the checkpoint that produced the
corresponding row of the paper's tables:

    preset     paper row                               fusion     DCN  MDT  E  P  gate
    ---------  --------------------------------------  ---------  ---  ---  -  -  ----
    full       VEP-MDTNet (Full)                       virtual     1    1   1  1   1
    base       Base (Table 2)                          concat      0    0   0  0   0
    wo_vb      w/o Virtual Branch = Average fusion     average     1    1   1  1   1
    concat     Concatenation fusion (Table 5)          concat      1    1   1  1   1
    wo_mdt     w/o MDT                                 virtual     1    0   1  1   1
    wo_ep      w/o E-P Skip Gate                       virtual     1    1   0  0   0
    wo_e       w/o E (Table 7)                         virtual     1    1   0  1   1
    wo_p       w/o P (Table 7)                         virtual     1    1   1  0   1
    wo_dcn     w/o DCN                                 virtual     0    1   1  1   1
    rgb_only   RGB-only input (Table 4)                --          1    1   1  0   1
    dem_only   DEM-only input (Table 4)                --          0    1   1  0   1

"virtual" is the recursive virtual-branch fusion, "average" the projected RGB/DEM mean and
"concat" a projected concatenation followed by a learnable 1x1 Conv-GN-SiLU.  Disabled
modules fall back to: zero E / P maps, an ordinary (ungated) U-Net skip, no F_MDT (the
decoder still starts from the projected V5), and the plain ResNet-50 3x3 convolutions
instead of DCN.  The single-modality variants keep one encoder only (no virtual branch)
and were trained with E and the E-P Skip Gate but without P.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .config import VEPMDTNetConfig

#               fusion     dcn    mdt    e      p      gate   modality
_TABLE = {
    "full":     ("virtual", True,  True,  True,  True,  True,  "rgbdem"),
    "base":     ("concat",  False, False, False, False, False, "rgbdem"),
    "wo_vb":    ("average", True,  True,  True,  True,  True,  "rgbdem"),
    "concat":   ("concat",  True,  True,  True,  True,  True,  "rgbdem"),
    "wo_mdt":   ("virtual", True,  False, True,  True,  True,  "rgbdem"),
    "wo_ep":    ("virtual", True,  True,  False, False, False, "rgbdem"),
    "wo_e":     ("virtual", True,  True,  False, True,  True,  "rgbdem"),
    "wo_p":     ("virtual", True,  True,  True,  False, True,  "rgbdem"),
    "wo_dcn":   ("virtual", False, True,  True,  True,  True,  "rgbdem"),
    "rgb_only": ("none",    True,  True,  True,  False, True,  "rgb"),
    "dem_only": ("none",    False, True,  True,  False, True,  "dem"),
}
PRESETS: Tuple[str, ...] = tuple(_TABLE)


def preset_switches(name: str) -> Dict[str, object]:
    if name not in _TABLE:
        raise KeyError(f"Unknown preset '{name}'. Choose from: {', '.join(PRESETS)}")
    fusion, dcn, mdt, e, p, gate, modality = _TABLE[name]
    return {
        "fusion": fusion, "dcn": dcn, "mdt": mdt, "e": e, "p": p, "gate": gate,
        "modality": modality,
    }


def make_config(
    name: str,
    *,
    pretrained: bool = True,
    input_size: Tuple[int, int] = (256, 256),
    image_channels: int = 3,
    terrain_channels: int = 1,
    num_classes: int = 2,
) -> Tuple[VEPMDTNetConfig, str]:
    """Return ``(config, modality)`` for a preset; modality is rgbdem, rgb or dem."""
    s = preset_switches(name)
    virtual = s["fusion"] == "virtual"
    dcn = bool(s["dcn"]) and s["modality"] != "dem"   # DCN only exists in the RGB encoder
    mdt = bool(s["mdt"])
    config = VEPMDTNetConfig(
        num_classes=num_classes,
        input_size=tuple(input_size),
        image_channels=image_channels,
        terrain_channels=terrain_channels,
        pretrained=pretrained,
        virtual_channels=(64, 128, 256, 384, 512),
        use_virtual_branch=virtual,
        virtual_branch_stages=(1, 2, 3, 4, 5) if virtual else (),
        use_previous_virtual=virtual,
        use_concat_fusion=s["fusion"] == "concat",
        use_rgb_dcn=dcn,
        rgb_dcn_stages=(3, 4, 5) if dcn else (),
        rgb_dcn_mode="all",
        force_fp32_dcn=dcn,
        use_mdt=mdt,
        use_mdt_encoder=mdt,
        use_mdt_decoder=mdt,
        use_mdt_local_conv=mdt,
        use_mdt_injection=mdt,
        mdt_v1_stride=2,
        mdt_out_channels=256,
        mdt_inject_stage=3,
        mdt_dim=256,
        mdt_heads=8,
        mdt_points=6,
        mdt_encoder_layers=4,
        mdt_decoder_layers=2,
        mdt_ffn_dim=1024,
        use_e_guidance=bool(s["e"]),
        use_p_guidance=bool(s["p"]),
        use_ep_skip_gate=bool(s["gate"]),
        use_final_refine=True,
        return_aux=False,
        force_fp32_tail=True,
    )
    return config, str(s["modality"])
