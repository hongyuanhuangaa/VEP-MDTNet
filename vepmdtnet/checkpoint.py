"""Checkpoint loading that restores the exact architecture stored at training time.

Checkpoints written during the study used the development names of the code
(package ``vf_emrt``, config fields such as ``use_msdeform_bridge``, modules such
as ``bridge.`` / ``guidance.e_heads.``).  They are converted to the current names
on load, so both old and new checkpoints work with the same code.
"""

from __future__ import annotations

import re
import sys
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from .config import VEPMDTNetConfig
from .model import build_model
from .presets import make_config
from .single_modality import build_model as build_single_modality_model

_TUPLE_FIELDS = (
    "input_size", "virtual_channels", "virtual_branch_stages", "rgb_dcn_stages",
    "decoder_channels",
)

# development config field -> current field
LEGACY_CONFIG_FIELDS = {
    "use_virtual_residual": "use_virtual_branch",
    "virtual_residual_stages": "virtual_branch_stages",
    "use_direct_concat_fusion": "use_concat_fusion",
    "use_msdeform_bridge": "use_mdt",
    "use_msdeform_encoder": "use_mdt_encoder",
    "use_msdeform_decoder": "use_mdt_decoder",
    "use_bridge_local_conv": "use_mdt_local_conv",
    "use_emrt_decoder_refinement": "use_mdt_injection",
    "emrt_v1_encoder_stride": "mdt_v1_stride",
    "emrt_refine_channels": "mdt_out_channels",
    "emrt_inject_stage": "mdt_inject_stage",
    "bridge_hidden_dim": "mdt_dim",
    "bridge_channels": "decoder_in_channels",
    "bridge_heads": "mdt_heads",
    "bridge_points": "mdt_points",
    "bridge_encoder_layers": "mdt_encoder_layers",
    "bridge_decoder_layers": "mdt_decoder_layers",
    "bridge_ffn_dim": "mdt_ffn_dim",
    "bridge_dropout": "mdt_dropout",
    "use_skip_gate": "use_ep_skip_gate",
}
# development options that are not part of VEP-MDTNet; they must be off in a checkpoint
UNSUPPORTED_LEGACY_OPTIONS = ("use_carafe", "use_deep_supervision", "use_gated_emrt_residual")

# development parameter-name prefix -> current prefix (applied in this order)
_LEGACY_PREFIXES = (
    ("encoder.rgb.", "encoder.rgb_encoder."),
    ("encoder.terrain.", "encoder.dem_encoder."),
    ("encoder.virtual_stages.", "encoder.virtual_branch."),
    ("encoder.stream.", "encoder.resnet."),
    ("bridge.simple_output.", "mdt.v5_projection."),
    ("bridge.projections.", "mdt.input_projections."),
    ("bridge.refine_output.", "mdt.output_conv."),
    ("bridge.", "mdt."),
    ("guidance.e_heads.", "ep_guidance.edge_heads."),
    ("guidance.p_heads.", "ep_guidance.patch_heads."),
    ("decoder.emrt_fusion.", "decoder.mdt_fusion."),
)
_LEGACY_PATTERNS = (
    (re.compile(r"^(encoder\.virtual_branch\.\d+\.)current_rgb_projection\."), r"\1rgb_projection."),
    (re.compile(r"^(encoder\.virtual_branch\.\d+\.)current_dem_projection\."), r"\1dem_projection."),
    (re.compile(r"^(encoder\.virtual_branch\.\d+\.)previous_virtual_projection\."), r"\1prev_virtual_projection."),
    (re.compile(r"^(encoder\.virtual_branch\.\d+\.)direct_concat_fusion\."), r"\1concat_fusion."),
    (re.compile(r"^(decoder\.blocks\.\d+\.)skip_gate\."), r"\1gate."),
)


class _LegacyConfig:
    """Stand-in for the pickled development config class ``vf_emrt.config.VFEMRTConfig``."""


def _register_legacy_modules() -> None:
    if "vf_emrt.config" in sys.modules:
        return
    root = types.ModuleType("vf_emrt")
    config = types.ModuleType("vf_emrt.config")
    config.VFEMRTConfig = _LegacyConfig
    root.config = config
    sys.modules.setdefault("vf_emrt", root)
    sys.modules.setdefault("vf_emrt.config", config)


def load_checkpoint(path: Path) -> Dict[str, object]:
    """Load on CPU; mmap keeps optimizer tensors of large training checkpoints on disk."""
    _register_legacy_modules()
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:          # torch < 2.1 has no mmap / weights_only arguments
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} does not contain a model_state_dict")
    return checkpoint


def config_values(config: object) -> Dict[str, object]:
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config):
        try:
            return asdict(config)
        except (AttributeError, TypeError):
            pass
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {}


def upgrade_config_values(values: Mapping[str, object]) -> Dict[str, object]:
    """Rename development config fields; reject options that VEP-MDTNet does not have."""
    for option in UNSUPPORTED_LEGACY_OPTIONS:
        if values.get(option):
            raise ValueError(f"Checkpoint uses '{option}', which is not part of VEP-MDTNet")
    return {LEGACY_CONFIG_FIELDS.get(key, key): value for key, value in values.items()}


def upgrade_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename development parameter names to the current module names."""
    upgraded = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        for old, new in _LEGACY_PREFIXES:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        for pattern, replacement in _LEGACY_PATTERNS:
            key = pattern.sub(replacement, key)
        upgraded[key] = value
    return upgraded


def config_from_checkpoint(checkpoint: Mapping[str, object]) -> VEPMDTNetConfig:
    """Stored model_config merged onto the full-model defaults (for fields added later)."""
    stored = config_values(checkpoint.get("model_config"))
    if not stored:
        raise ValueError("Checkpoint has no model_config; build the model from a preset instead")
    stored = upgrade_config_values(stored)
    values = asdict(make_config("full", pretrained=False)[0])
    values.update({key: value for key, value in stored.items() if key in values})
    for key in _TUPLE_FIELDS:
        values[key] = tuple(values[key])
    values["pretrained"] = False      # weights come from the checkpoint
    values["return_aux"] = False
    return VEPMDTNetConfig(**values)


def modality_from_checkpoint(checkpoint: Mapping[str, object]) -> str:
    modality = str(checkpoint.get("input_modality") or "rgbdem").lower()
    if modality not in ("rgbdem", "rgb", "dem"):
        raise ValueError(f"Unknown input_modality '{modality}' in checkpoint")
    return modality


def build(config: VEPMDTNetConfig, modality: str) -> nn.Module:
    if modality == "rgbdem":
        return build_model(config)
    return build_single_modality_model(config, modality=modality)


def load_model(
    path: Path,
    device: Optional[torch.device] = None,
    return_aux: bool = False,
) -> Tuple[nn.Module, VEPMDTNetConfig, str, Dict[str, object]]:
    """Return ``(model in eval mode, config, modality, checkpoint metadata)``.

    The metadata dict holds everything except the model/optimizer/scaler states, so the
    large mmap'd tensors can be released by the caller.
    """
    checkpoint = load_checkpoint(path)
    config = config_from_checkpoint(checkpoint)
    config.return_aux = return_aux
    modality = modality_from_checkpoint(checkpoint)
    model = build(config, modality)
    model.load_state_dict(upgrade_state_dict(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    if device is not None:
        model.to(device)
    meta = {
        key: value for key, value in checkpoint.items()
        if key not in ("model_state_dict", "optimizer_state_dict", "scaler_state_dict")
    }
    del checkpoint
    return model, config, modality, meta


def model_inputs(batch: Mapping[str, torch.Tensor], modality: str, device: torch.device):
    """Positional model inputs for a dataset batch."""
    if modality == "rgbdem":
        return (
            batch["image"].to(device, non_blocking=True),
            batch["terrain"].to(device, non_blocking=True),
        )
    key = "image" if modality == "rgb" else "terrain"
    return (batch[key].to(device, non_blocking=True),)


def logits_to_prediction(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(1) == 1:
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)
    return torch.argmax(logits, dim=1)
