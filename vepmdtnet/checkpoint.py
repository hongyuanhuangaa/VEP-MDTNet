"""Checkpoint loading that restores the exact architecture stored at training time."""

from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from .config import VFEMRTConfig
from .model import build_model
from .presets import make_config
from .single_modality import build_model as build_single_modality_model

_SUBMODULES = (
    "blocks", "bridge", "config", "decoder", "deformable", "encoder", "guidance",
    "heads", "losses", "model", "msdeform_transformer", "upsampling", "virtual_branch",
)
_TUPLE_FIELDS = (
    "input_size", "virtual_channels", "virtual_residual_stages", "rgb_dcn_stages",
    "decoder_channels",
)


def register_legacy_modules() -> None:
    """Checkpoints written during development pickled the config as ``vf_emrt.config``.

    Aliasing the old package name lets ``torch.load`` resolve those objects to this package.
    """
    package = importlib.import_module(__package__)
    sys.modules.setdefault("vf_emrt", package)
    for name in _SUBMODULES:
        sys.modules.setdefault(f"vf_emrt.{name}", importlib.import_module(f"{__package__}.{name}"))


def load_checkpoint(path: Path) -> Dict[str, object]:
    """Load on CPU; mmap keeps optimizer tensors of large training checkpoints on disk."""
    register_legacy_modules()
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


def config_from_checkpoint(checkpoint: Mapping[str, object]) -> VFEMRTConfig:
    """Stored model_config merged onto the full-model defaults (for fields added later)."""
    stored = config_values(checkpoint.get("model_config"))
    if not stored:
        raise ValueError("Checkpoint has no model_config; build the model from a preset instead")
    values = asdict(make_config("full", pretrained=False)[0])
    valid = VFEMRTConfig.__dataclass_fields__.keys()
    values.update({key: value for key, value in stored.items() if key in valid})
    for key in _TUPLE_FIELDS:
        values[key] = tuple(values[key])
    values["pretrained"] = False      # weights come from the checkpoint
    values["return_aux"] = False
    values["use_deep_supervision"] = False
    return VFEMRTConfig(**values)


def modality_from_checkpoint(checkpoint: Mapping[str, object]) -> str:
    modality = str(checkpoint.get("input_modality") or "rgbdem").lower()
    if modality not in ("rgbdem", "rgb", "dem"):
        raise ValueError(f"Unknown input_modality '{modality}' in checkpoint")
    return modality


def build(config: VFEMRTConfig, modality: str) -> nn.Module:
    if modality == "rgbdem":
        return build_model(config)
    return build_single_modality_model(config, modality=modality)


def load_model(
    path: Path,
    device: Optional[torch.device] = None,
    return_aux: bool = False,
) -> Tuple[nn.Module, VFEMRTConfig, str, Dict[str, object]]:
    """Return ``(model in eval mode, config, modality, checkpoint metadata)``.

    The metadata dict holds everything except the model/optimizer/scaler states, so the
    large mmap'd tensors can be released by the caller.
    """
    checkpoint = load_checkpoint(path)
    config = config_from_checkpoint(checkpoint)
    config.return_aux = return_aux
    modality = modality_from_checkpoint(checkpoint)
    model = build(config, modality)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
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
