"""Self-contained checkpoints: weights + everything needed to rebuild and preprocess."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from skeleton_action.models import build_model

FORMAT_VERSION = 1

REQUIRED_KEYS = (
    "format_version",
    "model_name",
    "model_params",
    "model_state",
    "num_classes",
    "num_frames",
    "num_joints",
    "in_channels",
    "class_names",
    "action_ids",
    "preprocess",
)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a checkpoint (write to temp file, then rename)."""
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"checkpoint payload missing keys: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load a checkpoint written by :func:`save_checkpoint` (``weights_only`` safe load)."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{path}: not a skeleton_action checkpoint (missing {missing})")
    if ckpt["format_version"] > FORMAT_VERSION:
        raise ValueError(
            f"{path}: checkpoint format {ckpt['format_version']} is newer than "
            f"supported {FORMAT_VERSION}"
        )
    return ckpt


def model_from_checkpoint(ckpt: dict, device: str | torch.device = "cpu") -> nn.Module:
    """Rebuild the architecture from checkpoint metadata and load the weights (eval mode)."""
    model = build_model(
        ckpt["model_name"],
        num_classes=ckpt["num_classes"],
        num_frames=ckpt["num_frames"],
        num_joints=ckpt["num_joints"],
        in_channels=ckpt["in_channels"],
        **ckpt["model_params"],
    )
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()
