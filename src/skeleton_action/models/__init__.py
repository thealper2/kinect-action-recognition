"""Model registry: build any model from (name, params) as stored in checkpoints."""

from __future__ import annotations

from torch import nn

from skeleton_action.models.lstm import LSTMClassifier
from skeleton_action.models.mlp import MLPClassifier
from skeleton_action.models.stgcn import STGCN

MODEL_REGISTRY = {"mlp": MLPClassifier, "lstm": LSTMClassifier, "stgcn": STGCN}


def build_model(
    name: str, num_classes: int, num_frames: int, num_joints: int, in_channels: int, **params
) -> nn.Module:
    """Instantiate a registered model. Data-dependent sizes are passed explicitly."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    common = {"num_classes": num_classes, "num_joints": num_joints, "in_channels": in_channels}
    if name == "mlp":
        common["num_frames"] = num_frames
    return MODEL_REGISTRY[name](**common, **params)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


__all__ = [
    "MODEL_REGISTRY",
    "STGCN",
    "LSTMClassifier",
    "MLPClassifier",
    "build_model",
    "count_parameters",
]
