"""Checkpoint-backed predictor: raw/preprocessed skeleton sequence -> class scores.

Everything needed (architecture, preprocessing, class names) comes from the checkpoint;
no training config or notebook state is required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from skeleton_action.constants import NUM_JOINTS, J
from skeleton_action.preprocessing.transforms import (
    PreprocessConfig,
    joint_validity,
    preprocess_sequence,
)
from skeleton_action.training.checkpoint import load_checkpoint, model_from_checkpoint

# Joints whose presence is required for a meaningful normalisation.
CORE_JOINTS = (J.NECK, J.TORSO, J.R_SHOULDER, J.L_SHOULDER, J.R_HIP, J.L_HIP)


@dataclass
class Prediction:
    valid: bool
    reason: str = ""
    label: str | None = None
    class_index: int | None = None
    confidence: float = 0.0
    topk: list[tuple[str, float]] = field(default_factory=list)
    probs: np.ndarray | None = None
    preprocess_ms: float = 0.0
    forward_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "label": self.label,
            "class_index": self.class_index,
            "confidence": round(self.confidence, 4),
            "topk": [(n, round(p, 4)) for n, p in self.topk],
            "preprocess_ms": round(self.preprocess_ms, 3),
            "forward_ms": round(self.forward_ms, 3),
        }


@dataclass
class ValidityPolicy:
    """Minimum input quality before a prediction is produced at all."""

    min_frames: int = 10
    min_valid_joint_fraction: float = 0.8  # over all (frame, joint) entries
    min_core_joint_fraction: float = 0.9  # per frame, CORE_JOINTS all valid


class Predictor:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cpu",
        policy: ValidityPolicy | None = None,
    ) -> None:
        self.ckpt_path = Path(checkpoint)
        self.ckpt = load_checkpoint(checkpoint, map_location="cpu")
        self.device = torch.device(device)
        self.model = model_from_checkpoint(self.ckpt, self.device)
        self.pre = PreprocessConfig.from_dict(self.ckpt["preprocess"])
        self.class_names: list[str] = list(self.ckpt["class_names"])
        self.policy = policy or ValidityPolicy()

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.pre.num_frames, self.ckpt["num_joints"], self.ckpt["in_channels"])

    def check_sequence(self, seq: np.ndarray, mask: np.ndarray) -> str:
        """Return an empty string if the raw sequence is usable, else the reason."""
        if seq.ndim != 3 or seq.shape[1:] != (NUM_JOINTS, 3):
            return f"bad shape {seq.shape}, expected [T, {NUM_JOINTS}, 3]"
        p = self.policy
        if seq.shape[0] < p.min_frames:
            return f"only {seq.shape[0]} frames (< {p.min_frames})"
        frac = float(mask.mean())
        if frac < p.min_valid_joint_fraction:
            return f"valid joint fraction {frac:.2f} < {p.min_valid_joint_fraction}"
        core = float(mask[:, list(CORE_JOINTS)].all(axis=1).mean())
        if core < p.min_core_joint_fraction:
            return f"core joints present in {core:.2f} of frames < {p.min_core_joint_fraction}"
        return ""

    @torch.no_grad()
    def _forward(self, x: np.ndarray, topk: int) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(self.device)
        probs = torch.softmax(self.model(xt).float(), dim=-1)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return probs.cpu().numpy(), (time.perf_counter() - t0) * 1e3

    def _result(self, probs: np.ndarray, topk: int, pre_ms: float, fwd_ms: float) -> Prediction:
        order = np.argsort(-probs)[:topk]
        c = int(order[0])
        return Prediction(
            True,
            "",
            self.class_names[c],
            c,
            float(probs[c]),
            [(self.class_names[i], float(probs[i])) for i in order],
            probs,
            pre_ms,
            fwd_ms,
        )

    def predict_raw(
        self, seq_mm: np.ndarray, mask: np.ndarray | None = None, topk: int = 5
    ) -> Prediction:
        """Predict from a raw KARD-convention sequence ``[T, 15, 3]`` (mm)."""
        seq_mm = np.asarray(seq_mm, dtype=np.float32)
        t0 = time.perf_counter()
        if mask is None:
            mask = joint_validity(seq_mm) if seq_mm.ndim == 3 else np.zeros((0, 0), bool)
        reason = self.check_sequence(seq_mm, mask)
        if reason:
            return Prediction(False, reason)
        x = preprocess_sequence(seq_mm, self.pre, mask)
        pre_ms = (time.perf_counter() - t0) * 1e3
        if not np.isfinite(x).all():
            return Prediction(False, "non-finite features after preprocessing")
        probs, fwd_ms = self._forward(x, topk)
        return self._result(probs, topk, pre_ms, fwd_ms)

    def predict_features(self, x: np.ndarray, topk: int = 5) -> Prediction:
        """Predict from an already preprocessed ``[T, J, C]`` tensor."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape != self.input_shape:
            return Prediction(False, f"feature shape {x.shape} != {self.input_shape}")
        if not np.isfinite(x).all():
            return Prediction(False, "non-finite features")
        probs, fwd_ms = self._forward(x, topk)
        return self._result(probs, topk, 0.0, fwd_ms)
