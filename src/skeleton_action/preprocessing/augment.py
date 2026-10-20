"""Training-only skeleton augmentations (never applied to validation/test data).

All magnitudes are small and label-preserving for KARD: a few degrees of yaw (the
subjects roughly face the sensor), +-10 % body scale, small global translation, sensor-like
jitter and a mild temporal crop. Mirroring is deliberately *not* offered because several
KARD actions are performed with a specific hand.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from skeleton_action.preprocessing.transforms import yaw_rotation


@dataclass
class AugmentConfig:
    """Augmentation configuration. ``enabled=False`` disables everything.

    Attributes:
        temporal_crop_min: keep a random contiguous fraction in ``[min, 1]`` of the raw
            frames before resampling (0 < min <= 1; 1 disables).
        yaw_deg: uniform random rotation about the vertical axis in ``[-yaw, yaw]``.
        pitch_deg: uniform random rotation about the X axis (camera tilt).
        scale_range: uniform isotropic scale factor range.
        translate_std: std of a global translation (in normalised units).
        jitter_std: std of i.i.d. Gaussian noise per coordinate (normalised units).
        joint_dropout_p: probability of dropping a joint for a contiguous segment (only
            for missing-joint robustness studies; default 0).
    """

    enabled: bool = True
    temporal_crop_min: float = 0.85
    yaw_deg: float = 15.0
    pitch_deg: float = 5.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    translate_std: float = 0.05
    jitter_std: float = 0.01
    joint_dropout_p: float = 0.0

    def __post_init__(self) -> None:
        self.scale_range = tuple(self.scale_range)  # type: ignore[assignment]
        if not 0.0 < self.temporal_crop_min <= 1.0:
            raise ValueError("temporal_crop_min must be in (0, 1]")
        if self.scale_range[0] <= 0 or self.scale_range[0] > self.scale_range[1]:
            raise ValueError("invalid scale_range")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["scale_range"] = list(self.scale_range)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> AugmentConfig:
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown augmentation keys: {sorted(unknown)}")
        return cls(**d)


def temporal_crop(
    seq: np.ndarray, mask: np.ndarray, min_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random contiguous crop keeping a fraction in ``[min_frac, 1]`` of the frames."""
    t = seq.shape[0]
    if min_frac >= 1.0 or t < 4:
        return seq, mask
    keep = max(2, round(t * rng.uniform(min_frac, 1.0)))
    start = int(rng.integers(0, t - keep + 1))
    return seq[start : start + keep], mask[start : start + keep]


def joint_dropout(mask: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Mark random joints invalid over a random contiguous segment (returns a new mask)."""
    if p <= 0:
        return mask
    mask = mask.copy()
    t, nj = mask.shape
    for j in range(nj):
        if rng.random() < p:
            length = int(rng.integers(1, max(2, t // 2)))
            start = int(rng.integers(0, t - length + 1))
            mask[start : start + length, j] = False
    return mask


def _pitch_rotation(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def make_spatial_aug(cfg: AugmentConfig, rng: np.random.Generator):
    """Return a callable ``[T, J, 3] -> [T, J, 3]`` with freshly sampled parameters."""
    yaw = np.deg2rad(rng.uniform(-cfg.yaw_deg, cfg.yaw_deg))
    pitch = np.deg2rad(rng.uniform(-cfg.pitch_deg, cfg.pitch_deg))
    rot = _pitch_rotation(pitch) @ yaw_rotation(yaw)
    scale = rng.uniform(*cfg.scale_range)
    shift = rng.normal(0.0, cfg.translate_std, size=3) if cfg.translate_std > 0 else np.zeros(3)

    def apply(x: np.ndarray) -> np.ndarray:
        y = (x @ rot.T) * scale + shift
        if cfg.jitter_std > 0:
            y = y + rng.normal(0.0, cfg.jitter_std, size=y.shape)
        return y

    return apply
