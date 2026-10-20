"""Deterministic skeleton preprocessing: validation, normalisation, resampling, features.

Input convention: KARD real-world coordinates ``[T, 15, 3]`` in millimetres (X right in
the sensor image, Y up, Z depth). Output: model input ``[T_out, 15, C]`` float32.

Pipeline (each step configurable, see :class:`PreprocessConfig`)::

    mm -> metres -> missing-joint fill -> centring -> body-scale -> yaw alignment
       -> temporal resampling to T_out -> [optional] velocity / mask channels

Design notes (details in docs/preprocessing.md):

* The default centring subtracts the torso position of the *first* frame of the
  sequence/window, not of every frame. Per-frame centring would delete the global
  trajectory of the body (the torso drop of "Sit down", the rise of "Stand up", the
  displacement of "Walk"), which carries class information.
* Body scale is the median over frames of the mean length of five torso bones, a
  rigid-ish quantity that is robust to arm/leg motion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

import numpy as np

from skeleton_action.constants import NUM_JOINTS, J

TORSO_BONES: list[tuple[int, int]] = [
    (J.NECK, J.TORSO),
    (J.NECK, J.R_SHOULDER),
    (J.NECK, J.L_SHOULDER),
    (J.TORSO, J.R_HIP),
    (J.TORSO, J.L_HIP),
]

CENTER_MODES = ("none", "first_frame", "per_frame")
SCALE_MODES = ("none", "torso_bones")
ORIENT_MODES = ("none", "yaw_hips", "yaw_shoulders")
VELOCITY_MODES = ("frame_diff", "per_second")


@dataclass
class PreprocessConfig:
    """Deterministic preprocessing configuration (stored in every checkpoint).

    Attributes:
        num_frames: temporal length T after resampling.
        center: ``none`` | ``first_frame`` (subtract the reference joint of the first
            frame) | ``per_frame`` (subtract it in every frame).
        center_joint: reference joint index (default torso, 8).
        scale: ``none`` | ``torso_bones`` (divide by the median torso-bone length).
        orientation: ``none`` | ``yaw_hips`` | ``yaw_shoulders``: rotate about the
            vertical (Y) axis so that the left-minus-right hip (shoulder) vector of the
            first frame points along +X.
        velocity: append per-joint velocities (C: 3 -> 6).
        velocity_mode: ``frame_diff`` = difference between consecutive *resampled*
            frames (units: normalised length per 1/(T-1) of the window), or
            ``per_second`` = time derivative using ``source_fps`` and the original length.
        source_fps: frame rate assumed for the raw sequence (only used by
            ``velocity_mode="per_second"`` and by the real-time window length).
        append_mask: append a per-joint validity channel (1 = observed, 0 = filled).
        input_unit_scale: multiply raw values by this (1e-3: mm -> m).
    """

    num_frames: int = 64
    center: str = "first_frame"
    center_joint: int = J.TORSO
    scale: str = "torso_bones"
    orientation: str = "none"
    velocity: bool = False
    velocity_mode: str = "frame_diff"
    source_fps: float = 30.0
    append_mask: bool = False
    input_unit_scale: float = 1e-3

    def __post_init__(self) -> None:
        if self.num_frames < 2:
            raise ValueError("num_frames must be >= 2")
        for name, val, allowed in (
            ("center", self.center, CENTER_MODES),
            ("scale", self.scale, SCALE_MODES),
            ("orientation", self.orientation, ORIENT_MODES),
            ("velocity_mode", self.velocity_mode, VELOCITY_MODES),
        ):
            if val not in allowed:
                raise ValueError(f"{name}={val!r} not in {allowed}")
        if not 0 <= self.center_joint < NUM_JOINTS:
            raise ValueError("center_joint out of range")

    @property
    def num_channels(self) -> int:
        return 3 + (3 if self.velocity else 0) + (1 if self.append_mask else 0)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> PreprocessConfig:
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown preprocessing keys: {sorted(unknown)}")
        return cls(**d)

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- validity


def joint_validity(seq: np.ndarray) -> np.ndarray:
    """``[T, J]`` bool: joint is finite and not the (0, 0, 0) untracked placeholder."""
    finite = np.isfinite(seq).all(axis=-1)
    nonzero = np.any(np.nan_to_num(seq) != 0.0, axis=-1)
    return finite & nonzero


def fill_missing(seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate invalid joints over time; hold the nearest valid value at the
    ends. Joints that are never valid are set to the reference value of the torso (or
    0 if the torso is never valid either), so they carry no motion.

    Args:
        seq: ``[T, J, 3]``.
        mask: ``[T, J]`` bool validity.
    """
    out = np.array(seq, dtype=np.float64, copy=True)
    t = np.arange(seq.shape[0])
    never: list[int] = []
    for j in range(seq.shape[1]):
        m = mask[:, j]
        if m.all():
            continue
        if not m.any():
            never.append(j)
            continue
        for c in range(seq.shape[2]):
            out[~m, j, c] = np.interp(t[~m], t[m], out[m, j, c])
    if never:
        torso_ok = J.TORSO not in never
        fill = out[:, J.TORSO, :] if torso_ok else np.zeros((seq.shape[0], seq.shape[2]))
        for j in never:
            out[:, j, :] = fill
    return out


# --------------------------------------------------------------------------- spatial


def body_scale(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Median (over frames) of the mean length of :data:`TORSO_BONES`.

    Only frames where all bone endpoints are valid are used, if any.
    """
    a = np.array([b[0] for b in TORSO_BONES])
    b = np.array([b[1] for b in TORSO_BONES])
    lengths = np.linalg.norm(x[:, a] - x[:, b], axis=-1).mean(axis=1)  # [T]
    if mask is not None:
        ok = mask[:, a].all(axis=1) & mask[:, b].all(axis=1)
        if ok.any():
            lengths = lengths[ok]
    s = float(np.median(lengths))
    return s if s > 1e-6 else 1.0


def yaw_rotation(theta: float) -> np.ndarray:
    """Rotation matrix about +Y by ``theta`` radians (applied as ``x @ R.T``)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def alignment_yaw(x0: np.ndarray, mode: str) -> float:
    """Yaw angle that rotates the left-minus-right hip/shoulder vector onto +X."""
    left, right = (J.L_HIP, J.R_HIP) if mode == "yaw_hips" else (J.L_SHOULDER, J.R_SHOULDER)
    v = x0[left] - x0[right]
    if np.hypot(v[0], v[2]) < 1e-6:
        return 0.0
    # R(theta) maps (vx, vz) -> (c*vx + s*vz, -s*vx + c*vz); zero z-component when
    # theta = atan2(vz, vx).
    return float(np.arctan2(v[2], v[0]))


def normalize_spatial(
    seq: np.ndarray, cfg: PreprocessConfig, mask: np.ndarray | None = None
) -> np.ndarray:
    """Unit conversion, missing fill, centring, scale and yaw alignment.

    Args:
        seq: ``[T, J, 3]`` raw coordinates (KARD real-world convention).
        cfg: preprocessing config.
        mask: optional ``[T, J]`` validity; computed from the data when omitted.

    Returns:
        ``[T, J, 3]`` float64 normalised coordinates (same T).
    """
    if seq.ndim != 3 or seq.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(f"expected [T, {NUM_JOINTS}, 3], got {seq.shape}")
    if seq.shape[0] < 1:
        raise ValueError("empty sequence")
    if mask is None:
        mask = joint_validity(seq)
    x = fill_missing(np.nan_to_num(seq.astype(np.float64)), mask) * cfg.input_unit_scale

    cj = cfg.center_joint
    if cfg.center == "first_frame":
        valid_t = np.flatnonzero(mask[:, cj])
        ref_t = int(valid_t[0]) if valid_t.size else 0
        x = x - x[ref_t, cj][None, None, :]
    elif cfg.center == "per_frame":
        x = x - x[:, cj : cj + 1, :]

    if cfg.scale == "torso_bones":
        x = x / body_scale(x, mask)

    if cfg.orientation != "none":
        theta = alignment_yaw(x[0], cfg.orientation)
        x = x @ yaw_rotation(theta).T
    return x


# --------------------------------------------------------------------------- temporal


def resample(x: np.ndarray, num_frames: int) -> np.ndarray:
    """Linear temporal interpolation of ``[T, ...]`` to ``[num_frames, ...]``.

    Endpoints are preserved exactly; T == 1 is repeated.
    """
    t_in = x.shape[0]
    if t_in == num_frames:
        return x.copy()
    if t_in == 1:
        return np.repeat(x, num_frames, axis=0)
    pos = np.linspace(0.0, t_in - 1, num_frames)
    i0 = np.floor(pos).astype(np.int64)
    i0 = np.clip(i0, 0, t_in - 2)
    w = (pos - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1.0 - w) * x[i0] + w * x[i0 + 1]


def velocities(x: np.ndarray, mode: str, n_source_frames: int, source_fps: float) -> np.ndarray:
    """Per-joint velocities of a resampled sequence ``[T, J, 3]``.

    ``frame_diff``: v_t = x_t - x_{t-1} (v_0 = v_1 so the first frame is not a spike).
    ``per_second``: frame_diff scaled by (T - 1) / duration_seconds of the source window.
    """
    v = np.zeros_like(x)
    if x.shape[0] > 1:
        v[1:] = x[1:] - x[:-1]
        v[0] = v[1]
    if mode == "per_second":
        duration = max(n_source_frames - 1, 1) / source_fps
        v = v * (x.shape[0] - 1) / duration
    return v


def preprocess_sequence(
    seq: np.ndarray, cfg: PreprocessConfig, mask: np.ndarray | None = None
) -> np.ndarray:
    """Full deterministic pipeline: raw ``[T, 15, 3]`` -> model input ``[T_out, 15, C]``."""
    if mask is None:
        mask = joint_validity(seq)
    x = normalize_spatial(seq, cfg, mask)
    return finalize_features(x, mask, cfg, n_source_frames=seq.shape[0])


def finalize_features(
    x_norm: np.ndarray,
    mask: np.ndarray,
    cfg: PreprocessConfig,
    n_source_frames: int,
    spatial_aug=None,
) -> np.ndarray:
    """Resample normalised coordinates and append feature channels.

    ``spatial_aug`` (training only) is a callable applied to the resampled ``[T, J, 3]``
    coordinates *before* velocities are computed, so velocities stay consistent.
    """
    xr = resample(x_norm, cfg.num_frames)
    if spatial_aug is not None:
        xr = spatial_aug(xr)
    feats = [xr]
    if cfg.velocity:
        feats.append(velocities(xr, cfg.velocity_mode, n_source_frames, cfg.source_fps))
    if cfg.append_mask:
        mr = resample(mask.astype(np.float64), cfg.num_frames)
        feats.append((mr > 0.5).astype(np.float64)[..., None])
    return np.concatenate(feats, axis=-1).astype(np.float32)
