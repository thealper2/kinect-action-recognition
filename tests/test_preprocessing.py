from __future__ import annotations

import numpy as np
import pytest

from skeleton_action.constants import J
from skeleton_action.data.dataset import SkeletonDataset
from skeleton_action.preprocessing.augment import AugmentConfig, temporal_crop
from skeleton_action.preprocessing.transforms import (
    TORSO_BONES,
    PreprocessConfig,
    fill_missing,
    joint_validity,
    normalize_spatial,
    preprocess_sequence,
    resample,
    velocities,
)


def test_config_validation_and_digest() -> None:
    with pytest.raises(ValueError):
        PreprocessConfig(center="bogus")
    with pytest.raises(ValueError):
        PreprocessConfig.from_dict({"unknown_key": 1})
    a, b = PreprocessConfig(), PreprocessConfig(velocity=True)
    assert a.digest() != b.digest() and a.num_channels == 3 and b.num_channels == 6
    assert PreprocessConfig(velocity=True, append_mask=True).num_channels == 7
    assert PreprocessConfig.from_dict(a.to_dict()) == a


def test_first_frame_centering_keeps_global_motion(standing_seq) -> None:
    seq = standing_seq.copy()
    seq[:, :, 1] -= np.linspace(0, 400, len(seq))[:, None]  # whole body moves down (sit)
    x = normalize_spatial(seq, PreprocessConfig(scale="none"))
    np.testing.assert_allclose(x[0, J.TORSO], 0, atol=1e-9)
    assert x[-1, J.TORSO, 1] < -0.35  # 400 mm drop kept (in metres)
    xp = normalize_spatial(seq, PreprocessConfig(scale="none", center="per_frame"))
    np.testing.assert_allclose(xp[:, J.TORSO], 0, atol=1e-9)


def test_translation_and_scale_invariance(standing_seq) -> None:
    cfg = PreprocessConfig(num_frames=24)
    a = preprocess_sequence(standing_seq, cfg)
    b = preprocess_sequence(standing_seq * 1.3 + np.array([500, -200, 900], np.float32), cfg)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_body_scale_normalisation(standing_seq) -> None:
    x = normalize_spatial(standing_seq, PreprocessConfig())
    lengths = np.mean([np.linalg.norm(x[:, i] - x[:, j], axis=-1) for i, j in TORSO_BONES], axis=0)
    assert abs(np.median(lengths) - 1.0) < 1e-6


def test_yaw_alignment(standing_seq) -> None:
    theta = np.deg2rad(30)
    rot = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    rotated = (standing_seq @ rot.T).astype(np.float32)
    x = normalize_spatial(rotated, PreprocessConfig(orientation="yaw_hips"))
    v = x[0, J.L_HIP] - x[0, J.R_HIP]
    assert abs(v[2]) < 1e-6 and v[0] > 0
    ref = normalize_spatial(standing_seq, PreprocessConfig(orientation="yaw_hips"))
    np.testing.assert_allclose(x, ref, atol=1e-5)


def test_resample_linear_exact() -> None:
    t = np.arange(10, dtype=float)
    x = np.stack([2 * t + 1, -t], axis=1)[:, None, :]  # [10, 1, 2] linear in time
    r = resample(x, 19)
    assert r.shape == (19, 1, 2)
    np.testing.assert_allclose(r[0], x[0])
    np.testing.assert_allclose(r[-1], x[-1])
    np.testing.assert_allclose(r[:, 0, 0], 2 * np.linspace(0, 9, 19) + 1)
    assert resample(x[:1], 5).shape == (5, 1, 2)
    np.testing.assert_allclose(resample(x, 10), x)


def test_velocity_modes() -> None:
    x = np.zeros((5, 15, 3))
    x[:, :, 0] = np.arange(5)[:, None] * 0.5
    v = velocities(x, "frame_diff", 5, 30.0)
    np.testing.assert_allclose(v[..., 0], 0.5)
    vs = velocities(x, "per_second", 9, 30.0)  # 9 source frames span 8/30 s over 4 steps
    np.testing.assert_allclose(vs[..., 0], 0.5 * 4 / (8 / 30))
    out = preprocess_sequence(
        np.random.default_rng(0).normal(500, 100, (30, 15, 3)).astype(np.float32),
        PreprocessConfig(num_frames=8, velocity=True),
    )
    assert out.shape == (8, 15, 6)
    np.testing.assert_allclose(out[1:, :, 3:], np.diff(out[:, :, :3], axis=0), atol=1e-5)


def test_missing_joint_handling(standing_seq) -> None:
    seq = standing_seq.copy()
    seq[5:9, J.L_HAND] = 0.0  # untracked placeholder
    seq[:, J.L_FOOT] = np.nan  # never valid
    mask = joint_validity(seq)
    assert not mask[5:9, J.L_HAND].any() and mask[4, J.L_HAND]
    assert not mask[:, J.L_FOOT].any()
    filled = fill_missing(np.nan_to_num(seq), mask)
    exp = np.linspace(seq[4, J.L_HAND], seq[9, J.L_HAND], 6)[1:-1]
    np.testing.assert_allclose(filled[5:9, J.L_HAND], exp, rtol=1e-5)
    np.testing.assert_allclose(filled[:, J.L_FOOT], filled[:, J.TORSO])
    out = preprocess_sequence(seq, PreprocessConfig(num_frames=16, append_mask=True))
    assert np.isfinite(out).all() and out.shape == (16, 15, 4)
    assert out[:, J.L_FOOT, 3].max() == 0.0 and out[:, J.HEAD, 3].min() == 1.0


def test_temporal_crop_bounds() -> None:
    rng = np.random.default_rng(0)
    seq = np.zeros((100, 15, 3))
    mask = np.ones((100, 15), bool)
    for _ in range(50):
        s, m = temporal_crop(seq, mask, 0.8, rng)
        assert 80 <= len(s) <= 100 and len(m) == len(s)


def test_augmentation_only_on_train(tiny_collection) -> None:
    pre = PreprocessConfig(num_frames=16)
    idx = np.arange(10)
    det = SkeletonDataset(tiny_collection, idx, pre, None)
    np.testing.assert_array_equal(det[3][0].numpy(), det[3][0].numpy())
    aug = SkeletonDataset(tiny_collection, idx, pre, AugmentConfig(), seed=0)
    assert not np.allclose(aug[3][0].numpy(), aug[3][0].numpy())
    off = SkeletonDataset(tiny_collection, idx, pre, AugmentConfig(enabled=False))
    np.testing.assert_allclose(off[3][0].numpy(), det[3][0].numpy())
    # Same seed -> same augmented stream (reproducibility).
    a1 = SkeletonDataset(tiny_collection, idx, pre, AugmentConfig(), seed=5)
    a2 = SkeletonDataset(tiny_collection, idx, pre, AugmentConfig(), seed=5)
    np.testing.assert_array_equal(a1[2][0].numpy(), a2[2][0].numpy())
