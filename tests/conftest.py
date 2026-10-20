"""Shared synthetic fixtures (no KARD data or Kinect hardware needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skeleton_action.constants import NUM_JOINTS, J
from skeleton_action.data.kard import KardCollection, KardSampleId
from skeleton_action.sensors.sources import SyntheticSource


def make_sequence(label: int, subject: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """Standing pose (mm) with a class-dependent hand trajectory + small noise."""
    base = SyntheticSource.base_pose()
    t = np.linspace(0, 2 * np.pi, length)
    seq = np.repeat(base[None], length, axis=0)
    amp = 50.0 + 15.0 * label
    seq[:, J.R_HAND, 0] += amp * np.sin(t * (1 + label % 3))
    seq[:, J.R_HAND, 1] += amp * np.cos(t)
    seq[:, :, 0] += 30.0 * subject  # subject-specific position
    seq += rng.normal(0, 3.0, seq.shape)
    return seq.astype(np.float32)


@pytest.fixture(scope="session")
def tiny_collection() -> KardCollection:
    rng = np.random.default_rng(0)
    items = []
    for a in range(1, 19):
        for s in (1, 2, 3, 4):
            for e in (1, 2):
                n = int(rng.integers(20, 40))
                items.append((KardSampleId(a, s, e), make_sequence(a - 1, s, n, rng)))
    return KardCollection.from_sequences(items, "realworld")


@pytest.fixture
def tiny_cfg(tmp_path: Path, tiny_collection: KardCollection) -> dict:
    interim = tmp_path / "interim.npz"
    tiny_collection.save(interim)
    return {
        "name": "tiny",
        "dataset": {
            "kind": "realworld",
            "raw_dir": str(tmp_path / "raw"),
            "interim_path": str(interim),
            "processed_dir": None,
            "preprocess": {"num_frames": 16},
            "augment": {"enabled": True},
        },
        "split": {"name": "tiny", "train": [1, 2], "val": [3], "test": [4]},
        "model": {
            "name": "stgcn",
            "params": {"channels": [8, 8, 16], "strides": [1, 2, 1], "temporal_kernel": 3},
        },
        "training": {
            "seed": 1,
            "device": "cpu",
            "epochs": 3,
            "batch_size": 16,
            "optimizer": {"name": "adamw", "lr": 0.003},
            "scheduler": {"name": "cosine", "warmup_epochs": 1},
            "early_stopping": {"patience": 0},
        },
    }


@pytest.fixture
def standing_seq() -> np.ndarray:
    rng = np.random.default_rng(1)
    return make_sequence(0, 1, 40, rng)


assert NUM_JOINTS == 15
