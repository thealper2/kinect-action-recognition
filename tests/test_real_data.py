"""Checks on the actual KARD files; skipped when the dataset has not been prepared."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skeleton_action.data.kard import KardCollection

INTERIM = Path("data/interim/kard_realworld.npz")
pytestmark = pytest.mark.requires_data


@pytest.mark.skipif(not INTERIM.exists(), reason="run `make prepare-data` first")
def test_full_dataset_parsed() -> None:
    col = KardCollection.load(INTERIM)
    assert len(col) == 540
    assert np.bincount(col.labels, minlength=18).tolist() == [30] * 18
    assert np.bincount(col.subjects, minlength=11)[1:].tolist() == [54] * 10
    assert np.isfinite(col.coords).all()
    # Joint-order sanity: head above torso, torso above feet in (almost) every frame.
    assert (col.coords[:, 0, 1] > col.coords[:, 8, 1]).mean() > 0.99
    assert (col.coords[:, 8, 1] > col.coords[:, 11, 1]).mean() > 0.99
