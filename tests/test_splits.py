from __future__ import annotations

import numpy as np
import pytest

from skeleton_action.data.splits import SubjectSplit, loso_splits
from skeleton_action.utils.config import load_yaml


def test_default_split_is_subject_independent() -> None:
    sp = SubjectSplit.from_dict(load_yaml("configs/dataset/split_subject_fixed.yaml"))
    assert set(sp.train) | set(sp.val) | set(sp.test) == set(range(1, 11))
    subjects = np.repeat(np.arange(1, 11), 54)
    parts = sp.indices(subjects)
    assert sum(len(v) for v in parts.values()) == 540
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(subjects[parts[a]]) & set(subjects[parts[b]])
        assert not set(parts[a]) & set(parts[b])


@pytest.mark.parametrize(
    "bad",
    [
        {"train": [1, 2], "val": [2], "test": [3]},
        {"train": [1, 1], "val": [2], "test": [3]},
        {"train": [1], "val": [11], "test": []},
        {"train": [], "val": [1], "test": [2]},
    ],
)
def test_invalid_splits_rejected(bad) -> None:
    with pytest.raises(ValueError):
        SubjectSplit.from_dict(bad)


def test_loso() -> None:
    folds = loso_splits()
    assert len(folds) == 10
    assert sorted(f.test[0] for f in folds) == list(range(1, 11))
    for f in folds:
        assert len(f.train) == 8 and len(f.val) == 1
        assert not (set(f.train) & set(f.val)) and not (set(f.train) & set(f.test))
        assert f.val != f.test
