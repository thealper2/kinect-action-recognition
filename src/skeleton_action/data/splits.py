"""Subject-independent splits: a fixed train/val/test split and leave-one-subject-out."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skeleton_action.constants import NUM_SUBJECTS


@dataclass
class SubjectSplit:
    """Explicit subject IDs (1-based) for each partition."""

    name: str = "subject_fixed"
    train: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    val: list[int] = field(default_factory=lambda: [7, 8])
    test: list[int] = field(default_factory=lambda: [9, 10])

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        sets = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        for k, s in sets.items():
            if len(s) != len(getattr(self, k)):
                raise ValueError(f"duplicate subject in {k}")
            if not s and k != "test":
                raise ValueError(f"{k} must contain at least one subject")
            bad = [x for x in s if not 1 <= x <= NUM_SUBJECTS]
            if bad:
                raise ValueError(f"subject ids out of range in {k}: {bad}")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            inter = sets[a] & sets[b]
            if inter:
                raise ValueError(f"subjects {sorted(inter)} appear in both {a} and {b}")

    def indices(self, subjects: np.ndarray) -> dict[str, np.ndarray]:
        """Sample indices per partition given the per-sample subject array."""
        return {
            part: np.flatnonzero(np.isin(subjects, getattr(self, part)))
            for part in ("train", "val", "test")
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SubjectSplit:
        return cls(
            name=d.get("name", "custom"),
            train=list(d["train"]),
            val=list(d["val"]),
            test=list(d.get("test", [])),
        )


def loso_splits(subjects: list[int] | None = None) -> list[SubjectSplit]:
    """Leave-one-subject-out folds.

    Fold k tests on subject k and validates on the next subject (cyclically); the
    remaining subjects are used for training, so validation and test stay disjoint.
    """
    subjects = sorted(subjects or range(1, NUM_SUBJECTS + 1))
    folds = []
    for i, s in enumerate(subjects):
        v = subjects[(i + 1) % len(subjects)]
        train = [x for x in subjects if x not in (s, v)]
        folds.append(SubjectSplit(name=f"loso_s{s:02d}", train=train, val=[v], test=[s]))
    return folds
