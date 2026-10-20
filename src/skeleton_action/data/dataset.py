"""PyTorch dataset over a :class:`KardCollection` with on-the-fly training augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from skeleton_action.data.kard import KardCollection
from skeleton_action.preprocessing.augment import (
    AugmentConfig,
    joint_dropout,
    make_spatial_aug,
    temporal_crop,
)
from skeleton_action.preprocessing.transforms import (
    PreprocessConfig,
    finalize_features,
    joint_validity,
    normalize_spatial,
    preprocess_sequence,
)


def preprocess_collection(col: KardCollection, cfg: PreprocessConfig) -> np.ndarray:
    """Deterministically preprocess every sequence -> ``[N, T, J, C]`` float32."""
    return np.stack([preprocess_sequence(col.sequence(i), cfg) for i in range(len(col))])


def cached_preprocess(
    col: KardCollection, cfg: PreprocessConfig, cache_dir: str | Path | None
) -> np.ndarray:
    """:func:`preprocess_collection` with an on-disk cache keyed by the config digest."""
    if cache_dir is None:
        return preprocess_collection(col, cfg)
    path = Path(cache_dir) / f"kard_{col.kind}_{cfg.digest()}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if [str(s) for s in z["seq_ids"]] == col.seq_ids:
                return z["x"]
    x = preprocess_collection(col, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x,
        y=col.labels,
        subjects=col.subjects,
        seq_ids=np.array(col.seq_ids, dtype="U16"),
        config=np.array(str(cfg.to_dict())),
    )
    return x


class SkeletonDataset(Dataset):
    """Samples of a subset of a KARD collection.

    Without augmentation the features are precomputed once (deterministic). With
    augmentation, every ``__getitem__`` draws new parameters from ``self.rng``; use
    ``num_workers=0`` (the default in this project, the data are tiny) for exact
    reproducibility and resumability via :meth:`rng_state`.

    Items: ``(x [T, J, C] float32, label int64, index int64)``.
    """

    def __init__(
        self,
        collection: KardCollection,
        indices: np.ndarray,
        pre_cfg: PreprocessConfig,
        aug_cfg: AugmentConfig | None = None,
        seed: int = 0,
        precomputed: np.ndarray | None = None,
    ) -> None:
        self.col = collection
        self.indices = np.asarray(indices, dtype=np.int64)
        self.pre_cfg = pre_cfg
        self.aug_cfg = aug_cfg if (aug_cfg is not None and aug_cfg.enabled) else None
        self.rng = np.random.default_rng(seed)
        self.labels = collection.labels[self.indices]
        self._fixed: np.ndarray | None = None
        if self.aug_cfg is None:
            if precomputed is not None:
                self._fixed = precomputed[self.indices]
            else:
                self._fixed = np.stack(
                    [preprocess_sequence(collection.sequence(i), pre_cfg) for i in self.indices]
                )

    def __len__(self) -> int:
        return len(self.indices)

    def _augmented(self, i: int) -> np.ndarray:
        cfg, aug = self.pre_cfg, self.aug_cfg
        assert aug is not None
        seq = self.col.sequence(i)
        mask = joint_validity(seq)
        seq, mask = temporal_crop(seq, mask, aug.temporal_crop_min, self.rng)
        mask = joint_dropout(mask, aug.joint_dropout_p, self.rng)
        x = normalize_spatial(seq, cfg, mask)
        return finalize_features(
            x,
            mask,
            cfg,
            n_source_frames=seq.shape[0],
            spatial_aug=make_spatial_aug(aug, self.rng),
        )

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        x = self._fixed[k] if self._fixed is not None else self._augmented(i)
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.tensor(int(self.labels[k]), dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
        )

    def rng_state(self) -> dict:
        return self.rng.bit_generator.state

    def set_rng_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state
