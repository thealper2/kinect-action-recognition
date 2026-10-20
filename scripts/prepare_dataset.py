#!/usr/bin/env python
"""Parse the raw KARD files into the interim cache and build deterministic feature caches.

Outputs:
    data/interim/kard_<kind>.npz               parsed [sum T, 15, 3] + offsets + metadata
    data/interim/kard_<kind>.rejected.json     rejected files with reasons
    data/processed/kard_<kind>_<digest>.npz    preprocessed [N, T, 15, C] per config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from skeleton_action.data.dataset import cached_preprocess
from skeleton_action.data.loading import get_collection
from skeleton_action.data.splits import SubjectSplit
from skeleton_action.preprocessing.transforms import PreprocessConfig
from skeleton_action.utils.config import load_yaml, resolve_config, suite_entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="configs/dataset/kard_realworld.yaml")
    ap.add_argument("--split", default="configs/dataset/split_subject_fixed.yaml")
    ap.add_argument(
        "--suite",
        default="configs/experiments/suite.yaml",
        help="also build feature caches for every experiment in this suite",
    )
    ap.add_argument("--rebuild", action="store_true", help="re-parse raw files")
    args = ap.parse_args()

    dcfg = load_yaml(args.dataset)
    col = get_collection(dcfg, rebuild=args.rebuild)
    print(f"collection: {len(col)} sequences, {int(col.lengths.sum())} frames, kind={col.kind}")
    split = SubjectSplit.from_dict(load_yaml(args.split))
    parts = split.indices(col.subjects)
    for name, idx in parts.items():
        subj = sorted({int(s) for s in col.subjects[idx]})
        counts = np.bincount(col.labels[idx], minlength=18)
        print(
            f"  {name:5s}: {len(idx):3d} sequences, subjects {subj}, "
            f"per-class min/max {counts.min()}/{counts.max()}"
        )

    configs = {
        PreprocessConfig.from_dict(dcfg.get("preprocess")).digest(): PreprocessConfig.from_dict(
            dcfg.get("preprocess")
        )
    }
    if args.suite and Path(args.suite).exists():
        for exp, _seeds in suite_entries(load_yaml(args.suite)):
            cfg = resolve_config(exp)
            if cfg["dataset"].get("kind", "realworld") != col.kind:
                continue
            pre = PreprocessConfig.from_dict(cfg["dataset"].get("preprocess"))
            configs[pre.digest()] = pre
    for digest, pre in configs.items():
        x = cached_preprocess(col, pre, dcfg.get("processed_dir", "data/processed"))
        print(
            f"  processed {digest}: {x.shape} finite={bool(np.isfinite(x).all())} "
            f"(T={pre.num_frames}, C={pre.num_channels}, center={pre.center}, "
            f"scale={pre.scale}, velocity={pre.velocity})"
        )


if __name__ == "__main__":
    main()
