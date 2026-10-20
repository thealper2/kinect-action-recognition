#!/usr/bin/env python
"""Offline inference from a checkpoint (no Kinect required).

Inputs (exactly one):
    --raw-file PATH       a raw KARD *_realworld.txt file (normal preprocessing pipeline)
    --seq-id aAA_sSS_eEE  a sequence from the interim cache (raw coordinates)
    --processed-seq-id ID a sequence from the processed feature cache (already preprocessed)
    --synthetic           procedural skeleton, smoke test only (NOT an accuracy measure)

Example:
    python scripts/infer.py --checkpoint outputs/checkpoints/e3_stgcn_xyz/seed42/best.pt \
        --raw-file data/raw/kard/KARD_a01_a02/a01_s09_e01_realworld.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from skeleton_action.constants import CLASS_NAMES
from skeleton_action.data.kard import (
    KardCollection,
    KardParseError,
    parse_filename,
    parse_skeleton_file,
)
from skeleton_action.inference.predictor import Predictor
from skeleton_action.sensors.sources import SyntheticSource
from skeleton_action.utils.seed import select_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--raw-file", type=Path)
    g.add_argument("--seq-id")
    g.add_argument("--processed-seq-id")
    g.add_argument("--synthetic", action="store_true")
    ap.add_argument("--interim", type=Path, default=Path("data/interim/kard_realworld.npz"))
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    pred = Predictor(args.checkpoint, select_device(args.device))
    truth = None
    if args.raw_file:
        try:
            parsed = parse_skeleton_file(args.raw_file)
        except KardParseError as exc:
            print(json.dumps({"valid": False, "reason": str(exc)}))
            sys.exit(1)
        try:
            sid, _ = parse_filename(args.raw_file)
            truth = CLASS_NAMES[sid.label]
        except KardParseError:
            pass
        res = pred.predict_raw(parsed.coords, parsed.valid_mask, topk=args.topk)
        source = str(args.raw_file)
    elif args.seq_id:
        col = KardCollection.load(args.interim)
        i = col.index_of(args.seq_id)
        truth = CLASS_NAMES[int(col.labels[i])]
        res = pred.predict_raw(col.sequence(i), topk=args.topk)
        source = f"interim:{args.seq_id}"
    elif args.processed_seq_id:
        digest = pred.pre.digest()
        path = (
            args.processed_dir / f"kard_{pred.ckpt.get('dataset_kind', 'realworld')}_{digest}.npz"
        )
        if not path.exists():
            raise SystemExit(
                f"{path} not found; run `make prepare-data` (digest {digest} "
                "must match the checkpoint preprocessing)"
            )
        with np.load(path, allow_pickle=False) as z:
            ids = [str(s) for s in z["seq_ids"]]
            i = ids.index(args.processed_seq_id)
            x, truth = z["x"][i], CLASS_NAMES[int(z["y"][i])]
        res = pred.predict_features(x, topk=args.topk)
        source = f"processed:{path.name}:{args.processed_seq_id}"
    else:
        src = SyntheticSource(num_frames=90)
        seq = np.stack([f.joints for f in src])
        res = pred.predict_raw(seq, topk=args.topk)
        source = "synthetic (smoke test only; the prediction is meaningless)"

    out = {
        "source": source,
        "checkpoint": str(args.checkpoint),
        "model": pred.ckpt["model_name"],
        **res.to_dict(),
    }
    if truth is not None:
        out["ground_truth"] = truth
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
