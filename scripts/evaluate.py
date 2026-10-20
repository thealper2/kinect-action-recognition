#!/usr/bin/env python
"""Evaluate saved checkpoints on a partition of their subject-independent split.

Examples:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/e3_stgcn_xyz/seed42/best.pt
    python scripts/evaluate.py --all            # every best.pt under outputs/checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skeleton_action.evaluation.evaluator import evaluate_checkpoint
from skeleton_action.utils.seed import select_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--partition", choices=["train", "val", "test"], default="test")
    ap.add_argument("--output-root", type=Path, default=Path("outputs"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    ckpts = list(args.checkpoint)
    if args.all:
        ckpts += sorted((args.output_root / "checkpoints").rglob("best.pt"))
    if not ckpts:
        raise SystemExit(
            "No checkpoints given (use --checkpoint PATH or --all). "
            "Train a model first, e.g. `make train-stgcn`."
        )
    device = select_device(args.device)
    ck_root = (args.output_root / "checkpoints").resolve()
    for ck in ckpts:
        try:
            run_id = ck.resolve().parent.relative_to(ck_root)
        except ValueError:
            run_id = Path(ck.stem)
        m = evaluate_checkpoint(
            ck,
            args.partition,
            args.output_root / "metrics" / run_id,
            args.output_root / "figures" / run_id,
            device=device,
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(ck),
                    "partition": args.partition,
                    "subjects": m["subjects"],
                    "accuracy": round(m["accuracy"], 4),
                    "macro_f1": round(m["macro_f1"], 4),
                    "weighted_f1": round(m["weighted_f1"], 4),
                    "top_confusions": m["top_confusions"][:5],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
