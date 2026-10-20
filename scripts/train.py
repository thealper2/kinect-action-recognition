#!/usr/bin/env python
"""Train one model from an experiment config (or explicit component configs).

Examples:
    python scripts/train.py --config configs/experiments/e3_stgcn_xyz.yaml
    python scripts/train.py --config configs/experiments/e3_stgcn_xyz.yaml \
        --set training.seed=43 training.epochs=200
    python scripts/train.py --config configs/experiments/e3_stgcn_xyz.yaml \
        --resume outputs/checkpoints/e3_stgcn_xyz/seed42/last.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skeleton_action.training.trainer import train
from skeleton_action.utils.config import resolve_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", help="experiment YAML")
    ap.add_argument("--dataset")
    ap.add_argument("--split")
    ap.add_argument("--model")
    ap.add_argument("--training")
    ap.add_argument("--name")
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted overrides, e.g. training.epochs=10 model.params.dropout=0.5",
    )
    ap.add_argument("--resume", type=Path, help="checkpoint (last.pt) to resume from")
    ap.add_argument("--overwrite", action="store_true", help="restart an existing run")
    ap.add_argument("--output-root", default="outputs")
    ap.add_argument("--no-test", action="store_true", help="skip the final test evaluation")
    args = ap.parse_args()

    cfg = resolve_config(
        args.config,
        dataset=args.dataset,
        split=args.split,
        model=args.model,
        training=args.training,
        overrides=args.set,
        name=args.name,
    )
    summary = train(
        cfg,
        args.output_root,
        resume=args.resume,
        overwrite=args.overwrite,
        evaluate_test=not args.no_test,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
