#!/usr/bin/env python
"""Parameter count, checkpoint size and forward latency of checkpoints or fresh models.

Examples:
    python scripts/benchmark.py --checkpoint outputs/checkpoints/e3_stgcn_xyz/seed42/best.pt
    python scripts/benchmark.py --fresh       # untrained default models (no data needed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from skeleton_action.evaluation.benchmark import measure_latency
from skeleton_action.models import build_model, count_parameters
from skeleton_action.training.checkpoint import (
    load_checkpoint,
    model_from_checkpoint,
)
from skeleton_action.utils.config import load_yaml


def bench(model, shape, label: str, extra: dict) -> dict:
    res = {
        "model": label,
        "params": count_parameters(model),
        **extra,
        "cpu_1thread": measure_latency(model, shape, "cpu", num_threads=1),
        "cpu_all_threads": measure_latency(model, shape, "cpu"),
    }
    if torch.cuda.is_available():
        res["cuda"] = measure_latency(model, shape, "cuda")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("outputs/metrics/benchmark.json"))
    args = ap.parse_args()

    results = []
    for ck in args.checkpoint:
        c = load_checkpoint(ck)
        m = model_from_checkpoint(c)
        shape = (1, c["num_frames"], c["num_joints"], c["in_channels"])
        results.append(
            bench(
                m, shape, f"{c['model_name']} ({ck})", {"checkpoint_size_bytes": ck.stat().st_size}
            )
        )
    if args.fresh or not args.checkpoint:
        for name in ("mlp", "lstm", "stgcn"):
            params = load_yaml(f"configs/model/{name}.yaml").get("params", {})
            m = build_model(name, 18, args.frames, 15, args.channels, **params)
            results.append(bench(m, (1, args.frames, 15, args.channels), f"{name} (untrained)", {}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for r in results:
        line = (
            f"{r['model']:<40s} params {r['params']:>9,d} | cpu1 "
            f"{r['cpu_1thread']['median_ms']:.2f} ms | "
            f"cpu* {r['cpu_all_threads']['median_ms']:.2f} ms"
        )
        if "cuda" in r:
            line += f" | cuda {r['cuda']['median_ms']:.2f} ms"
        print(line)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
