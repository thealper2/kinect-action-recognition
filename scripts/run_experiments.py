#!/usr/bin/env python
"""Run the controlled experiment suite (multiple seeds) or leave-one-subject-out CV,
then aggregate the results into outputs/metrics/<tag>_summary.{csv,md}.

Runs that already finished (summary.json present) are skipped, interrupted runs are
resumed from last.pt, so the command can simply be re-issued after an interruption.

Examples:
    python scripts/run_experiments.py --suite configs/experiments/suite.yaml       # 14 runs
    python scripts/run_experiments.py --suite configs/experiments/suite_full.yaml  # 24 runs
    python scripts/run_experiments.py --loso configs/experiments/e3_stgcn_xyz.yaml --seeds 42
    python scripts/run_experiments.py --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from skeleton_action.data.splits import loso_splits
from skeleton_action.training.trainer import run_paths, train
from skeleton_action.utils.config import load_yaml, resolve_config, suite_entries

METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")


def run_one(cfg: dict, output_root: str) -> None:
    paths = run_paths(cfg, output_root)
    if (paths["metrics"] / "summary.json").exists():
        print(f"[skip] {paths['run_id']} already finished")
        return
    last = paths["ckpt"] / "last.pt"
    resume = last if last.exists() else None
    print(f"[run] {paths['run_id']}" + (f" (resume {last})" if resume else ""))
    train(cfg, output_root, resume=resume)


def aggregate(output_root: str, tag: str, prefix: str | None = None) -> pd.DataFrame | None:
    rows = []
    for sj in sorted(Path(output_root, "metrics").rglob("summary.json")):
        s = json.loads(sj.read_text())
        name, seed = s["run_id"].rsplit("/seed", 1)
        if prefix and not name.startswith(prefix):
            continue
        if not prefix and ("_loso_" in name or name.startswith("smoke")):
            continue
        row = {
            "experiment": name,
            "seed": int(seed),
            "model": s["model"],
            "params": s["params"],
            "best_epoch": s["best_epoch"],
            "val_macro_f1": s["best_val_macro_f1"],
        }
        for m in METRICS:
            row[f"test_{m}"] = s.get("test_metrics", {}).get(m)
        tm = sj.parent / "test_metrics.json"
        if tm.exists():
            t = json.loads(tm.read_text())
            row["ckpt_MB"] = t["checkpoint_size_bytes"] / 1e6
            row["cpu1_ms"] = t.get("latency_cpu_1thread", {}).get("median_ms")
            row["cuda_ms"] = t.get("latency_cuda", {}).get("median_ms")
        rows.append(row)
    if not rows:
        print("no finished runs to aggregate")
        return None
    df = pd.DataFrame(rows)
    out = Path(output_root, "metrics")
    df.to_csv(out / f"{tag}_runs.csv", index=False)
    num = [c for c in df.columns if c not in ("experiment", "model", "seed")]
    g = df.groupby(["experiment", "model"])[num]
    summ = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_std"))
    summ["n_seeds"] = df.groupby(["experiment", "model"]).size()
    summ.to_csv(out / f"{tag}_summary.csv")
    lines = [
        "| experiment | n | val macro-F1 | test acc | test macro-F1 | test weighted-F1 "
        "| params | cpu1 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (exp, _m), r in summ.iterrows():

        def pm(c, r=r):
            if int(r["n_seeds"]) < 2:  # a single seed has no spread
                return f"{r[c + '_mean']:.3f}"
            return f"{r[c + '_mean']:.3f} ± {r[c + '_std']:.3f}"

        lines.append(
            f"| {exp} | {int(r['n_seeds'])} | {pm('val_macro_f1')} | "
            f"{pm('test_accuracy')} | {pm('test_macro_f1')} | {pm('test_weighted_f1')} | "
            f"{int(r['params_mean']):,} | {r.get('cpu1_ms_mean', float('nan')):.2f} |"
        )
    md = "\n".join(lines) + "\n"
    (out / f"{tag}_summary.md").write_text(md)
    print(md)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--suite", help="suite YAML with `experiments` and `seeds`")
    ap.add_argument("--loso", help="experiment YAML to evaluate with leave-one-subject-out")
    ap.add_argument("--seeds", type=int, nargs="*", help="override the suite seeds")
    ap.add_argument(
        "--only", nargs="*", help="run only experiments whose name contains one of these substrings"
    )
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides for every run")
    ap.add_argument("--output-root", default="outputs")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if not args.aggregate_only:
        if args.suite:
            entries = suite_entries(load_yaml(args.suite), args.seeds)
            print(f"[suite] {args.suite}: {sum(len(s) for _, s in entries)} runs")
            for exp, seeds in entries:
                for seed in seeds:
                    cfg = resolve_config(exp, overrides=[*args.set, f"training.seed={seed}"])
                    if args.only and not any(s in cfg["name"] for s in args.only):
                        continue
                    run_one(cfg, args.output_root)
        if args.loso:
            seeds = args.seeds or [42]
            base = resolve_config(args.loso, overrides=args.set)
            for fold in loso_splits():
                for seed in seeds:
                    cfg = resolve_config(args.loso, overrides=[*args.set, f"training.seed={seed}"])
                    cfg["split"] = fold.to_dict()
                    cfg["name"] = f"{base['name']}_{fold.name}"
                    run_one(cfg, args.output_root)
    aggregate(args.output_root, "suite")
    if args.loso:
        base = resolve_config(args.loso)
        df = aggregate(args.output_root, f"loso_{base['name']}", prefix=f"{base['name']}_loso")
        if df is not None:
            print(
                f"LOSO over {len(df)} folds/seeds: test acc mean "
                f"{df['test_accuracy'].mean():.3f} ± {df['test_accuracy'].std(ddof=0):.3f}, "
                f"macro-F1 {df['test_macro_f1'].mean():.3f} ± "
                f"{df['test_macro_f1'].std(ddof=0):.3f}"
            )


if __name__ == "__main__":
    main()
