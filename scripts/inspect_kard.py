#!/usr/bin/env python
"""Inspect the extracted KARD skeleton files and write a JSON + Markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skeleton_action.data.inspection import build_report, report_to_markdown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/kard"))
    ap.add_argument("--kind", choices=["realworld", "screen"], default="realworld")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/reports"))
    args = ap.parse_args()

    if not args.raw_dir.exists() or not any(args.raw_dir.rglob("*.txt")):
        raise SystemExit(
            f"No skeleton .txt files under {args.raw_dir}. Run `make extract-data` first "
            "(see docs/dataset_setup.md)."
        )
    report = build_report(args.raw_dir, args.kind)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jpath = args.out_dir / f"kard_inspection_{args.kind}.json"
    mpath = args.out_dir / f"kard_inspection_{args.kind}.md"
    jpath.write_text(json.dumps(report, indent=2))
    md = report_to_markdown(report)
    mpath.write_text(md)
    print(md)
    print(f"Saved: {jpath} and {mpath}")
    if report["valid_sequences"] != report["expected_sequences"]:
        print(
            f"WARNING: {report['valid_sequences']} valid of {report['expected_sequences']} "
            "expected sequences.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
