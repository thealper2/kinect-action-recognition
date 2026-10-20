#!/usr/bin/env python
"""Extract only the KARD skeleton text files from the official .tar parts.

The Mendeley archives (KARD_aXX_aYY.tar) also contain RGB videos (.mp4) and bz2 depth
maps, which this project does not use. Only ``*_realworld.txt`` and ``*_screen.txt``
(and ``loadJoints.m`` if present next to the archives) are copied to
``data/raw/kard/<archive stem>/``. The source archives are opened read-only.

Usage:
    python scripts/extract_kard.py --tar-dir "/path/to/KARD tar folder"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skeleton_action.data.kard import FILENAME_RE


def extract(tar_dir: Path, out_dir: Path, overwrite: bool = False) -> dict:
    tars = sorted(tar_dir.glob("*.tar"))
    if not tars:
        raise SystemExit(f"No .tar files found in {tar_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"tar_dir": str(tar_dir), "archives": {}}
    for tar_path in tars:
        dest = out_dir / tar_path.stem
        entry = {
            "size_bytes": tar_path.stat().st_size,
            "members_total": 0,
            "extracted": [],
            "skipped_other_types": 0,
            "rejected_names": [],
        }
        with tarfile.open(tar_path, "r:") as tf:
            members = tf.getmembers()
            entry["members_total"] = len(members)
            for m in members:
                name = Path(m.name).name
                if not m.isfile():
                    continue
                if not FILENAME_RE.match(name):
                    entry["skipped_other_types"] += 1
                    continue
                # Defensive: refuse anything that is not a plain relative file name.
                if m.name != name and (".." in Path(m.name).parts or m.name.startswith("/")):
                    entry["rejected_names"].append(m.name)
                    continue
                target = dest / name
                if target.exists() and not overwrite:
                    entry["extracted"].append(name)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    entry["rejected_names"].append(m.name)
                    continue
                with src, target.open("wb") as fh:
                    shutil.copyfileobj(src, fh)
                entry["extracted"].append(name)
        manifest["archives"][tar_path.name] = entry
        print(
            f"{tar_path.name}: {len(entry['extracted'])} skeleton files "
            f"({entry['skipped_other_types']} video/depth members skipped)"
        )
    loader = tar_dir / "loadJoints.m"
    if loader.exists():
        shutil.copy2(loader, out_dir / "loadJoints.m")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tar-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw/kard"))
    ap.add_argument("--manifest", type=Path, default=Path("data/interim/extraction_manifest.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    manifest = extract(args.tar_dir, args.out_dir, args.overwrite)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    total = sum(len(a["extracted"]) for a in manifest["archives"].values())
    print(f"Total skeleton files: {total}. Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
