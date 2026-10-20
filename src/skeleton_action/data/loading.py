"""Load the parsed KARD collection from the interim cache, building it if needed."""

from __future__ import annotations

import json
from pathlib import Path

from skeleton_action.data.kard import KardCollection, load_collection
from skeleton_action.utils.logging import get_logger

log = get_logger()


def get_collection(dataset_cfg: dict, rebuild: bool = False) -> KardCollection:
    """Return the parsed collection described by a dataset config section.

    Uses ``dataset_cfg["interim_path"]`` as cache; parses ``dataset_cfg["raw_dir"]`` when
    the cache is missing (or ``rebuild``). Rejected files are written next to the cache.
    """
    kind = dataset_cfg.get("kind", "realworld")
    interim = Path(dataset_cfg.get("interim_path", f"data/interim/kard_{kind}.npz"))
    if interim.exists() and not rebuild:
        col = KardCollection.load(interim)
        if col.kind != kind:
            raise ValueError(f"{interim} holds {col.kind!r} data, config asks for {kind!r}")
        return col
    raw = Path(dataset_cfg.get("raw_dir", "data/raw/kard"))
    if not raw.exists() or not any(raw.rglob(f"*_{kind}.txt")):
        raise FileNotFoundError(
            f"No *_{kind}.txt files under {raw} and no cache at {interim}. "
            "Run `make extract-data` and `make prepare-data` (docs/dataset_setup.md)."
        )
    col, rejected = load_collection(raw, kind)
    col.save(interim)
    interim.with_suffix(".rejected.json").write_text(json.dumps(rejected, indent=2))
    log.info(
        "parsed %d %s sequences from %s (%d rejected) -> %s",
        len(col),
        kind,
        raw,
        len(rejected),
        interim,
    )
    return col
