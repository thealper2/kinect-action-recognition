"""Structured logging: human-readable console lines + machine-readable JSONL."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path


def get_logger(name: str = "skeleton_action", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class JsonlWriter:
    """Append one JSON object per line (``event`` + payload + wall-clock time)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload) -> None:
        rec = {"event": event, "time": time.time(), **payload}
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, default=_default) + "\n")


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)
