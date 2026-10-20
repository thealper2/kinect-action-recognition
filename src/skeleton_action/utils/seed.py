"""Seeding, device selection and environment/version capture."""

from __future__ import annotations

import os
import platform
import random
import sys

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) and optionally request determinism."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seed the legacy global RNG for third-party code
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(name: str = "auto") -> torch.device:
    """``auto`` -> CUDA if available, else CPU. ``cuda``/``cuda:1``/``cpu`` explicit."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return dev


def software_versions() -> dict:
    import sklearn

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        # str(): torch.__version__ is a TorchVersion object, refused by weights_only loads.
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "sklearn": str(sklearn.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def rng_states() -> dict:
    """Capture global RNG states (Python, torch CPU/CUDA).

    The NumPy legacy global RNG is not used by this project (augmentation uses a
    dataset-owned ``Generator`` whose state is checkpointed separately), and its state
    contains arrays that ``torch.load(weights_only=True)`` would refuse.
    """
    st = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_states(st: dict) -> None:
    if "python" in st:
        s = st["python"]
        random.setstate((s[0], tuple(s[1]), s[2]))
    if "torch" in st:
        torch.set_rng_state(st["torch"].cpu())
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
