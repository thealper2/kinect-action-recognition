"""Model-forward latency measurement."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device | str = "cpu",
    warmup: int = 10,
    iters: int = 100,
    num_threads: int | None = None,
) -> dict:
    """Median / p95 / mean wall-clock latency (ms) of ``model(x)`` for a fixed input.

    CUDA timings synchronise before and after each call. ``num_threads`` (CPU) is
    restored afterwards.
    """
    device = torch.device(device)
    prev_threads = torch.get_num_threads()
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    try:
        model = model.to(device).eval()
        x = torch.randn(*input_shape, device=device)
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    finally:
        torch.set_num_threads(prev_threads)
    t = np.array(times)
    return {
        "device": str(device),
        "batch_size": int(input_shape[0]),
        "threads": num_threads or prev_threads,
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_ms": float(t.mean()),
        "iters": iters,
    }
