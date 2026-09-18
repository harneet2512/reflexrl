"""Measured latency: real ms/decision per exit on the deployment path.

``features_to`` is the honest forward (it actually stops at the chosen
exit); ``features_collect`` is the training/analysis path. Both are timed
here with CUDA events (or wall clock on CPU). This is the number that
decides L4 vs A10G in Phase 0.
"""

from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def time_forward(backbone, obs: np.ndarray, mode: str, exit_idx: int | None = None,
                 n_warm: int = 5, n_iter: int = 30) -> dict:
    """Mean/median/p95 ms per call."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def once():
        if mode == "collect":
            backbone.features_collect(obs)
        else:
            backbone.features_to(obs, exit_idx)

    for _ in range(n_warm):
        once()
    if dev == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        once()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(ts)
    return {
        "mode": mode, "exit_idx": exit_idx, "n": n_iter,
        "ms_mean": float(arr.mean()), "ms_median": float(np.median(arr)),
        "ms_p95": float(np.percentile(arr, 95)),
    }


@torch.no_grad()
def profile_all(backbone, obs: np.ndarray) -> list[dict]:
    """One row per exit (truncated path) + the collect path."""
    rows = []
    n_exits = 3
    for i in range(n_exits):
        rows.append(time_forward(backbone, obs, mode="to", exit_idx=i))
    rows.append(time_forward(backbone, obs, mode="collect"))
    return rows
