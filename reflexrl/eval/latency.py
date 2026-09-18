"""End-to-end per-decision latency, measured the way the controller runs:
batch size 1, from the rendered frame(s) to a chosen action, GPU synchronised.

Student path: preprocess (resize + stack) -> CNN -> action.
Teacher path: PIL conversion + processor + Qwen forward -> action.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from reflexrl.env.observations import to_student_frame


def _summ(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"ms_mean": float(a.mean()), "ms_median": float(np.median(a)),
            "ms_p95": float(np.percentile(a, 95)), "actions_per_s": float(1000.0 / a.mean()),
            "n": len(a)}


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def student_latency(policy, frames: list[np.ndarray], device: str, n: int = 500,
                    warmup: int = 50) -> dict:
    """frames: raw full-res screens; the last 4 form the stack."""
    policy = policy.to(device).eval()
    ms = []
    for i in range(n + warmup):
        t0 = time.perf_counter()
        stack = np.concatenate([to_student_frame(f) for f in frames[-4:]], 0)
        obs = torch.as_tensor(stack[None], device=device)
        a = policy.act(obs)
        int(a[0])  # materialise the action on host
        _sync(device)
        if i >= warmup:
            ms.append((time.perf_counter() - t0) * 1000)
    out = _summ(ms)
    if device.startswith("cuda"):
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out


def teacher_latency(teacher, frames: list[np.ndarray], scenario, n: int = 50,
                    warmup: int = 5) -> dict:
    ms = []
    if teacher.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for i in range(n + warmup):
        t0 = time.perf_counter()
        probs = teacher.action_probs([frames[-2:]], scenario)
        int(np.argmax(probs[0]))
        if i >= warmup:
            ms.append((time.perf_counter() - t0) * 1000)
    out = _summ(ms)
    if teacher.device.startswith("cuda"):
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out


def cost_per_10k(ms_mean: float, usd_per_hour: float) -> float:
    """Dedicated-hardware cost of 10,000 sequential decisions at measured latency."""
    return 10_000 * ms_mean / 1000 / 3600 * usd_per_hour
