"""Training-curve analysis from metrics.jsonl files.

Produces the learning story: episode return, mean depth, and compute
fraction vs environment decisions — the raw material for the
depth-over-training plot.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_metrics(workdir: str | Path) -> list[dict]:
    rows = []
    for line in Path(workdir, "metrics.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def curve(workdir: str | Path) -> dict:
    rows = load_metrics(workdir)
    return {
        "decisions": [r["decisions"] for r in rows],
        "ep_return_mean": [r["ep_return_mean"] for r in rows],
        "mean_depth": [r["mean_depth"] for r in rows],
        "mean_cost_frac": [r["mean_cost_frac"] for r in rows],
        "depth_hist": [r["depth_hist"] for r in rows],
        "entropy": [r["entropy"] for r in rows],
    }


def plot_curve(workdir: str | Path, out: str | Path | None = None):
    import matplotlib.pyplot as plt

    c = curve(workdir)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(c["decisions"], c["ep_return_mean"], color="tab:blue", label="return")
    ax1.set_xlabel("environment decisions")
    ax1.set_ylabel("episode return", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(c["decisions"], c["mean_cost_frac"], color="tab:red", label="compute")
    ax2.set_ylabel("mean compute fraction", color="tab:red")
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    out = Path(out or Path(workdir) / "curve.png")
    fig.savefig(out, dpi=140)
    return out
