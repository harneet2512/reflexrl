"""Pareto frontier aggregation.

Consumes eval result JSONs (one per condition/checkpoint) and emits the
frontier table: mean per-decision compute vs mean episode return per
condition, plus whether each point is dominated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def frontier_point(name: str, mean_return: float, cost_frac: float,
                   gflops: float | None = None, n_episodes: int = 0,
                   meta: dict | None = None) -> dict:
    return {
        "name": name,
        "mean_return": float(mean_return),
        "cost_frac": float(cost_frac),
        "gflops": gflops,
        "n_episodes": n_episodes,
        **(meta or {}),
    }


def is_dominated(points: list[dict]) -> list[bool]:
    """True if another point has >= return at strictly less compute."""
    dom = []
    for p in points:
        d = any(
            q["mean_return"] >= p["mean_return"] and q["cost_frac"] < p["cost_frac"]
            for q in points if q is not p
        )
        dom.append(d)
    return dom


def summarize_eval_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    return {
        "mean_return": float(np.mean(data["episode_returns"])),
        "n_episodes": len(data["episode_returns"]),
        "mean_depth": data.get("mean_depth"),
        "depth_hist": data.get("depth_hist"),
    }


def build_frontier(rows: list[dict], out: str | Path | None = None) -> list[dict]:
    dom = is_dominated(rows)
    for r, d in zip(rows, dom, strict=True):
        r["dominated"] = d
    rows.sort(key=lambda r: r["cost_frac"])
    if out:
        Path(out).write_text(json.dumps(rows, indent=2))
    return rows


def frontier_text(rows: list[dict]) -> str:
    lines = [f"{'condition':<22} {'return':>8} {'cost%':>7} {'GFLOPs':>8}  dominated"]
    for r in rows:
        lines.append(
            f"{r['name']:<22} {r['mean_return']:>8.2f} "
            f"{r['cost_frac'] * 100:>6.1f} {(r['gflops'] or 0):>8.1f}  {r['dominated']}"
        )
    return "\n".join(lines)
