"""Automaticity measurement: does compute fall as competence rises?

Replays each training checkpoint on the SAME fixed-seed eval episodes and
records success vs mean exit depth — per encounter type when debug labels
are available (enemy visible vs not), else pooled.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from reflexrl.eval.evaluate import evaluate
from reflexrl.models.heads import ExitPolicy


def load_agent(backbone, ckpt_path: str | Path, n_actions: int = 6):
    from reflexrl.models.spinal import ReflexPolicy
    from reflexrl.rl.agent import ExitAgent, ReflexAgent

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hybrid = any(k.startswith("spinal.") for k in ckpt["policy"])
    if hybrid:
        policy = ReflexPolicy(n_actions=n_actions)
        policy.load_state_dict(ckpt["policy"])
        return ReflexAgent(backbone, policy).to(backbone.device), ckpt
    policy = ExitPolicy(n_actions=n_actions)
    policy.load_state_dict(ckpt["policy"])
    return ExitAgent(backbone, policy).to(backbone.device), ckpt


def encounter_type(debug: dict) -> str:
    ev = debug.get("enemies_visible")
    if ev is None:
        return "unknown"
    return "enemy_visible" if ev > 0 else "clear"


def automaticity_curve(backbone, run_dir: str | Path, n_episodes: int = 5,
                       seed: int = 20_000) -> dict:
    """For each ckpt_*.pt in run_dir: success rate + mean depth, overall and
    split by encounter type."""
    run_dir = Path(run_dir)
    curve = []
    for ckpt_path in sorted(run_dir.glob("ckpt_*.pt")):
        agent, ckpt = load_agent(backbone, ckpt_path)
        res = evaluate(agent, n_episodes=n_episodes, mode="router", seed=seed)
        by_type: dict[str, list[int]] = {}
        for s in res.steps:
            by_type.setdefault(encounter_type(s.debug), []).append(s.depth)
        curve.append({
            "checkpoint": ckpt_path.name,
            "decisions": ckpt.get("decisions"),
            "success_rate": float(np.mean([r > 0 for r in res.episode_returns])) if res.episode_returns else None,
            "mean_return": float(np.mean(res.episode_returns)) if res.episode_returns else None,
            "mean_depth": res.mean_depth,
            "depth_hist": res.depth_hist,
            "depth_by_encounter": {k: float(np.mean(v)) for k, v in by_type.items()},
            "mean_oracle_depth": res.mean_oracle_depth,
        })
    return {"run_dir": str(run_dir), "curve": curve}


def save_curve(curve: dict, out: str | Path):
    Path(out).write_text(json.dumps(curve, indent=2))
