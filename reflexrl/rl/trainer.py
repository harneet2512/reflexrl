"""Training loop: rollout -> GAE -> PPO -> metrics -> checkpoints.

Every run writes config.json + metrics.jsonl + checkpoints under
workdir/. Checkpoints land at fixed env-step milestones so the
automaticity eval can replay them.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from reflexrl.envs.vizdoom_env import VecEnvs
from reflexrl.eval.compute import default_model
from reflexrl.rl.ppo import ppo_update
from reflexrl.rl.rollout import collect_rollout, compute_gae


@dataclass
class TrainConfig:
    scenario: str = "defend_the_center.cfg"
    n_envs: int = 8
    rollout_steps: int = 256
    total_decisions: int = 300_000
    frame_skip: int = 4
    frame_stack: int = 2
    seed: int = 0
    # compute cost
    lam: float = 0.1
    warmup_decisions: int = 20_000
    # ppo
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 512
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    # fixed-depth baseline: None = learned router, else 0/1/2
    forced_d: int | None = None
    # random-depth baseline: heads trained under uniform depth sampling
    random_depth: bool = False
    # spinal arc present: pathway 0 is a pixel convnet (hybrid store)
    spinal: bool = False
    # checkpoint milestones in env decisions
    ckpt_at: tuple = (10_000, 50_000, 150_000, 300_000)
    device: str = "cpu"
    tag: str = "run"


def _agent_state(agent) -> dict:
    """state_dict of the trainable policy — agent.policy if the agent wraps
    one (ExitAgent), else the agent itself (CNNSpecialist)."""
    return (agent.policy if getattr(agent, "policy", None) is not None else agent
            ).state_dict()


def train(agent, cfg: TrainConfig, workdir: str | Path,
          budget_check=None) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    envs = VecEnvs(
        cfg.n_envs, config=cfg.scenario, frame_skip=cfg.frame_skip,
        frame_stack=cfg.frame_stack, seed=cfg.seed, render_labels=False,
    )
    model = default_model(n_frames=cfg.frame_stack)
    cost_per_exit = model.cost_vector()
    if getattr(agent, "stores", "features") == "hybrid":
        from reflexrl.models.spinal import spinal_flops_per_decision
        deep_flops = model.flops(len(model.exit_layers) - 1)
        cost_per_exit = [
            spinal_flops_per_decision(cfg.frame_stack) / deep_flops
        ] + cost_per_exit
    opt = torch.optim.Adam(agent.trainable_parameters(), lr=cfg.lr)

    (workdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    metrics_f = open(workdir / "metrics.jsonl", "a")

    decisions = 0
    iteration = 0
    ckpt_iter = iter(sorted(cfg.ckpt_at))
    next_ckpt = next(ckpt_iter, None)
    history = {"ep_returns": [], "mean_depth": [], "iterations": []}
    t0 = time.time()

    while decisions < cfg.total_decisions:
        warmup = (cfg.random_depth or decisions < cfg.warmup_decisions) and cfg.forced_d is None
        store, next_value, stats = collect_rollout(
            agent, envs, cfg.rollout_steps, cost_per_exit, cfg.lam,
            forced_d=cfg.forced_d, warmup_uniform_d=warmup, device=cfg.device,
        )
        adv, ret = compute_gae(store, next_value, cfg.gamma, cfg.gae_lambda)
        logs = ppo_update(
            agent, store, adv, ret, opt, clip=cfg.clip, epochs=cfg.epochs,
            minibatch_size=cfg.minibatch_size, vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef,
        )
        decisions += cfg.rollout_steps * cfg.n_envs
        iteration += 1

        row = {
            "iter": iteration,
            "decisions": decisions,
            "ep_return_mean": float(np.mean(stats["ep_returns"])) if stats["ep_returns"] else None,
            "ep_len_mean": float(np.mean(stats["ep_lens"])) if stats["ep_lens"] else None,
            "n_episodes": len(stats["ep_returns"]),
            "mean_depth": stats["mean_depth"],
            "depth_hist": stats["depth_hist"],
            "mean_cost_frac": stats["mean_cost_frac"],
            "pg_loss": logs["pg"], "v_loss": logs["v"], "entropy": logs["ent"],
            "approx_kl": logs["kl"], "clipfrac": logs["clipfrac"],
            "warmup": warmup,
            "wall_s": time.time() - t0,
        }
        metrics_f.write(json.dumps(row) + "\n")
        metrics_f.flush()
        history["ep_returns"].append(row["ep_return_mean"])
        history["mean_depth"].append(row["mean_depth"])
        history["iterations"].append(iteration)

        if next_ckpt is not None and decisions >= next_ckpt:
            torch.save(
                {"policy": _agent_state(agent), "decisions": decisions,
                 "iter": iteration, "config": asdict(cfg)},
                workdir / f"ckpt_{next_ckpt}.pt",
            )
            next_ckpt = next(ckpt_iter, None)

        stop_reason = None
        if budget_check is not None:
            try:
                accrued = budget_check()
            except Exception as e:
                stop_reason = str(e)
            else:
                metrics_f.write(json.dumps({"iter": iteration, "accrued_usd": accrued}) + "\n")
                metrics_f.flush()
        if stop_reason:
            break

    metrics_f.close()
    envs.close()
    torch.save(
        {"policy": _agent_state(agent), "decisions": decisions,
         "config": asdict(cfg)},
        workdir / "ckpt_final.pt",
    )
    (workdir / "done.json").write_text(json.dumps(
        {"decisions": decisions, "wall_s": time.time() - t0,
         "stopped_reason": stop_reason}))
    return history
