"""Local training driver: runs any condition on this machine.

Qwen backbone needs a GPU this box doesn't have, so local runs use the
StubEncoder (pipeline work) or the CNN specialist (the $0 baseline, which
can train on the RTX 2060 or CPU).

Usage:
    python scripts/train.py --agent stub --decisions 20000 --workdir runs/stub0
    python scripts/train.py --agent cnn --decisions 300000 --workdir runs/cnn0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from reflexrl.models.heads import ExitPolicy
from reflexrl.models.qwen_vl import StubEncoder
from reflexrl.rl.agent import ExitAgent
from reflexrl.rl.trainer import TrainConfig, train


def build_agent(kind: str, n_actions: int, device: str):
    if kind == "stub":
        return ExitAgent(StubEncoder(), ExitPolicy(n_actions=n_actions))
    if kind == "cnn":
        from reflexrl.baselines.cnn_policy import CNNSpecialist
        return CNNSpecialist(n_actions=n_actions)
    if kind == "qwen":
        from reflexrl.models.qwen_vl import QwenVLBackbone
        return ExitAgent(QwenVLBackbone(device=device),
                         ExitPolicy(n_actions=n_actions))
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="stub", choices=["stub", "cnn", "qwen"])
    p.add_argument("--decisions", type=int, default=20_000)
    p.add_argument("--workdir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--forced-d", type=int, default=None)
    p.add_argument("--random-depth", action="store_true")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    agent = build_agent(args.agent, n_actions=6, device=args.device)
    cfg = TrainConfig(
        total_decisions=args.decisions, seed=args.seed, lam=args.lam,
        forced_d=args.forced_d, random_depth=args.random_depth,
        n_envs=args.n_envs, rollout_steps=args.rollout_steps,
        device=args.device, tag=Path(args.workdir).name,
        ckpt_at=tuple(x for x in (10_000, 50_000, 150_000, 300_000)
                      if x <= args.decisions),
    )
    train(agent, cfg, args.workdir)
    print(f"done -> {args.workdir}")


if __name__ == "__main__":
    main()
