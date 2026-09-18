"""CPU end-to-end smoke: StubEncoder + real ViZDoom + full PPO loop.

Exercises everything the paid run will touch — rollout, GAE, PPO update,
checkpoints, metrics — with a pixel-dependent stub encoder instead of
Qwen. The house rule: this must be green before any GPU spend.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from reflexrl.models.heads import ExitPolicy
from reflexrl.models.qwen_vl import StubEncoder
from reflexrl.rl.agent import ExitAgent
from reflexrl.rl.trainer import TrainConfig, train


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="reflexrl_smoke_"))
    torch.manual_seed(0)
    agent = ExitAgent(StubEncoder(), ExitPolicy(n_actions=6))
    cfg = TrainConfig(
        n_envs=4,
        rollout_steps=64,
        total_decisions=512,
        warmup_decisions=256,
        ckpt_at=(256, 512),
        minibatch_size=128,
        epochs=2,
        device="cpu",
        tag="smoke",
    )
    history = train(agent, cfg, workdir)

    ckpts = sorted(workdir.glob("ckpt_*.pt"))
    metrics = (workdir / "metrics.jsonl").read_text().strip().splitlines()
    assert (workdir / "config.json").exists(), "config.json missing"
    assert len(ckpts) == 3, f"expected 2 milestones + final, got {ckpts}"
    assert len(metrics) == 2, f"expected 2 iterations logged, got {len(metrics)}"
    assert any(r is not None for r in history["ep_returns"]), "no episodes completed"

    import json
    row = json.loads(metrics[-1])
    assert row["decisions"] == 512
    assert sum(row["depth_hist"]) == 256  # per-iteration: rollout_steps * n_envs
    print(f"SMOKE OK workdir={workdir}")
    print(f"  iterations={len(metrics)} checkpoints={[c.name for c in ckpts]}")
    print(f"  ep_return_mean={row['ep_return_mean']} mean_depth={row['mean_depth']:.2f} "
          f"depth_hist={row['depth_hist']} cost_frac={row['mean_cost_frac']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
