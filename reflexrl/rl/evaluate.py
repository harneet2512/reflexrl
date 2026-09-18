"""Student-only evaluation: the number every learning curve is built from.

Runs the policy alone (no teacher, no intervention) on a fixed set of eval
seeds that no training env uses. Stochastic actions by default, matching how
the policy is deployed in the demo.
"""

from __future__ import annotations

import numpy as np
import torch

from reflexrl.env.vizdoom_env import DoomEnv

EVAL_SEED_BASE = 900_000


@torch.no_grad()
def evaluate_policy(policy, scenario: str, n_episodes: int, device: str,
                    greedy: bool = False, seed_base: int = EVAL_SEED_BASE,
                    n_envs: int = 8) -> dict:
    n_envs = min(n_envs, n_episodes)
    envs = [DoomEnv(scenario, seed=seed_base + i) for i in range(n_envs)]
    obs = [e.reset()[0] for e in envs]
    active = [True] * n_envs
    started = n_envs
    returns, lengths = [], []
    was_training = policy.training
    policy.eval()
    while any(active):
        idx = [i for i in range(n_envs) if active[i]]
        batch = torch.as_tensor(np.stack([obs[i] for i in idx]), device=device)
        actions = policy.act(batch, greedy=greedy).cpu().numpy()
        for a, i in zip(actions, idx, strict=True):
            obs[i], _, term, trunc, info = envs[i].step(int(a))
            if term or trunc:
                returns.append(info["episode"]["r"])
                lengths.append(info["episode"]["l"])
                if started < n_episodes:
                    obs[i] = envs[i].reset()[0]
                    started += 1
                else:
                    active[i] = False
    for e in envs:
        e.close()
    policy.train(was_training)
    r = np.asarray(returns, dtype=np.float64)
    return {"return_mean": float(r.mean()),
            "return_se": float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0,
            "len_mean": float(np.mean(lengths)), "n": len(r), "returns": r.tolist()}
