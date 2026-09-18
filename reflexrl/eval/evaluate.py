"""Evaluation loop shared by every analysis.

Runs a greedy agent on fresh env instances and records, per decision:
depth chosen, action, reward, all-exit argmax actions (for the oracle),
reflex-head entropy (for the confidence baseline), and the privileged
debug_state (encounter typing — logging only, never policy input).

Modes:
- "router": learned router picks d (ReflexRL eval)
- int d: forced depth (matched-compute points on the same checkpoint)
- "random": uniform depth
- ("confidence", tau): escalate to DEEP when reflex-head entropy > tau
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.distributions import Categorical

from reflexrl.envs.vizdoom_env import VizdoomPixelsEnv


@dataclass
class StepRecord:
    depth: int
    action: int
    reward: float
    oracle_depth: int  # cheapest exit whose argmax matches DEEP's argmax
    reflex_entropy: float
    done: bool
    debug: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    episode_returns: list[float]
    episode_lens: list[int]
    steps: list[StepRecord]
    n_pathways: int = 3

    @property
    def mean_depth(self) -> float:
        return float(np.mean([s.depth for s in self.steps])) if self.steps else 0.0

    @property
    def mean_oracle_depth(self) -> float:
        return float(np.mean([s.oracle_depth for s in self.steps])) if self.steps else 0.0

    @property
    def depth_hist(self) -> list[int]:
        return np.bincount([s.depth for s in self.steps],
                           minlength=self.n_pathways).tolist()


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    return Categorical(logits=logits).entropy()


@torch.no_grad()
def evaluate(agent, n_episodes: int = 5, mode="router", seed: int = 10_000,
             env_kwargs: dict | None = None, max_steps: int = 600,
             budget_check=None) -> EvalResult:
    env = VizdoomPixelsEnv(render_labels=True, max_steps=max_steps, **(env_kwargs or {}))
    env.reset(seed=seed)
    steps: list[StepRecord] = []
    returns, lens = [], []
    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len = 0.0, 0

    while len(returns) < n_episodes:
        if budget_check is not None:
            budget_check()
        h = agent.backbone.features_collect(obs[None])
        all_a = agent.all_action_logits_at(h, obs[None])
        # (n_pathways, 1, n_actions); row 0 is spinal for a hybrid agent
        argmax_by_exit = all_a.argmax(-1).squeeze(1)  # (n_pathways,)

        d_logits = agent.depth_logits_at(h, obs[None])
        if mode == "router":
            d = int(d_logits.argmax(-1).item())
        elif mode == "random":
            d = int(np.random.randint(all_a.size(0)))
        elif isinstance(mode, tuple) and mode[0] == "confidence":
            d = 0 if _entropy(all_a[0]).item() <= mode[1] else all_a.size(0) - 1
        else:
            d = int(mode)

        deep_action = int(argmax_by_exit[-1].item())
        oracle_d = min(i for i, a in enumerate(argmax_by_exit.tolist()) if a == deep_action)
        a = int(all_a[d].argmax(-1).item())

        next_obs, r, term, trunc, info = env.step(a)
        steps.append(StepRecord(
            depth=d, action=a, reward=r, oracle_depth=oracle_d,
            reflex_entropy=_entropy(all_a[0]).item(), done=term or trunc,
            debug=env.debug_state(),
        ))
        ep_ret += r
        ep_len += 1
        obs = next_obs
        if term or trunc:
            returns.append(ep_ret)
            lens.append(ep_len)
            ep_ret, ep_len = 0.0, 0
            obs, _ = env.reset()

    env.close()
    return EvalResult(returns, lens, steps, n_pathways=all_a.size(0))


def confidence_tau_for_target(agent, target_cost_frac: float, cost_vector,
                              n_episodes: int = 3, seed: int = 10_000) -> float:
    """Calibrate the confidence baseline's tau so its mean cost matches a
    target fraction of DEEP — the matched-compute comparison."""
    res = evaluate(agent, n_episodes=n_episodes, mode=0, seed=seed)
    ents = np.array([s.reflex_entropy for s in res.steps])
    # P(escalate) needed: mean_cost = c0 + P_esc * (c_deep - c0)
    c0, cdeep = cost_vector[0], cost_vector[-1]
    p_esc = min(max((target_cost_frac - c0) / max(cdeep - c0, 1e-9), 0.0), 1.0)
    return float(np.quantile(ents, 1 - p_esc))
