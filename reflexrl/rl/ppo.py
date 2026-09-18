"""PPO update over cached transitions.

The joint action (a, d) shares one advantage: the environment and the
compute bill are one decision. logp_joint = logp_a + logp_d; the ratio is
taken on the joint logprob.
"""

from __future__ import annotations

import torch
from torch import nn


def ppo_update(agent, store: dict, advantages: torch.Tensor, returns: torch.Tensor,
               optimizer: torch.optim.Optimizer, clip: float = 0.2, epochs: int = 4,
               minibatch_size: int = 512, vf_coef: float = 0.5, ent_coef: float = 0.01,
               max_grad_norm: float = 0.5, target_kl: float | None = None) -> dict:
    T, B = store["action"].shape
    n = T * B

    flat = {
        "action": store["action"].reshape(n),
        "depth": store["depth"].reshape(n),
        "logp": store["logp"].reshape(n),
        "value": store["value"].reshape(n),
    }
    if store.get("obs") is not None:
        flat["obs"] = store["obs"].reshape(n, *store["obs"].shape[2:])
    if store.get("h_stack") is not None:
        flat["h_stack"] = store["h_stack"].reshape(n, *store["h_stack"].shape[2:])
    adv = advantages.reshape(n)
    ret = returns.reshape(n)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    device = next(agent.parameters()).device
    idx = torch.arange(n)
    logs = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "n_updates": 0}

    for _ in range(epochs):
        perm = idx[torch.randperm(n)]
        for start in range(0, n, minibatch_size):
            mb_i = perm[start : start + minibatch_size]
            mb = {k: v[mb_i].to(device) for k, v in flat.items()}
            out = agent.evaluate(mb)

            logratio = out["logp"] - mb["logp"]
            ratio = logratio.exp()
            mb_adv, mb_ret = adv[mb_i].to(device), ret[mb_i].to(device)

            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * ratio.clamp(1 - clip, 1 + clip)
            pg_loss = torch.max(pg1, pg2).mean()

            v_clip = mb["value"] + (out["value"] - mb["value"]).clamp(-clip, clip)
            v_loss = 0.5 * torch.max(
                (out["value"] - mb_ret) ** 2, (v_clip - mb_ret) ** 2
            ).mean()

            loss = pg_loss - ent_coef * out["entropy"].mean() + vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.trainable_parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = ((ratio - 1) - logratio).mean().item()
                clipfrac = ((ratio - 1).abs() > clip).float().mean().item()
            logs["pg"] += pg_loss.item()
            logs["v"] += v_loss.item()
            logs["ent"] += out["entropy"].mean().item()
            logs["kl"] += kl
            logs["clipfrac"] += clipfrac
            logs["n_updates"] += 1

        if target_kl is not None and kl > target_kl:
            break

    m = max(logs["n_updates"], 1)
    return {k: (v / m if k != "n_updates" else v) for k, v in logs.items()}
