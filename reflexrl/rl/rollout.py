"""Rollout collection for feature-cached policies.

Per decision the agent sees all exits' hidden states (one full-collect
pass), samples (a, d), and stores the feature stack — the reward's compute
term charges the *declared* depth, matching what a truncated deployment
forward would spend.
"""

from __future__ import annotations

import torch


def collect_rollout(agent, vec_envs, n_steps: int, cost_per_exit, lam: float,
                    forced_d=None, warmup_uniform_d: bool = False,
                    device: str = "cpu", debug_log: bool = False):
    """Collect n_steps of joint (a,d) decisions across vec_envs.

    Returns a dict of tensors shaped (n_steps, n_envs, ...) plus episode
    stats and the bootstrap value for the final observation.
    """
    n_envs = vec_envs.num_envs
    mode = getattr(agent, "stores", "features")
    store_obs = mode in ("obs", "hybrid")
    store_feat = mode in ("features", "hybrid")
    policy = getattr(agent, "policy", None)
    n_exits = len(policy.depths) if hasattr(policy, "depths") else 1
    # h_stack holds cortical features only: for a hybrid agent that's the
    # inner ExitPolicy's exits, not the spinal pathway.
    n_h = len(getattr(policy, "exit_policy", policy).depths)
    hidden = policy.exit_norms[0].weight.numel() if store_feat else 0

    store = {
        "obs": torch.zeros(n_steps, n_envs, *vec_envs.single_observation_space.shape,
                           dtype=torch.uint8) if store_obs else None,
        "h_stack": torch.zeros(n_steps, n_envs, n_h, hidden) if store_feat else None,
        "action": torch.zeros(n_steps, n_envs, dtype=torch.long),
        "depth": torch.zeros(n_steps, n_envs, dtype=torch.long),
        "logp": torch.zeros(n_steps, n_envs),
        "value": torch.zeros(n_steps, n_envs),
        "reward_env": torch.zeros(n_steps, n_envs),
        "terminated": torch.zeros(n_steps, n_envs, dtype=torch.bool),
        "truncated": torch.zeros(n_steps, n_envs, dtype=torch.bool),
        # V(final_obs) for truncated steps; zeros elsewhere.
        "final_v": torch.zeros(n_steps, n_envs),
    }
    ep_returns, ep_lens, dbg_rows = [], [], []

    obs = vec_envs.reset()
    for t in range(n_steps):
        with torch.no_grad():
            if warmup_uniform_d:
                d = torch.randint(0, n_exits, (n_envs,))
                out = agent.act(obs, forced_d=d)
            else:
                out = agent.act(obs, forced_d=forced_d)

        actions = out["action"].cpu().numpy()
        next_obs, rewards, terms, truncs, infos = vec_envs.step(actions)

        if store_obs:
            store["obs"][t] = torch.as_tensor(obs)
        if store_feat:
            store["h_stack"][t] = out["h_stack"].cpu()
        store["action"][t] = out["action"].cpu()
        store["depth"][t] = out.get("depth", torch.zeros(n_envs, dtype=torch.long)).cpu()
        store["logp"][t] = out["logp"].cpu()
        store["value"][t] = out["value"].cpu()
        store["reward_env"][t] = torch.as_tensor(rewards)
        store["terminated"][t] = torch.as_tensor(terms)
        store["truncated"][t] = torch.as_tensor(truncs)

        for i, info in enumerate(infos):
            if "episode_return" in info:
                ep_returns.append(info["episode_return"])
                ep_lens.append(info["episode_len"])
            if "final_obs" in info and truncs[i]:
                store["final_v"][t, i] = agent.bootstrap_value(info["final_obs"][None])
            if debug_log:
                dbg_rows.append(vec_envs.envs[i].debug_state())

        obs = next_obs

    with torch.no_grad():
        next_value = agent.bootstrap_value(obs)

    # Reward charged at declared depth: r = r_env - lambda * C(d)/C(DEEP)
    cost = torch.as_tensor(cost_per_exit, dtype=torch.float32)[store["depth"]]
    store["reward"] = store["reward_env"] - lam * cost
    store["cost_frac"] = cost

    stats = {
        "ep_returns": ep_returns,
        "ep_lens": ep_lens,
        "mean_depth": store["depth"].float().mean().item(),
        "depth_hist": torch.bincount(store["depth"].flatten(), minlength=n_exits).tolist(),
        "mean_cost_frac": store["cost_frac"].mean().item(),
        "debug": dbg_rows,
    }
    return store, next_value, stats


def compute_gae(store: dict, next_value: torch.Tensor, gamma: float, gae_lambda: float):
    """GAE over stored rewards/values.

    nextvalue per step: 0 if terminated, V(final_obs) if truncated,
    V(s_{t+1}) otherwise.
    """
    T, B = store["reward"].shape
    values = store["value"]
    advantages = torch.zeros_like(store["reward"])
    lastgaelam = torch.zeros(B)

    for t in reversed(range(T)):
        nv_next = next_value if t == T - 1 else values[t + 1]
        nextvalue = torch.where(
            store["terminated"][t],
            torch.zeros(B),
            torch.where(store["truncated"][t], store["final_v"][t], nv_next),
        )
        # delta bootstraps on anything but a true terminal; the advantage
        # chain stops at every episode boundary, terminal or truncated.
        delta_mask = (~store["terminated"][t]).float()
        prop_mask = (~store["terminated"][t] & ~store["truncated"][t]).float()
        delta = store["reward"][t] + gamma * nextvalue * delta_mask - values[t]
        lastgaelam = delta + gamma * gae_lambda * prop_mask * lastgaelam
        advantages[t] = lastgaelam

    return advantages, advantages + values
