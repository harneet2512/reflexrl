"""Behaviour cloning from Qwen labels, and the frozen teacher proxy built from it.

BC fits the reflex architecture to Qwen's action distributions (soft-label
cross-entropy = KL to pi_T). The same fitted network, frozen, serves as the
fast teacher proxy during guided RL, so Qwen is only ever run at labelling
time; its fidelity to Qwen is measured on held-out labelled episodes.
"""

from __future__ import annotations

import numpy as np
import torch

from reflexrl.policy.actor_critic import ActorCritic


def _split_by_episode(episode: np.ndarray, holdout_frac: float, seed: int):
    eps = np.unique(episode)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(eps, size=max(1, int(len(eps) * holdout_frac)), replace=False))
    val = np.isin(episode, list(held))
    return ~val, val


def train_bc(labels: dict, n_actions: int, device: str = "cuda", epochs: int = 10,
             batch: int = 256, lr: float = 3e-4, seed: int = 0,
             holdout_frac: float = 0.15) -> tuple[ActorCritic, dict]:
    torch.manual_seed(seed)
    obs = torch.as_tensor(labels["obs"])
    target = torch.as_tensor(labels["probs"].astype(np.float32))
    target = target / target.sum(1, keepdim=True)
    tr, va = _split_by_episode(labels["episode"], holdout_frac, seed)
    tr_idx, va_idx = np.flatnonzero(tr), np.flatnonzero(va)

    policy = ActorCritic(n_actions).to(device)
    opt = torch.optim.Adam(list(policy.encoder.parameters()) + list(policy.pi.parameters()), lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(tr_idx)
        for k in range(0, len(tr_idx), batch):
            j = tr_idx[k:k + batch]
            logp = torch.log_softmax(policy(obs[j].to(device))[0], -1)
            loss = -(target[j].to(device) * logp).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    fid = fidelity(policy, obs[va_idx], target[va_idx], device)
    fid.update({"n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "epochs": epochs})
    return policy, fid


@torch.no_grad()
def fidelity(policy: ActorCritic, obs: torch.Tensor, target: torch.Tensor, device: str) -> dict:
    """How faithfully the network reproduces Qwen on held-out episodes."""
    kls, agree = [], []
    for k in range(0, len(obs), 512):
        logp = torch.log_softmax(policy(obs[k:k + 512].to(device))[0], -1)
        t = target[k:k + 512].to(device)
        kls.append((t * (torch.log(t + 1e-8) - logp)).sum(-1))
        agree.append((logp.argmax(-1) == t.argmax(-1)).float())
    return {"val_kl": float(torch.cat(kls).mean()), "val_top1_agree": float(torch.cat(agree).mean())}


class ProxyTeacher:
    """Frozen BC network standing in for Qwen at training time."""

    def __init__(self, policy: ActorCritic, device: str = "cuda"):
        self.policy = policy.to(device).eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.device = device
        self.calls = 0

    @torch.no_grad()
    def probs(self, obs_u8: np.ndarray) -> np.ndarray:
        self.calls += len(obs_u8)
        logits = self.policy(torch.as_tensor(obs_u8, device=self.device))[0]
        return torch.softmax(logits, -1).cpu().numpy()
