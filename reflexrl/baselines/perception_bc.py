"""Distil the teacher's *perception*, not its actions.

The teacher is pi_T(a|o) = sum_c q(c|o) J(a|c): Qwen says where the nearest
monster is, Jev turns that into an action. Cloning pi_T directly makes the
student solve a 5-way action problem; cloning q instead leaves it a 4-way
"where is the monster" problem and keeps Jev as the decision maker.

q is recovered from the stored pi_T labels by least squares (J has full row
rank), so this needs no extra teacher queries.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from reflexrl.baselines.bc import _split_by_episode
from reflexrl.policy.actor_critic import Encoder
from reflexrl.teacher.perception_jev import CLASSES


def recover_perception(pi_T: np.ndarray, J: np.ndarray) -> np.ndarray:
    """pi_T (N, A) and J (C, A) -> q (N, C), the simplex-projected least-squares fit."""
    q, *_ = np.linalg.lstsq(J.T, pi_T.T, rcond=None)
    q = np.clip(q.T, 0.0, None)
    return q / np.clip(q.sum(1, keepdims=True), 1e-8, None)


class PerceptionNet(nn.Module):
    """Student-sized encoder with a 4-way perception head."""

    def __init__(self, n_classes: int = len(CLASSES), feat: int = 256):
        super().__init__()
        self.encoder = Encoder(feat=feat)
        self.head = nn.Linear(feat, n_classes)

    def forward(self, obs_u8: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(obs_u8))


class PerceptionJevProxy:
    """Frozen perception student + the cached Jev table = the fast training-time teacher."""

    def __init__(self, net: PerceptionNet, J: np.ndarray, device: str = "cuda"):
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.J = torch.as_tensor(J, dtype=torch.float32, device=device)
        self.device = device
        self.calls = 0

    @torch.no_grad()
    def probs(self, obs_u8: np.ndarray) -> np.ndarray:
        self.calls += len(obs_u8)
        q = torch.softmax(self.net(torch.as_tensor(obs_u8, device=self.device)), -1)
        return (q @ self.J).cpu().numpy()


def train_perception_bc(labels: dict, J: np.ndarray, device: str = "cuda", epochs: int = 30,
                        batch: int = 128, lr: float = 3e-4, seed: int = 0,
                        holdout_frac: float = 0.15) -> tuple[PerceptionNet, dict]:
    torch.manual_seed(seed)
    obs = torch.as_tensor(labels["obs"])
    q = torch.as_tensor(recover_perception(labels["probs"].astype(np.float32), J), dtype=torch.float32)
    tr, va = _split_by_episode(labels["episode"], holdout_frac, seed)
    tr_idx, va_idx = np.flatnonzero(tr), np.flatnonzero(va)
    net = PerceptionNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        rng.shuffle(tr_idx)
        for k in range(0, len(tr_idx), batch):
            j = tr_idx[k:k + batch]
            logp = torch.log_softmax(net(obs[j].to(device)), -1)
            loss = -(q[j].to(device) * logp).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        acc = perception_accuracy(net, obs[va_idx], q[va_idx], device)
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, {"val_top1_agree": float(best), "epochs": epochs,
                 "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
                 "class_prior": q[torch.as_tensor(tr_idx)].mean(0).tolist()}


@torch.no_grad()
def perception_accuracy(net: PerceptionNet, obs: torch.Tensor, q: torch.Tensor, device: str) -> float:
    net.eval()
    hits = []
    for k in range(0, len(obs), 512):
        pred = net(obs[k:k + 512].to(device)).argmax(-1)
        hits.append((pred == q[k:k + 512].to(device).argmax(-1)).float())
    return float(torch.cat(hits).mean()) if hits else 0.0


class PerceptionJevPolicy(nn.Module):
    """Perception student + Jev table as a playable policy (for evaluation and as the
    training-time teacher stand-in). Deployment still uses the reflex policy, not this."""

    def __init__(self, net: PerceptionNet, J: np.ndarray):
        super().__init__()
        self.net = net
        self.register_buffer("J", torch.as_tensor(J, dtype=torch.float32))

    def action_probs(self, obs_u8: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(obs_u8), -1) @ self.J

    @torch.no_grad()
    def act(self, obs_u8: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        p = self.action_probs(obs_u8)
        return p.argmax(-1) if greedy else torch.distributions.Categorical(probs=p).sample()
