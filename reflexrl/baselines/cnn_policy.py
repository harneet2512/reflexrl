"""Small CNN specialist baseline.

A conventional Nature-DQN-sized conv policy trained end-to-end on ViZDoom
pixels. It exists to answer the fair question "why use Qwen at all?" — if
this beats the VLM agent on performance, compute, and adaptation, that is
a reported result, not a footnote.

Compute for this agent is a fixed constant (its own per-decision FLOPs,
measured in eval/latency.py); depth is always 0 and lam is set to 0 in the
run config since the penalty is uniform.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class CNNSpecialist(nn.Module):
    stores = "obs"

    def __init__(self, n_actions: int, frame_stack: int = 2):
        super().__init__()
        c = 3 * frame_stack
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, 180, 320)).numel()
        self.fc = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU())
        self.pi = nn.Linear(512, n_actions)
        self.v = nn.Linear(512, 1)

    def _features(self, obs_u8: torch.Tensor) -> torch.Tensor:
        # (B, K, H, W, 3) uint8 -> (B, K*3, H, W) float
        x = obs_u8.float() / 255.0
        b, k, h, w, c = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(b, k * c, h, w)
        return self.fc(self.conv(x))

    @torch.no_grad()
    def act(self, obs: np.ndarray, forced_d=None, greedy: bool = False) -> dict:
        f = self._features(torch.as_tensor(obs))
        dist = Categorical(logits=self.pi(f))
        a = torch.argmax(dist.logits, -1) if greedy else dist.sample()
        return {
            "action": a,
            "depth": torch.zeros_like(a),
            "logp": dist.log_prob(a),
            "value": self.v(f).squeeze(-1),
        }

    @torch.no_grad()
    def bootstrap_value(self, obs: np.ndarray) -> torch.Tensor:
        f = self._features(torch.as_tensor(obs))
        return self.v(f).squeeze(-1).cpu()

    def evaluate(self, mb: dict) -> dict:
        f = self._features(mb["obs"])
        dist = Categorical(logits=self.pi(f))
        return {
            "logp": dist.log_prob(mb["action"]),
            "entropy": dist.entropy(),
            "value": self.v(f).squeeze(-1),
        }

    def trainable_parameters(self):
        return self.parameters()


def cnn_flops_per_decision(frame_stack: int = 2, h: int = 180, w: int = 320) -> float:
    """Analytic conv FLOPs, same accounting convention as eval/compute.py."""
    c = 3 * frame_stack
    h1, w1 = (h - 8) // 4 + 1, (w - 8) // 4 + 1
    h2, w2 = (h1 - 4) // 2 + 1, (w1 - 4) // 2 + 1
    h3, w3 = h2 - 2, w2 - 2
    conv = (
        2 * h1 * w1 * 32 * c * 64
        + 2 * h2 * w2 * 64 * 32 * 16
        + 2 * h3 * w3 * 64 * 64 * 9
    )
    flat = 64 * h3 * w3
    return float(conv + 2 * flat * 512 + 2 * 512 * 8)
