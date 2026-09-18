"""The reflex policy: a small CNN over a short stack of downsampled RGB frames.

Temporal context comes from the 4-frame stack (motion is visible across
frames); there is no recurrent state, which keeps PPO minibatching simple and
inference a single conv pass. Roughly 0.8M parameters.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from reflexrl.env.observations import STUDENT_H, STUDENT_STACK, STUDENT_W


def _init(layer: nn.Module, gain: float = 2 ** 0.5) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3 * STUDENT_STACK, feat: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            _init(nn.Conv2d(in_ch, 32, 8, 4)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_ch, STUDENT_H, STUDENT_W)).shape[1]
        self.fc = nn.Sequential(_init(nn.Linear(n_flat, feat)), nn.ReLU())

    def forward(self, obs_u8: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(obs_u8.float() / 255.0))


class ActorCritic(nn.Module):
    def __init__(self, n_actions: int, feat: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.encoder = Encoder(feat=feat)
        self.pi = _init(nn.Linear(feat, n_actions), gain=0.01)
        self.v = _init(nn.Linear(feat, 1), gain=1.0)

    def forward(self, obs_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs_u8)
        return self.pi(z), self.v(z).squeeze(-1)

    def dist(self, obs_u8: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs_u8)[0])

    @torch.no_grad()
    def act(self, obs_u8: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        logits, _ = self.forward(obs_u8)
        return logits.argmax(-1) if greedy else Categorical(logits=logits).sample()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
