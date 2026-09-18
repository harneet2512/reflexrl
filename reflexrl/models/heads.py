"""Trainable heads for ReflexRL.

Everything the optimizer touches lives here: per-exit norms, action heads,
the compute router, and the value head. All inputs are backbone hidden
states at the action position; the backbone is frozen, so these are the
only parameters PPO sees.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _mlp(hidden: int, mid: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(hidden, mid), nn.GELU(), nn.Linear(mid, out))


class ExitPolicy(nn.Module):
    """Router + per-exit action heads + value head on fixed backbone features.

    ``h_*`` inputs are backbone hidden states at the action position,
    detached: the backbone never sees gradients.
    """

    DEPTHS = (9, 19, 28)

    def __init__(self, hidden: int = 2048, n_actions: int = 6, depths: tuple = DEPTHS):
        super().__init__()
        self.depths = tuple(depths)
        self.exit_norms = nn.ModuleList(RMSNorm(hidden) for _ in self.depths)
        self.action_heads = nn.ModuleList(_mlp(hidden, 256, n_actions) for _ in self.depths)
        self.router = _mlp(hidden, 128, len(self.depths))
        self.value = _mlp(hidden, 256, 1)

    def head_out(self, h: torch.Tensor, exit_idx: int) -> torch.Tensor:
        """Action logits from exit `exit_idx`'s normalized hidden state."""
        return self.action_heads[exit_idx](self.exit_norms[exit_idx](h))

    def depth_logits(self, h_reflex: torch.Tensor) -> torch.Tensor:
        return self.router(self.exit_norms[0](h_reflex))

    def value_of(self, h_reflex: torch.Tensor) -> torch.Tensor:
        return self.value(self.exit_norms[0](h_reflex)).squeeze(-1)

    def action_logits_at(self, h_d: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Per-sample action logits: row i uses the head at depth d[i]."""
        outs = [self.head_out(h_d[i : i + 1], int(di)) for i, di in enumerate(d.tolist())]
        return torch.cat(outs, dim=0)

    def all_action_logits(self, h_by_exit: list[torch.Tensor]) -> torch.Tensor:
        """Action logits at every exit. h_by_exit[i] = (B, hidden) at exit i.

        Returns (n_exits, B, n_actions). Used by oracle/counterfactual evals.
        """
        return torch.stack(
            [self.head_out(h_by_exit[i], i) for i in range(len(self.depths))], dim=0
        )
