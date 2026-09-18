"""Spinal arc: a cheap pixel pathway parallel to the cortical exits.

A deliberately small convnet reading raw frames, with its own action
head, value head, and the escalation router. Routing lives on spinal
features so that choosing SPINAL honestly costs ~0.01% of a DEEP forward
— no cortical computation is needed to stay in reflex.

The torso is intentionally weaker than the CNN specialist (fewer
channels, smaller fc): competent on routine scenes, limited on hard
ones — the capability gap that gives deliberation somewhere to earn its
keep.
"""

from __future__ import annotations

import torch
from torch import nn

from reflexrl.models.heads import ExitPolicy


class SpinalArc(nn.Module):
    """pixels -> z -> {action logits, value, router logits over pathways}."""

    def __init__(self, n_actions: int, n_pathways: int, frame_stack: int = 2,
                 z_dim: int = 96):
        super().__init__()
        c = 3 * frame_stack
        self.torso = nn.Sequential(
            nn.Conv2d(c, 16, 8, 4), nn.ReLU(),
            nn.Conv2d(16, 24, 4, 2), nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.torso(torch.zeros(1, c, 180, 320)).numel()
        self.fc = nn.Sequential(nn.Linear(n_flat, z_dim), nn.ReLU())
        self.pi = nn.Linear(z_dim, n_actions)
        self.v = nn.Linear(z_dim, 1)
        self.router = nn.Linear(z_dim, n_pathways)

    def features(self, obs_u8: torch.Tensor) -> torch.Tensor:
        # (B, K, H, W, 3) uint8 -> (B, z_dim) float, same layout as CNNSpecialist
        x = obs_u8.float() / 255.0
        b, k, h, w, c = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(b, k * c, h, w)
        return self.fc(self.torso(x))


class ReflexPolicy(nn.Module):
    """ExitPolicy + SpinalArc: depth space [spinal, *vlm_exits].

    ``depths`` is (0, 9, 19, 28): index 0 is the spinal pathway, indices
    1..3 are the truncated-Qwen exits. Everything downstream that reads
    ``len(policy.depths)`` — rollout buffers, depth histograms, eval
    modes — sees four pathways.
    """

    def __init__(self, hidden: int = 2048, n_actions: int = 6,
                 frame_stack: int = 2):
        super().__init__()
        self.exit_policy = ExitPolicy(hidden=hidden, n_actions=n_actions)
        self.spinal = SpinalArc(
            n_actions=n_actions,
            n_pathways=len(self.exit_policy.depths) + 1,
            frame_stack=frame_stack,
        )
        self.depths = (0,) + self.exit_policy.depths

    @property
    def exit_norms(self):
        return self.exit_policy.exit_norms

    def value_of(self, h_reflex: torch.Tensor) -> torch.Tensor:
        return self.exit_policy.value_of(h_reflex)


def spinal_flops_per_decision(frame_stack: int = 2, h: int = 180, w: int = 320,
                              z_dim: int = 96, n_actions: int = 6,
                              n_pathways: int = 4) -> float:
    """Analytic conv FLOPs, same convention as baselines.cnn_policy."""
    c = 3 * frame_stack
    h1, w1 = (h - 8) // 4 + 1, (w - 8) // 4 + 1
    h2, w2 = (h1 - 4) // 2 + 1, (w1 - 4) // 2 + 1
    h3, w3 = h2 - 2, w2 - 2
    conv = (
        2 * h1 * w1 * 16 * c * 64
        + 2 * h2 * w2 * 24 * 16 * 16
        + 2 * h3 * w3 * 24 * 24 * 9
    )
    flat = 24 * h3 * w3
    dense = 2 * flat * z_dim + 2 * z_dim * (n_actions + 1 + n_pathways)
    return float(conv + dense)
