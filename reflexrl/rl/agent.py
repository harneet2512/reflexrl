"""ExitAgent: backbone + ExitPolicy glued into an act/evaluate interface.

Rollout calls ``act``; PPO calls ``evaluate`` on cached features. The
backbone is only ever used for inference inside ``act`` — ``evaluate``
recomputes logprobs from stored hidden states, so updates never backprop
through Qwen.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from reflexrl.models.heads import ExitPolicy


class ExitAgent(torch.nn.Module):
    def __init__(self, backbone, policy: ExitPolicy):
        super().__init__()
        self.backbone = backbone  # features_collect / features_to
        self.policy = policy

    stores = "features"

    def trainable_parameters(self):
        return self.policy.parameters()

    @torch.no_grad()
    def bootstrap_value(self, obs: np.ndarray) -> torch.Tensor:
        h = self.backbone.features_collect(obs)
        return self.policy.value_of(h[:, 0]).cpu()

    @torch.no_grad()
    def act(self, obs: np.ndarray, forced_d: torch.Tensor | int | None = None,
            greedy: bool = False) -> dict:
        """obs (B,K,H,W,3) -> joint action sample.

        Returns tensors (B,): action, depth, logp_joint, value, plus the
        (B,3,H) feature stack for the buffer.
        """
        h = self.backbone.features_collect(obs)  # (B,3,hidden)
        h_reflex = h[:, 0]
        d_logits = self.policy.depth_logits(h_reflex)
        d_dist = Categorical(logits=d_logits)

        if forced_d is None:
            d = torch.argmax(d_logits, -1) if greedy else d_dist.sample()
        elif isinstance(forced_d, int):
            d = torch.full_like(torch.argmax(d_logits, -1), forced_d)
        else:
            d = forced_d.to(h_reflex.device)

        all_a = self.policy.all_action_logits([h[:, i] for i in range(h.size(1))])
        a_logits = all_a.gather(
            0, d.view(1, -1, 1).expand(1, -1, all_a.size(-1))
        ).squeeze(0)
        a_dist = Categorical(logits=a_logits)
        a = torch.argmax(a_logits, -1) if greedy else a_dist.sample()

        logp = a_dist.log_prob(a) + d_dist.log_prob(d)
        return {
            "action": a,
            "depth": d,
            "logp": logp,
            "value": self.policy.value_of(h_reflex),
            "h_stack": h,
            "d_logits": d_logits,
        }

    def evaluate(self, mb: dict) -> dict:
        """logp/entropy/value for a minibatch of cached transitions."""
        h_reflex = mb["h_stack"][:, 0]
        d = mb["depth"]
        all_a = self.policy.all_action_logits(
            [mb["h_stack"][:, i] for i in range(mb["h_stack"].size(1))]
        )
        a_logits = all_a.gather(
            0, d.view(1, -1, 1).expand(1, -1, all_a.size(-1))
        ).squeeze(0)
        d_logits = self.policy.depth_logits(h_reflex)
        a_dist = Categorical(logits=a_logits)
        d_dist = Categorical(logits=d_logits)

        return {
            "logp": a_dist.log_prob(mb["action"]) + d_dist.log_prob(d),
            "entropy": a_dist.entropy() + d_dist.entropy(),
            "value": self.policy.value_of(h_reflex),
        }

    # ---- eval interface (features already collected by the caller) ------

    def all_action_logits_at(self, h: torch.Tensor, obs=None) -> torch.Tensor:
        """(n_pathways, B, n_actions) action logits at every pathway."""
        return self.policy.all_action_logits(
            [h[:, i] for i in range(h.size(1))]
        )

    def depth_logits_at(self, h: torch.Tensor, obs=None) -> torch.Tensor:
        """(B, n_pathways) router logits."""
        return self.policy.depth_logits(h[:, 0])


class ReflexAgent(ExitAgent):
    """ExitAgent with a spinal arc: pathway 0 is a pixel convnet, 1..3 are
    the truncated-Qwen exits.

    The router reads spinal features, so a SPINAL decision honestly costs
    ~0.01% of DEEP — no cortical forward is needed to stay reflexive.
    Training still runs features_collect for the buffer (the declared
    depth is what the analytic cost charges), matching ExitAgent.
    """

    stores = "hybrid"  # buffer keeps obs (spinal) AND h_stack (cortical heads)

    def __init__(self, backbone, policy):
        super().__init__(backbone, policy)
        self.spinal = policy.spinal
        self.exit_policy = policy.exit_policy

    def _z(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.spinal.features(obs_t.to(next(self.parameters()).device))

    def _combine_logits(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """(n_pathways, B, n_actions): row 0 spinal, rows 1..3 VLM exits."""
        vlm = self.exit_policy.all_action_logits(
            [h[:, i] for i in range(h.size(1))]
        )
        return torch.cat([self.spinal.pi(z).unsqueeze(0), vlm], dim=0)

    @torch.no_grad()
    def act(self, obs: np.ndarray, forced_d: torch.Tensor | int | None = None,
            greedy: bool = False) -> dict:
        z = self._z(torch.as_tensor(obs))
        d_logits = self.spinal.router(z)  # (B, n_pathways)
        d_dist = Categorical(logits=d_logits)

        if forced_d is None:
            d = torch.argmax(d_logits, -1) if greedy else d_dist.sample()
        elif isinstance(forced_d, int):
            d = torch.full_like(torch.argmax(d_logits, -1), forced_d)
        else:
            d = forced_d.to(z.device)

        h = self.backbone.features_collect(obs)  # (B, 3, hidden)
        all_a = self._combine_logits(z, h)
        a_logits = all_a.gather(
            0, d.view(1, -1, 1).expand(1, -1, all_a.size(-1))
        ).squeeze(0)
        a_dist = Categorical(logits=a_logits)
        a = torch.argmax(a_logits, -1) if greedy else a_dist.sample()

        return {
            "action": a,
            "depth": d,
            "logp": a_dist.log_prob(a) + d_dist.log_prob(d),
            "value": self.policy.value_of(h[:, 0]),
            "h_stack": h,
            "d_logits": d_logits,
        }

    def evaluate(self, mb: dict) -> dict:
        z = self._z(mb["obs"])
        d = mb["depth"]
        d_logits = self.spinal.router(z)
        all_a = self._combine_logits(z, mb["h_stack"])
        a_logits = all_a.gather(
            0, d.view(1, -1, 1).expand(1, -1, all_a.size(-1))
        ).squeeze(0)
        a_dist = Categorical(logits=a_logits)
        d_dist = Categorical(logits=d_logits)

        return {
            "logp": a_dist.log_prob(mb["action"]) + d_dist.log_prob(d),
            "entropy": a_dist.entropy() + d_dist.entropy(),
            "value": self.policy.value_of(mb["h_stack"][:, 0]),
        }

    def all_action_logits_at(self, h: torch.Tensor, obs=None) -> torch.Tensor:
        z = self._z(torch.as_tensor(obs))
        return self._combine_logits(z, h)

    def depth_logits_at(self, h: torch.Tensor, obs=None) -> torch.Tensor:
        z = self._z(torch.as_tensor(obs))
        return self.spinal.router(z)
