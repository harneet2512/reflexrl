"""Frozen Qwen3-VL backbone with early-exit access.

Two forward modes:

- ``features_collect``: one full pass returning hidden states at every
  exit depth. Used for training rollouts and counterfactual/oracle evals —
  all exits observed from a single forward.
- ``features_to``: truncated pass that stops at the chosen exit. Used for
  deployment and honest latency measurement.

Exit convention: ``hidden_states[k]`` is the capture at position k in the
returned tuple (transformers 5.x appends the post-norm state after the last
executed layer, so a truncated run's tuple ends at index n with the norm).
DEEP uses the post-norm final hidden state. DeepStack visual features are
injected after decoder layers 0-2, so every exit sees fused features.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Decoder-layer indices behind each exit. -1 = final (post-norm) hidden state.
REFLEX_LAYER, FAST_LAYER, DEEP_LAYER = 9, 19, -1
EXIT_LAYERS = (REFLEX_LAYER, FAST_LAYER, DEEP_LAYER)

PROMPT_SUFFIX = "\nAct."


def _n_run_layers(layer_idx: int, total: int) -> int:
    """Layers to execute for a hidden_states[layer_idx] target.

    transformers 5.x appends the post-norm state at index n of a truncated
    run, so index layer_idx is only a real layer capture if at least
    layer_idx + 1 layers ran.
    """
    return total if layer_idx == -1 else layer_idx + 1


class QwenVLBackbone(nn.Module):
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda", dtype=None):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        self.dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=self.dtype, device_map=device
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config.text_config
        self.n_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size

    # ---- input encoding -------------------------------------------------

    def encode(self, obs: np.ndarray) -> dict:
        """(B, K, H, W, 3) uint8 -> processor batch on device."""
        b, k = obs.shape[0], obs.shape[1]
        images = [obs[i, j] for i in range(b) for j in range(k)]
        block = "<|vision_start|><|image_pad|><|vision_end|>"
        text = [block * k + PROMPT_SUFFIX] * b
        batch = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        return {k2: v.to(self.device) if hasattr(v, "to") else v for k2, v in batch.items()}

    @staticmethod
    def _action_pos(attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.sum(1).long() - 1

    def _inner_forward(self, batch: dict) -> tuple:
        # Pass the full processor batch (input_ids, attention_mask,
        # pixel_values, image_grid_thw, mm_token_type_ids on transformers
        # >=5) rather than named keys — the inner model takes them all and
        # newer versions require mm_token_type_ids for M-RoPE.
        kwargs = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = self.model.model(**kwargs, output_hidden_states=True)
        return out.hidden_states

    def _at_action_pos(self, hidden_states: tuple, attention_mask: torch.Tensor,
                       layer_idx: int) -> torch.Tensor:
        h = hidden_states[layer_idx]
        pos = self._action_pos(attention_mask)
        return h[torch.arange(h.size(0), device=h.device), pos].float()

    # ---- forward modes --------------------------------------------------

    @contextmanager
    def _truncated_layers(self, n_layers: int):
        inner = self.model.model
        lm = getattr(inner, "language_model", None) or inner.text_model
        full = lm.layers
        lm.layers = full[:n_layers]
        try:
            yield
        finally:
            lm.layers = full

    @torch.no_grad()
    def features_collect(self, obs: np.ndarray, exit_layers=EXIT_LAYERS) -> torch.Tensor:
        """One full pass -> (B, n_exits, hidden) hidden states at action pos."""
        batch = self.encode(obs)
        hidden_states = self._inner_forward(batch)
        cols = [
            self._at_action_pos(hidden_states, batch["attention_mask"], li)
            for li in exit_layers
        ]
        return torch.stack(cols, dim=1)

    @torch.no_grad()
    def features_to(self, obs: np.ndarray, exit_idx: int, exit_layers=EXIT_LAYERS) -> torch.Tensor:
        """Truncated pass -> (B, hidden) at exit `exit_idx`, honestly stopping."""
        layer_idx = exit_layers[exit_idx]
        n = _n_run_layers(layer_idx, self.n_layers)
        batch = self.encode(obs)
        if n < self.n_layers:
            with self._truncated_layers(n):
                hidden_states = self._inner_forward(batch)
        else:
            hidden_states = self._inner_forward(batch)
        return self._at_action_pos(hidden_states, batch["attention_mask"], layer_idx)

    @torch.no_grad()
    def describe(self, frame: np.ndarray, prompt: str | None = None,
                 max_new_tokens: int = 40) -> str:
        """Post-hoc scene description for the demo video.

        Generation is NOT part of the policy loop — this narrates a frame
        after the fact, for the overlay only.
        """
        text = (prompt
                or "<|vision_start|><|image_pad|><|vision_end|>\n"
                   "You are playing Doom. In one short sentence: what do you "
                   "see and what will you do?")
        batch = self.processor(text=[text], images=[frame],
                               return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) if hasattr(v, "to") else v
                 for k, v in batch.items()}
        out = self.model.generate(**batch, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        gen = out[0][batch["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()


class StubEncoder(nn.Module):
    """Tiny image-dependent encoder standing in for Qwen on CPU.

    Mean-pools each frame into (Hp*Wp*3) patch means and projects to the
    shared hidden size. Real dependence on pixels keeps the whole pipeline
    (router, heads, PPO) meaningful in CPU tests and smoke runs.
    """

    def __init__(self, hidden: int = 2048, hp: int = 6, wp: int = 10, n_exits: int = 3):
        super().__init__()
        self.hp, self.wp, self.hidden = hp, wp, hidden
        self.proj = nn.Linear(hp * wp * 3, hidden)
        self.mix = nn.Parameter(torch.linspace(0.7, 1.3, n_exits).view(1, n_exits, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    def _feat(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(obs, dtype=torch.float32) / 255.0
        b, k, h, w, c = x.shape
        x = x.reshape(b, k, self.hp, h // self.hp, self.wp, w // self.wp, c)
        x = x.mean(dim=(3, 5)).reshape(b, k, -1).mean(dim=1)
        return self.proj(x)

    @torch.no_grad()
    def features_collect(self, obs: np.ndarray, exit_layers=None) -> torch.Tensor:
        h = self._feat(obs)
        return h.unsqueeze(1) * self.mix.to(h.dtype)

    @torch.no_grad()
    def features_to(self, obs: np.ndarray, exit_idx: int, exit_layers=None) -> torch.Tensor:
        return self.features_collect(obs)[:, exit_idx]
