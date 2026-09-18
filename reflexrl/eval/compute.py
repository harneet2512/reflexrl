"""Compute accounting: analytic FLOP table per exit + cost model for reward.

FLOPs are counted analytically from the HF config (parameter shapes x
sequence lengths), not measured, so the table is hardware-independent.
lm_head is never computed in our pipeline (we call the inner model), so it
is not counted. Per-exit norms/heads are included but negligible.

Cost model for reward shaping: C(d) normalized by the DEEP cost, so
``reward = env_reward - lambda * C(d)/C(DEEP)`` charges each decision a
fraction of full-model compute.
"""

from __future__ import annotations

from dataclasses import dataclass

PATCH = 16
MERGE = 2


@dataclass
class ComputeModel:
    """FLOPs for one decision (single forward at the action position)."""

    n_frames: int
    frame_h: int
    frame_w: int
    text_tokens: int
    # vision_config
    vit_hidden: int = 1024
    vit_depth: int = 24
    vit_intermediate: int = 4096
    vit_out: int = 2048
    vit_heads: int = 16
    # text_config
    llm_hidden: int = 2048
    llm_layers: int = 28
    llm_intermediate: int = 6144
    llm_kv: int = 1024  # num_key_value_heads * head_dim
    exit_layers: tuple = (9, 19, -1)

    # ---- derived sizes ---------------------------------------------------

    @property
    def grid_h(self) -> int:
        # Qwen processor resizes each spatial dim to a multiple of 32.
        return -(-self.frame_h // 32) * 32 // PATCH

    @property
    def grid_w(self) -> int:
        return -(-self.frame_w // 32) * 32 // PATCH

    @property
    def vit_tokens(self) -> int:
        return self.n_frames * self.grid_h * self.grid_w

    @property
    def llm_tokens(self) -> int:
        return self.n_frames * (self.grid_h * self.grid_w) // (MERGE * MERGE) + self.text_tokens

    # ---- FLOP pieces ------------------------------------------------------

    def _vit_flops(self) -> float:
        s = self.vit_tokens
        h, inter = self.vit_hidden, self.vit_intermediate
        params = 4 * h * h + 2 * h * inter  # qkv+proj+mlp per block
        blocks = self.vit_depth * 2 * params * s
        patch_embed = 2 * s * (3 * 2 * PATCH * PATCH) * h
        merge_in = h * MERGE * MERGE
        merger = 2 * s * (merge_in * merge_in + merge_in * self.vit_out)
        deepstack = 3 * merger  # three extra mergers, same shape
        per_image = s // self.n_frames
        attn = self.n_frames * 4 * per_image * per_image * h  # QK^T + AV
        return float(blocks + patch_embed + 4 * merger + deepstack + attn)

    def _llm_layer_flops(self) -> float:
        h, inter, kv = self.llm_hidden, self.llm_intermediate, self.llm_kv
        return 2 * (2 * h * h + 2 * h * kv + 3 * h * inter)

    def _llm_flops(self, n_layers: int) -> float:
        s = self.llm_tokens
        linear = n_layers * self._llm_layer_flops() * s
        attn = n_layers * 2 * s * s * self.llm_hidden  # causal ~ 2 matmuls
        return float(linear + attn)

    def _head_flops(self) -> float:
        return float(2 * self.llm_hidden * 256)

    def flops(self, exit_idx: int) -> float:
        """Total FLOPs for one decision resolved at exit `exit_idx`."""
        layer = self.exit_layers[exit_idx]
        # hidden_states[k] requires k+1 executed layers (transformers 5.x
        # appends the post-norm state at index n of a truncated run).
        n = self.llm_layers if layer == -1 else layer + 1
        return self._vit_flops() + self._llm_flops(n) + self._head_flops()

    def table(self) -> list[dict]:
        deep = self.flops(len(self.exit_layers) - 1)
        out = []
        for i, layer in enumerate(self.exit_layers):
            f = self.flops(i)
            out.append(
                {"exit": i, "layer": layer, "gflops": f / 1e9, "frac_of_deep": f / deep}
            )
        return out

    def cost_vector(self) -> list[float]:
        """C(d)/C(DEEP) per exit — the reward penalty coefficients."""
        return [row["frac_of_deep"] for row in self.table()]


def default_model(n_frames: int = 2, text_tokens: int = 12) -> ComputeModel:
    """Qwen3-VL-2B numbers at the $20-config input size (320x180 frames)."""
    return ComputeModel(
        n_frames=n_frames, frame_h=180, frame_w=320, text_tokens=text_tokens
    )
