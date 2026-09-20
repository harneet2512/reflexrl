"""Teacher = Qwen3-VL sees, Jev decides.

Qwen answers the probe-validated perception question ("where is the nearest
monster: left / center / right / none"), averaged over option permutations,
giving q(c|o). Jev 1.13's action distribution for each perception class,
J(a|c), is read from a cached decision table. The teacher policy is the
expectation of Jev's decision under Qwen's perception:

    pi_T(a|o) = sum_c q(c|o) * J(a|c)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from reflexrl.env.scenarios import Scenario
from reflexrl.teacher.prompts import LETTERS

CLASSES = ("left", "center", "right", "none")
LABELS = {"left": "on the left side", "center": "in the center, straight ahead",
          "right": "on the right side", "none": "no monster is visible"}


class PerceptionJevTeacher:
    def __init__(self, qwen, jev_table_path: str | Path, n_perm: int = 4, seed: int = 0):
        self.qwen = qwen
        data = json.loads(Path(jev_table_path).read_text())
        self.actions = data["actions"]
        self.J = np.array([data["table"][c] for c in CLASSES], np.float64)  # (4, A)
        self.J /= self.J.sum(1, keepdims=True)
        rng = np.random.default_rng(seed)
        self.perms = [np.arange(4)] + [rng.permutation(4) for _ in range(n_perm - 1)]
        self.perception_log: list[list[float]] = []

    def __getattr__(self, name):  # counters (calls, samples, seconds, ...) live on qwen
        return getattr(self.qwen, name)

    def perceive(self, frames: list[list[np.ndarray]]) -> np.ndarray:
        q = np.zeros((len(frames), 4), np.float64)
        for perm in self.perms:
            options = "\n".join(f"{LETTERS[k]}. {LABELS[CLASSES[i]]}" for k, i in enumerate(perm))
            question = ("This is a first-person view from the game Doom. Where is the nearest "
                        f"monster in the last image?\n{options}\nAnswer with a single letter.")
            q[:, perm] += self.qwen.choice_probs(frames, question, 4)
        return q / len(self.perms)

    def action_probs(self, frames: list[list[np.ndarray]], scenario: Scenario) -> np.ndarray:
        if scenario.action_names != self.actions:
            raise ValueError(f"Jev table is for {self.actions}, not {scenario.action_names}")
        t0 = time.perf_counter()
        q = self.perceive(frames)
        self.qwen.seconds += time.perf_counter() - t0
        self.qwen.calls += 1
        self.qwen.samples += len(frames)
        self.perception_log.extend(q.round(4).tolist())
        return (q @ self.J).astype(np.float32)


def calibrate(q: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Divide out the model's answer prior (measured on a blank frame).

    Vision-language models over-report that a target is present; contextual
    calibration (Zhao et al., 2021) removes that bias without any labels.
    """
    out = q / np.clip(prior, 1e-6, None)
    return out / out.sum(1, keepdims=True)
