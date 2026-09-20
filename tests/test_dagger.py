"""A DAgger round must grow the label set from the student's own states and refit."""

import json

import numpy as np
import pytest
import torch

from reflexrl.baselines.perception_bc import PerceptionNet
from reflexrl.rl.dagger import DaggerTeacher, jev_state
from reflexrl.teacher.jev import JevClient
from reflexrl.teacher.perception_jev import CLASSES


def _labels(n=64):
    rng = np.random.default_rng(0)
    return {"obs": rng.integers(0, 255, (n, 12, 64, 112), dtype=np.uint8),
            "probs": rng.dirichlet(np.ones(5), size=n).astype(np.float16),
            "action": np.zeros(n, np.int16), "episode": np.arange(n, dtype=np.int32) // 8,
            "step": np.arange(n, dtype=np.int32)}


def test_jev_state_is_typed_and_cacheable():
    from reflexrl.env.scenarios import get_scenario
    q = np.array([0.05, 0.9, 0.03, 0.02])
    state, key = jev_state(get_scenario("dtc"), q, "FIRE")
    assert "in the center" in state and "Previous action: FIRE" in state
    assert key == "center|high|FIRE"


def test_jev_client_falls_back_to_the_cached_table_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    c = JevClient("experiments/configs/jev_table_dtc.json")
    assert not c.live
    M = c.table_probs(CLASSES)
    assert M.shape == (4, 5)
    np.testing.assert_allclose(M.sum(1), 1, atol=1e-6)


@pytest.mark.env
def test_dagger_round_grows_labels_and_refits(monkeypatch):
    data = json.load(open("experiments/configs/jev_table_dtc.json"))
    J = np.array([data["table"][c] for c in CLASSES], np.float64)
    J /= J.sum(1, keepdims=True)
    jev = JevClient("experiments/configs/jev_table_dtc.json")
    t = DaggerTeacher(PerceptionNet(), J, jev, "experiments/configs/jev_table_dtc.json",
                      "dtc", _labels(), device="cpu", every=0, steps_per_round=4, max_rounds=1)

    class StubQwen:  # stands in for Qwen3-VL: always "center"
        def perceive(self, frames):
            return np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (len(frames), 1))
    t.teacher = StubQwen()
    from reflexrl.policy.actor_critic import ActorCritic
    info = t.maybe_round(0, ActorCritic(5))
    assert info["added"] == 4 and info["labels"] == 68
    assert t.probs(np.zeros((2, 12, 64, 112), np.uint8)).shape == (2, 5)
