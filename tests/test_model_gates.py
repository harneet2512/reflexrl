"""Model gates: heads, agent glue, GAE — everything that runs without
downloading Qwen weights. The Qwen path itself is tested in Modal's
run_tests where the checkpoint is available.
"""

from __future__ import annotations

import numpy as np
import torch

from reflexrl.eval.compute import default_model
from reflexrl.models.heads import ExitPolicy
from reflexrl.models.qwen_vl import StubEncoder
from reflexrl.rl.agent import ExitAgent
from reflexrl.rl.rollout import compute_gae


def _agent(seed: int = 0):
    torch.manual_seed(seed)
    return ExitAgent(StubEncoder(), ExitPolicy(n_actions=6))


def _obs(b: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(b, 2, 180, 320, 3), dtype=np.uint8)


def test_stub_encoder_is_obs_dependent():
    enc = StubEncoder()
    a, b = _obs(1, 0), _obs(1, 1)
    fa = enc.features_collect(a)
    fb = enc.features_collect(b)
    assert fa.shape == (1, 3, 2048)
    assert not torch.allclose(fa, fb), "features must depend on pixels"


def test_act_shapes_and_joint_logp():
    agent = _agent()
    out = agent.act(_obs(4))
    assert out["action"].shape == (4,) and out["depth"].shape == (4,)
    assert out["h_stack"].shape == (4, 3, 2048)
    assert out["logp"].shape == (4,) and out["value"].shape == (4,)
    assert (out["depth"] >= 0).all() and (out["depth"] < 3).all()


def test_forced_depth_respected():
    agent = _agent()
    for d in range(3):
        out = agent.act(_obs(4), forced_d=d)
        assert (out["depth"] == d).all()


def test_evaluate_recovers_logp():
    agent = _agent()
    obs = _obs(8)
    out = agent.act(obs)
    mb = {
        "h_stack": out["h_stack"],
        "action": out["action"],
        "depth": out["depth"],
    }
    ev = agent.evaluate(mb)
    assert torch.allclose(ev["logp"], out["logp"], atol=1e-5), (
        f"evaluate logp {ev['logp']} != act logp {out['logp']}"
    )


def test_gae_terminal_and_truncation():
    # 3 steps, 1 env. rewards 1,1,1; values all 0.5; step1 truncated w/ final_v=2.
    store = {
        "reward": torch.tensor([[1.0], [1.0], [1.0]]),
        "value": torch.full((3, 1), 0.5),
        "terminated": torch.tensor([[False], [False], [True]]),
        "truncated": torch.tensor([[True], [False], [False]]),
        "final_v": torch.tensor([[2.0], [0.0], [0.0]]),
    }
    next_value = torch.zeros(1)
    adv, ret = compute_gae(store, next_value, gamma=1.0, gae_lambda=1.0)

    # t=2 terminal: delta = 1 - .5 = .5
    # t=1 normal:   delta = 1 + .5 - .5 = 1; adv = 1 + .5 = 1.5
    # t=0 truncated:delta = 1 + 2.0 - .5 = 2.5; chain cut -> adv = 2.5
    assert torch.allclose(adv[:, 0], torch.tensor([2.5, 1.5, 0.5]), atol=1e-6)
    assert torch.allclose(ret, adv + store["value"])


def test_flop_table_monotonic():
    cm = default_model(n_frames=2)
    rows = cm.table()
    fs = [r["gflops"] for r in rows]
    assert fs[0] < fs[1] < fs[2], f"exit flops not increasing: {fs}"
    assert rows[0]["frac_of_deep"] < 1.0 and rows[2]["frac_of_deep"] == 1.0
    cv = cm.cost_vector()
    assert len(cv) == 3 and cv[2] == 1.0
