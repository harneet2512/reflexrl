"""Gates for the claims the write-up depends on.

- the policy sees pixels only;
- the action vocabularies are legal for each scenario's buttons;
- teacher dependence schedules reach exactly zero;
- the intervention correction leaves student-chosen samples unweighted;
- deployment needs no teacher (the student acts from pixels alone).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reflexrl.env.observations import STUDENT_H, STUDENT_STACK, STUDENT_W
from reflexrl.env.scenarios import SCENARIOS, get_scenario
from reflexrl.policy.actor_critic import ActorCritic, n_params
from reflexrl.rl.ppo import _gae
from reflexrl.rl.teacher_guidance import LADDER, AdaptiveSchedule, FixedSchedule
from reflexrl.teacher.prompts import action_letters, build_prompt_text


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_action_vocab_is_legal(name):
    s = SCENARIOS[name]
    vecs = s.button_vectors()
    assert len(vecs) == len(s.actions) >= 3
    assert all(len(v) == len(s.buttons) for v in vecs)
    assert len({tuple(v) for v in vecs}) == len(vecs), "duplicate actions"
    assert all(any(v) for v in vecs)


def test_prompt_lists_every_action_once():
    s = get_scenario("dc")
    text = build_prompt_text(s)
    for letter, name in zip(action_letters(s), s.action_names, strict=True):
        assert f"{letter}. {name}" in text


@pytest.mark.env
@pytest.mark.parametrize("name", ["dtc", "hg", "dc", "hgs"])
def test_env_observation_is_pixels_only(name):
    from reflexrl.env.vizdoom_env import DoomEnv
    env = DoomEnv(name, seed=0)
    try:
        obs, info = env.reset()
        assert obs.shape == (3 * STUDENT_STACK, STUDENT_H, STUDENT_W)
        assert obs.dtype == np.uint8
        assert info == {}
        assert not env.game.is_depth_buffer_enabled()
        assert not env.game.is_labels_buffer_enabled()
        assert not env.game.is_automap_buffer_enabled()
        obs, _, _, _, info = env.step(0)
        assert set(info) <= {"episode"}, "no privileged info in step()"
    finally:
        env.close()


def test_fixed_schedule_reaches_zero():
    s = FixedSchedule(horizon=1000)
    assert s.p(0) == LADDER[0] == 1.0
    assert s.p(999) > 0.0
    assert s.p(1000) == 0.0 and s.p(10**9) == 0.0
    ps = [s.p(t) for t in range(0, 1001, 10)]
    assert all(a >= b for a, b in zip(ps, ps[1:], strict=False)), "monotone non-increasing"


def test_adaptive_schedule_hands_over_only_when_student_matches_teacher():
    s = AdaptiveSchedule(teacher_return=10.0, horizon=10**6, min_steps_per_rung=100)
    s.on_eval(200, 5.0)
    assert s.p(200) == 1.0, "student below teacher: keep full guidance"
    s.on_eval(300, 10.5)
    assert s.p(300) == 0.5
    s.on_eval(350, 11.0)
    assert s.p(350) == 0.5, "rungs are rate-limited"
    for t in range(500, 5000, 500):
        s.on_eval(t, 99.0)
    assert s.p(5000) == 0.0
    assert AdaptiveSchedule(teacher_return=1e9, horizon=100).p(100) == 0.0, "hard horizon"


def test_importance_weight_is_one_for_student_steps():
    probs_s = torch.tensor([[0.2, 0.3, 0.5]])
    a = torch.tensor([2])
    lo = torch.log(probs_s.gather(1, a[:, None]).squeeze(1))
    lb = lo.clone()  # behaviour == student
    assert torch.clamp(torch.exp(lo - lb), max=1.0).item() == pytest.approx(1.0)
    teacher = torch.tensor([[0.0, 0.0, 1.0]])
    lb_t = torch.log(teacher.gather(1, a[:, None]).squeeze(1))
    w = torch.clamp(torch.exp(lo - lb_t), max=1.0).item()
    assert w == pytest.approx(0.5), "teacher step weighted by pi_old/b, truncated at 1"


def test_gae_matches_hand_computation():
    rew = torch.tensor([[1.0], [0.0], [2.0]])
    val = torch.tensor([[0.5], [0.4], [0.3]])
    done = torch.tensor([[0.0], [1.0], [0.0]])
    adv = _gae(rew, val, done, torch.tensor([0.2]), gamma=0.9, lam=0.8)
    d2 = 2.0 + 0.9 * 0.2 - 0.3
    d1 = 0.0 - 0.4  # terminal: no bootstrap
    d0 = 1.0 + 0.9 * 0.4 - 0.5
    assert adv[2, 0].item() == pytest.approx(d2)
    assert adv[1, 0].item() == pytest.approx(d1)
    assert adv[0, 0].item() == pytest.approx(d0 + 0.9 * 0.8 * d1)


def test_student_acts_from_pixels_alone_and_is_small():
    policy = ActorCritic(n_actions=5).eval()
    obs = torch.zeros((2, 3 * STUDENT_STACK, STUDENT_H, STUDENT_W), dtype=torch.uint8)
    a = policy.act(obs)
    assert a.shape == (2,) and int(a.max()) < 5
    assert n_params(policy) < 2_000_000
