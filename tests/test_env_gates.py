"""Environment gates.

The load-bearing property: the policy observation is pixels and nothing
else. Privileged state exists only behind debug_state(); the step/reset
return surface must never carry it.
"""

from __future__ import annotations

import numpy as np
import pytest

from reflexrl.envs.vizdoom_env import H, VecEnvs, VizdoomPixelsEnv, W

PRIVILEGED_KEYS = {"health", "ammo", "kills", "angle", "enemies_visible", "episode_time"}


@pytest.fixture
def env():
    e = VizdoomPixelsEnv(seed=0)
    yield e
    e.close()


def test_observation_is_pixels_only(env):
    obs, info = env.reset()
    assert obs.shape == (2, H, W, 3)
    assert obs.dtype == np.uint8
    assert isinstance(info, dict)
    assert not (PRIVILEGED_KEYS & set(info))


def test_step_surface_has_no_privileged_state(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.dtype == np.uint8
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    leaked = PRIVILEGED_KEYS & set(info)
    assert not leaked, f"privileged keys leaked into info: {leaked}"


def test_frame_stack_shifts(env):
    obs0, _ = env.reset()
    obs1, *_ = env.step(1)
    assert not np.array_equal(obs0, obs1), "stack did not update"


def test_episode_terminates_and_reports_return(env):
    env.reset()
    for _ in range(600):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            assert "episode_return" in info and "episode_len" in info
            return
    pytest.fail("episode never ended within 600 decisions")


def test_debug_state_is_the_only_privileged_channel(env):
    env.reset()
    dbg = env.debug_state()
    assert PRIVILEGED_KEYS <= set(dbg), "debug channel missing expected keys"


def test_vecenv_autoresets():
    vec = VecEnvs(3, seed=1)
    obs = vec.reset()
    assert obs.shape == (3, 2, H, W, 3)
    done_seen = False
    for _ in range(600):
        obs, rewards, terms, truncs, infos = vec.step(
            np.array([vec.single_action_space.sample() for _ in range(3)])
        )
        assert obs.shape == (3, 2, H, W, 3)
        if any(terms) or any(truncs):
            done_seen = True
            done_infos = [
                i for i, d in zip(infos, terms | truncs, strict=True) if d
            ]
            assert all("final_obs" in i for i in done_infos)
            break
    vec.close()
    assert done_seen, "no episode ended in 600 vec steps"
