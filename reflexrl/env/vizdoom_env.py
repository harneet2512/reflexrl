"""ViZDoom environment. Pixels in, buttons out, nothing else.

Observation is a stack of downsampled RGB frames. No depth buffer, no labels,
no automap, no game variables. ``teacher_frames()`` exposes the last
full-resolution frames so the teacher sees the same screen the student does;
``debug_state()`` exposes privileged state for eval logging only and is never
part of the observation (tests/test_env.py enforces both).
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

# Spawned env workers must not each allocate a full BLAS thread pool: with 8+
# workers on a 12-core box that exhausts memory before the first step.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from reflexrl.env.observations import (
    FULL_H,
    FULL_W,
    STUDENT_H,
    STUDENT_STACK,
    STUDENT_W,
    stack_frames,
    to_student_frame,
)
from reflexrl.env.scenarios import get_scenario

TEACHER_FRAMES = 2


class DoomEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario: str = "dtc", frame_skip: int = 4, seed: int | None = None,
                 keep_full_frames: bool = False, eval_labels: bool = False):
        super().__init__()
        self.scenario = get_scenario(scenario)
        self.game = vzd.DoomGame()
        self.game.load_config(str(Path(vzd.scenarios_path) / self.scenario.cfg))
        self.game.set_window_visible(False)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X180)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(False)
        # Labels are privileged: only the offline teacher probe may enable them,
        # to score Qwen against ground truth. No policy ever reads them.
        self.game.set_labels_buffer_enabled(eval_labels)
        self.game.set_automap_buffer_enabled(False)
        self.game.set_mode(vzd.Mode.PLAYER)
        if self.scenario.episode_timeout is not None:
            self.game.set_episode_timeout(self.scenario.episode_timeout)
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.init()

        self.frame_skip = frame_skip
        self._buttons = self.scenario.button_vectors()
        self._stack: deque[np.ndarray] = deque(maxlen=STUDENT_STACK)
        self._full: deque[np.ndarray] | None = (
            deque(maxlen=TEACHER_FRAMES) if keep_full_frames else None)
        self._ep_return = 0.0
        self._ep_len = 0

        self.action_space = spaces.Discrete(len(self._buttons))
        self.observation_space = spaces.Box(
            0, 255, shape=(3 * STUDENT_STACK, STUDENT_H, STUDENT_W), dtype=np.uint8)

    # ---- frames -----------------------------------------------------------

    def _screen(self) -> np.ndarray:
        buf = np.asarray(self.game.get_state().screen_buffer, dtype=np.uint8)
        if buf.shape != (FULL_H, FULL_W, 3):
            buf = np.moveaxis(buf, 0, -1)
        return buf

    def _push(self, screen: np.ndarray) -> None:
        self._stack.append(to_student_frame(screen))
        if self._full is not None:
            self._full.append(screen.copy())

    def _obs(self) -> np.ndarray:
        return stack_frames(list(self._stack))

    def teacher_frames(self) -> list[np.ndarray]:
        """Last TEACHER_FRAMES full-res RGB frames, oldest first."""
        if self._full is None:
            raise RuntimeError("construct DoomEnv(keep_full_frames=True) to use the teacher")
        return list(self._full)

    # ---- gym API ----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        self._ep_return, self._ep_len = 0.0, 0
        screen = self._screen()
        self._stack.clear()
        if self._full is not None:
            self._full.clear()
        self._last_screen = screen
        for _ in range(STUDENT_STACK):
            self._push(screen)
        return self._obs(), {}

    def step(self, action: int):
        reward = float(self.game.make_action(self._buttons[int(action)], self.frame_skip))
        self._ep_return += reward
        self._ep_len += 1
        finished = self.game.is_episode_finished()
        terminated = self.game.is_player_dead()
        truncated = finished and not terminated
        info = {}
        if finished:
            # No screen after the terminal tic: repeat the last frame.
            self._push(self._last_screen)
            info["episode"] = {"r": self._ep_return, "l": self._ep_len}
        else:
            self._last_screen = self._screen()
            self._push(self._last_screen)
        return self._obs(), reward, terminated, truncated, info

    def debug_state(self) -> dict:
        """Privileged state for eval logging only. Never an observation."""
        out = {"episode_time": self.game.get_episode_time()}
        for name, var in (("health", vzd.GameVariable.HEALTH),
                          ("kills", vzd.GameVariable.KILLCOUNT)):
            try:
                out[name] = float(self.game.get_game_variable(var))
            except Exception:
                out[name] = None
        return out

    def close(self):
        self.game.close()


def make_env_fn(scenario: str, seed: int, frame_skip: int = 4):
    def _thunk():
        return DoomEnv(scenario, frame_skip=frame_skip, seed=seed)
    return _thunk


def make_vec_env(scenario: str, n_envs: int, seed: int, asynchronous: bool = True):
    """Vector env with gymnasium's same-step autoreset."""
    fns = [make_env_fn(scenario, seed * 1000 + i) for i in range(n_envs)]
    kwargs = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    if asynchronous:
        return gym.vector.AsyncVectorEnv(fns, context="spawn", **kwargs)
    return gym.vector.SyncVectorEnv(fns, **kwargs)
