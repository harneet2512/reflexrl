"""ViZDoom environments. Pixels in, buttons out, nothing else.

The policy observation is a stack of RGB frames and nothing more. Game
variables (health, ammo, kills, visible-actor labels) are exposed only via
``debug_state()`` for evaluation logging; they are never part of the
observation. tests/test_env_gates.py asserts the policy path cannot see
them.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

# defend_the_center.cfg ships 3 buttons: TURN_LEFT, TURN_RIGHT, ATTACK.
# Combos give the compact discrete vocabulary the policy acts over.
DTC_ACTIONS = [
    [0, 0, 0],  # idle
    [1, 0, 0],  # turn left
    [0, 1, 0],  # turn right
    [0, 0, 1],  # shoot
    [1, 0, 1],  # turn left + shoot
    [0, 1, 1],  # turn right + shoot
]

RESOLUTION = vzd.ScreenResolution.RES_320X180
H, W = 180, 320


class VizdoomPixelsEnv(gym.Env):
    """Pixels-only ViZDoom env.

    Observation: ``(frame_stack, H, W, 3)`` uint8. Action: ``Discrete``
    over button combos. ``debug_state()`` returns privileged game state for
    eval-time logging only.
    """

    def __init__(
        self,
        config: str = "defend_the_center.cfg",
        frame_skip: int = 4,
        frame_stack: int = 2,
        actions: list[list[int]] | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
        render_labels: bool = False,
        scenario_wad: str | None = None,
    ):
        super().__init__()
        self.game = vzd.DoomGame()
        cfg = Path(config)
        if not cfg.is_file():
            cfg = Path(vzd.scenarios_path) / config
        self.game.load_config(str(cfg))
        if scenario_wad is not None:
            self.game.set_doom_scenario_path(scenario_wad)
        self.game.set_window_visible(False)
        self.game.set_screen_resolution(RESOLUTION)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        if render_labels:
            self.game.set_labels_buffer_enabled(True)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.init()

        self.frame_skip = frame_skip
        self.actions = actions or DTC_ACTIONS
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)
        self._steps = 0
        self._max_steps = max_steps
        self._episode_return = 0.0

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            0, 255, shape=(frame_stack, H, W, 3), dtype=np.uint8
        )

    def _frame(self) -> np.ndarray:
        state = self.game.get_state()
        buf = state.screen_buffer
        # vizdoom >= 1.3 returns (H, W, C); older builds return (C, H, W).
        if buf.ndim == 3 and buf.shape[0] == 3 and buf.shape != (H, W, 3):
            buf = np.moveaxis(buf, 0, -1)
        return np.asarray(buf, dtype=np.uint8).copy()

    def _obs(self) -> np.ndarray:
        return np.stack(list(self._frames), axis=0)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        if not self.game.is_running() or self.game.is_episode_finished():
            self.game.new_episode()
        self._steps = 0
        self._episode_return = 0.0
        frame = self._frame()
        self._frames.clear()
        for _ in range(self._frames.maxlen):
            self._frames.append(frame)
        return self._obs(), {}

    def step(self, action: int):
        reward = float(self.game.make_action(self.actions[int(action)], self.frame_skip))
        self._episode_return += reward
        self._steps += 1

        terminated = self.game.is_player_dead()
        truncated = (
            self.game.is_episode_finished() and not terminated
        ) or (self._max_steps is not None and self._steps >= self._max_steps)
        done = terminated or truncated

        if done:
            # Final frame may be unavailable; repeat the last one.
            frame = self._frames[-1]
            info = {
                "episode_return": self._episode_return,
                "episode_len": self._steps,
            }
        else:
            frame = self._frame()
            info = {}
        self._frames.append(frame)
        return self._obs(), reward, terminated, truncated, info

    def debug_state(self) -> dict:
        """Privileged state for eval logging. Never call from the policy path."""
        out = {
            "episode_time": self.game.get_episode_time(),
            "player_dead": self.game.is_player_dead(),
        }
        for name, var in [
            ("health", vzd.GameVariable.HEALTH),
            ("ammo", vzd.GameVariable.SELECTED_WEAPON_AMMO),
            ("kills", vzd.GameVariable.KILLCOUNT),
            ("angle", vzd.GameVariable.ANGLE),
        ]:
            try:
                out[name] = float(self.game.get_game_variable(var))
            except Exception:
                out[name] = None
        out["enemies_visible"] = self._enemies_visible()
        return out

    def _enemies_visible(self) -> int | None:
        state = self.game.get_state()
        if state is None or state.labels_buffer is None:
            return None
        # Labels encode the actor's rendered position; count distinct ids
        # that are not the player weapon (object_name heuristics are flaky,
        # so report raw non-zero label count minus weapon sprite).
        labels = np.asarray(state.labels)
        return int(sum(1 for lab in labels if lab.object_name not in ("DoomPlayer", "Weapon")))

    def close(self):
        if self.game.is_running():
            self.game.close()


class VecEnvs:
    """Synchronous vector wrapper: N envs stepped together on one process.

    ViZDoom sim is cheap; the GPU forward is the bottleneck, so one process
    with N games beats subprocess plumbing at this scale.
    """

    def __init__(self, n: int, **env_kwargs):
        self.envs = [VizdoomPixelsEnv(**env_kwargs) for _ in range(n)]
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space
        self.num_envs = n

    def reset(self, seed: int | None = None):
        obs = [e.reset(seed=None if seed is None else seed + i)[0] for i, e in enumerate(self.envs)]
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        obs, rewards, terms, truncs, infos = [], [], [], [], []
        for env, a in zip(self.envs, actions, strict=True):
            o, r, te, tr, info = env.step(int(a))
            if te or tr:
                info["final_obs"] = o
                o, _ = env.reset()
            obs.append(o)
            rewards.append(r)
            terms.append(te)
            truncs.append(tr)
            infos.append(info)
        return (
            np.stack(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms),
            np.asarray(truncs),
            infos,
        )

    def debug_states(self) -> list[dict]:
        return [e.debug_state() for e in self.envs]

    def close(self):
        for e in self.envs:
            e.close()
