"""Latency-aware real-time evaluation (the demo's split-screen, measured).

Doom runs at 35 tics/s regardless of the controller. Each decision's measured
wall-clock latency L becomes round(L / 28.57 ms) lag tics during which the
game keeps running with the previously chosen action held; then the new
action is applied for the env's frame skip. A controller faster than one tic
incurs no lag. The game advances one tic at a time, so a recording is the
literal 35 fps game timeline, lag included.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from reflexrl.env.vizdoom_env import DoomEnv

TIC_MS = 1000.0 / 35.0


@dataclass
class Tic:
    frame: np.ndarray | None  # full-res RGB, only when recording
    action: int | None  # action in force this tic (None: nothing pressed yet)
    fresh: bool  # a new decision landed on this tic
    latency_ms: float  # latency of the decision in force
    score: float  # cumulative return so far


def realtime_episode(env: DoomEnv, decide: Callable[[DoomEnv, np.ndarray], int],
                     record: bool = False) -> dict:
    """decide(env, obs) -> action; its wall-clock time is the charged latency."""
    obs, _ = env.reset()
    game = env.game
    buttons = env.scenario.button_vectors()
    held, held_a, held_ms = [0] * len(buttons[0]), None, 0.0
    total, lat_ms, lag_tics, tics = 0.0, [], 0, []

    def tic(button_vec, a, fresh, ms):
        nonlocal total
        total += game.make_action(button_vec, 1)
        done = game.is_episode_finished()
        if record:
            frame = None if done else env._screen().copy()
            tics.append(Tic(frame, a, fresh, ms, total))
        return done

    done = False
    while not done:  # the scenario episode_timeout guarantees termination
        t0 = time.perf_counter()
        a = decide(env, obs)
        ms = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(ms)
        for _ in range(int(round(ms / TIC_MS))):  # the world does not wait
            lag_tics += 1
            if (done := tic(held, held_a, False, held_ms)):
                break
        if done:
            break
        held, held_a, held_ms = buttons[a], a, ms
        for k in range(env.frame_skip):
            if (done := tic(held, a, k == 0, ms)):
                break
        if not done:
            env._last_screen = env._screen()
            env._push(env._last_screen)
            obs = env._obs()
    lat = np.asarray(lat_ms)
    return {"return": float(total), "decisions": len(lat_ms), "lag_tics": lag_tics,
            "game_seconds": (lag_tics + len(lat_ms) * env.frame_skip) / 35.0,
            "ms_mean": float(lat.mean()), "ms_median": float(np.median(lat)),
            "ms_p95": float(np.percentile(lat, 95)),
            "max_actions_per_s": float(1000.0 / lat.mean()),
            "tics": tics if record else None}
