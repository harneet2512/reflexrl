"""The sample-efficiency claim, as footage instead of a table.

The headline number (2.95x fewer environment steps) is invisible in a normal
demo, because a finished policy looks competent whatever trained it. This
freezes the *budget* instead: three agents that have each seen exactly the same
number of Doom frames, playing the same map from the same seed, side by side.

    PPO from scratch          no teacher
    shuffled-teacher control  guidance of the right shape, decoupled from the frame
    ReflexRL                  the real Qwen3-VL + Jev teacher

Run under WSL/Linux, where ViZDoom is installed. Checkpoints come from the
local backup mirror, so this costs no Kaggle quota.

    python scripts/make_budget_video.py --budget 250368
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.demo import compose as C  # noqa: E402
from reflexrl.env.vizdoom_env import DoomEnv  # noqa: E402
from reflexrl.eval.realtime import realtime_episode  # noqa: E402
from reflexrl.policy.actor_critic import ActorCritic  # noqa: E402

SEED = 909090  # a map seed no training or reported evaluation ever used

# label, colour, path under the backup root, in the order they appear on screen
ARMS = [
    ("PPO FROM SCRATCH", (150, 150, 160),
     "reflexrl-train-ppo/runs/train/defend_the_center/ppo_s0"),
    ("SHUFFLED TEACHER", C.QWEN,
     "reflexrl-ablation/runs/train/defend_the_center/reflexrl_shuffled_s0"),
    ("REFLEXRL", C.ACCENT,
     "reflexrl-train-lane0/runs/train/defend_the_center/reflexrl_s0"),
]
SUBTITLE = {"PPO FROM SCRATCH": "no teacher",
            "SHUFFLED TEACHER": "guided, but the advice ignores the frame",
            "REFLEXRL": "guided by what Qwen3-VL saw"}


def decider(ckpt: Path, n_act: int):
    policy = ActorCritic(n_act)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.eval()

    @torch.no_grad()
    def decide(env, obs):
        return int(policy.act(torch.as_tensor(obs[None]))[0])
    return decide


def roll(ckpt: Path, scenario: str, episodes: int) -> list[tuple[list, float]]:
    """Recorded episodes from fixed seeds, identical for every arm.

    One episode is worth nothing: a single Doom episode swings by several kills
    on luck alone, and the first cut of this video happened to show ReflexRL on
    a 7-kill run against PPO on a 2-kill run, which overstates a gap the
    evaluation curves put closer to 5 against 3.3. Several episodes are shown
    back to back with a running mean, so the number on screen converges on the
    measured one instead of advertising a lucky draw.
    """
    out = []
    decide = decider(ckpt, 5)
    for i in range(episodes):
        # the policy SAMPLES its actions, so without this the same map seed
        # gives different play every run and the on-screen numbers are not
        # reproducible (first cut moved PPO's mean from 3.67 to 2.17 on a rerun)
        torch.manual_seed(SEED + i)
        env = DoomEnv(scenario, seed=SEED + i)
        try:
            r = realtime_episode(env, decide, record=True)
        finally:
            env.close()
        out.append((r["tics"], r["return"]))
    return out


def episode_row(arms: list[tuple[str, tuple, list, float]], budget: int, speed: int,
                ep: int, n_eps: int, running: dict[str, list[float]]) -> list[np.ndarray]:
    """One episode, three panels, synchronised by game tic and played sped up.

    Full episodes are recorded and sampled every `speed` tics, so a 60 s Doom
    episode becomes a ~20 s clip that still shows every kill. Arms die at
    different times, so the canvas runs until the longest ends and a finished
    panel is marked rather than frozen without explanation.
    """
    longest = max(len(t) for _, _, t, _ in arms)
    pw, ph = 404, 227
    xs = [16, 438, 860]
    frames, last = [], [None] * len(arms)
    for t in range(0, longest, speed):
        img = C.canvas()
        C.text(img, "SAME TRAINING BUDGET", (C.W // 2, 44), 1.0, C.FG, 2, center=True)
        C.text(img, f"every agent below has seen {budget:,} frames of Doom",
               (C.W // 2, 74), 0.6, C.DIM, 1, center=True)
        C.text(img, f"{speed}x speed", (C.W - 128, 40), 0.55, C.DIM)
        C.text(img, f"episode {ep + 1} of {n_eps}", (C.W - 128, 68), 0.55, C.DIM)
        C.text(img, f"{t / C.FPS:4.1f} s of game time", (16, 40), 0.55, C.DIM)
        for i, (name, col, tics, _f) in enumerate(arms):
            over = t >= len(tics)
            tic = tics[min(t, len(tics) - 1)]
            last[i] = tic.frame if tic.frame is not None else last[i]
            x0 = xs[i]
            C.text(img, name, (x0, 118), 0.6, col, 1)
            C.text(img, SUBTITLE[name], (x0, 142), 0.42, C.DIM, 1)
            img[156:156 + ph, x0:x0 + pw] = C._panel(last[i], (pw, ph))
            if over:
                cv2.rectangle(img, (x0, 156), (x0 + pw, 156 + ph), (0, 0, 0), -1)
                C.text(img, "ELIMINATED", (x0 + pw // 2, 156 + ph // 2 - 8), 0.75, col,
                       2, center=True)
                C.text(img, f"survived {len(tics) / C.FPS:.0f} s",
                       (x0 + pw // 2, 156 + ph // 2 + 22), 0.45, C.DIM, 1, center=True)
            C.text(img, "KILLS THIS EPISODE", (x0, 428), 0.45, C.DIM)
            C.text(img, f"{tic.score:.0f}", (x0, 478), 1.5, col, 3)
            # running mean over episodes already finished, so the number on
            # screen walks toward the measured mean rather than one lucky draw
            seen = running[name] + ([tic.score] if over else [])
            if seen:
                C.text(img, f"mean of {len(seen)}: {float(np.mean(seen)):.1f}",
                       (x0, 512), 0.5, C.FG)
            # every episode so far as a small bar, so consistency is visible and
            # not just asserted: a steady arm makes a flat row of equal bars
            C.text(img, "EVERY EPISODE", (x0, 556), 0.42, C.DIM)
            # bars are sized to the panel so any episode count fits inside it
            gap, base, top = 6, 640, 68
            bw = max(8, (pw - gap * (n_eps - 1)) // n_eps)
            for k in range(n_eps):
                bx = x0 + k * (bw + gap)
                cv2.rectangle(img, (bx, base - top), (bx + bw, base), (38, 38, 44), -1)
                val = (running[name][k] if k < len(running[name])
                       else (tic.score if k == ep else None))
                if val is None:
                    continue
                h = int(min(1.0, val / 10.0) * top)
                cv2.rectangle(img, (bx, base - h), (bx + bw, base),
                              col if k <= ep else (60, 60, 68), -1)
                C.text(img, f"{val:.0f}", (bx + bw // 2, base + 22), 0.45,
                       C.FG if k == ep else C.DIM, 1, center=True)
        frames.append(img)
    return frames + [frames[-1]] * int(1.0 * C.FPS)  # hold the final scoreline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", default="/mnt/d/reflexrl_backup")
    ap.add_argument("--budgets", type=int, nargs="*", default=[250368])
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--speed", type=int, default=3,
                    help="play back every Nth game tic; 3 turns a 60 s episode into 20 s")
    ap.add_argument("--out", default="results/demo")
    args = ap.parse_args()
    root, out = Path(args.backup), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    summary: dict = {"seed_base": SEED, "scenario": args.scenario,
                     "episodes_per_arm": args.episodes, "budgets": {}}
    for budget in args.budgets:
        rolls = {}
        for name, col, rel in ARMS:
            ckpt = root / rel / f"ckpt_{budget}.pt"
            if not ckpt.exists():
                raise SystemExit(f"missing checkpoint: {ckpt}")
            rolls[name] = (col, roll(ckpt, args.scenario, args.episodes))
            scores = [r for _, r in rolls[name][1]]
            print(f"{budget:>9,}  {name:<18} kills {scores} mean "
                  f"{float(np.mean(scores)):.2f}", flush=True)
        summary["budgets"][str(budget)] = {
            n: {"per_episode": [r for _, r in v[1]],
                "mean": float(np.mean([r for _, r in v[1]]))} for n, v in rolls.items()}
        running: dict[str, list[float]] = {n: [] for n in rolls}
        for ep in range(args.episodes):
            arms = [(n, rolls[n][0], rolls[n][1][ep][0], rolls[n][1][ep][1])
                    for n, _, _ in ARMS]
            frames += episode_row(arms, budget, args.speed, ep, args.episodes, running)
            for n, _, _ in ARMS:
                running[n].append(rolls[n][1][ep][1])

    (out / "budget_video.json").write_text(json.dumps(summary, indent=2))
    raw = out / "budget_raw.mp4"
    C.write_video(frames, raw)
    print(f"wrote {raw} ({len(frames)} frames, {len(frames) / C.FPS:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
