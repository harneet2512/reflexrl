"""Health-Gathering perception probe: can the 8B see medkits well enough to teach?

Same protocol as the DTC probe (permutation-averaged lettered choice, oracle
from the eval-only labels buffer), retargeted at medkits and distance.

    python scripts/hg_probe.py --model Qwen/Qwen3-VL-8B-Instruct --nf4 --frames 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.debias_probe import CLASSES, collect, perception_probs  # noqa: E402
from reflexrl.env.vizdoom_env import DoomEnv  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402
from reflexrl.teacher.perception_jev import calibrate  # noqa: E402
from reflexrl.teacher.prompts import LETTERS  # noqa: E402

NEAR_BBOX_H = 24  # px at 180-high render: a medkit within a step or two


def collect_with_oracles(scenario: str, n: int, seed: int):
    """One rollout -> (frames, position labels, distance labels).

    Frames and both label sets must come from the SAME states; an earlier version
    generated distance labels from a second rollout, which made that metric
    meaningless.
    """
    rng = np.random.default_rng(seed)
    env = DoomEnv(scenario, seed=seed, keep_full_frames=True, eval_labels=True)
    env.reset()
    frames, pos, dist = [], [], []
    while len(frames) < n:
        _, _, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
        if term or trunc:
            env.reset()
            continue
        if rng.random() < 0.25:
            objs = [x for x in env.game.get_state().labels if "medi" in x.object_name.lower()]
            near = max(objs, key=lambda x: x.height) if objs else None
            frames.append(env.teacher_frames())
            if near is None:
                pos.append("none")
                dist.append("none")
            else:
                cx = (near.x + near.width / 2) / env.game.get_screen_width()
                pos.append("left" if cx < 0.4 else "right" if cx > 0.6 else "center")
                dist.append("close" if near.height >= NEAR_BBOX_H else "far")
    env.close()
    return frames, pos, dist


def batched(teacher, frames, question, n_opt, bs=4):
    return np.concatenate([teacher.choice_probs(frames[i:i + bs], question, n_opt)
                           for i in range(0, len(frames), bs)])


def distance_probs(teacher, frames, perms) -> np.ndarray:
    opts = {"close": "the nearest medkit is close, a step or two away",
            "far": "the nearest medkit is far away",
            "none": "no medkit is visible"}
    keys = list(opts)
    total = np.zeros((len(frames), 3), np.float64)
    for perm in perms:
        text = "\n".join(f"{LETTERS[k]}. {opts[keys[i]]}" for k, i in enumerate(perm))
        q = ("This is a first-person view from the game Doom. How far away is the nearest "
             f"medkit in the last image?\n{text}\nAnswer with a single letter.")
        total[:, perm] += batched(teacher, frames, q, 3)
    return total / len(perms), keys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--nf4", action="store_true")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--perms", type=int, default=2)
    p.add_argument("--out", default="runs/hg_probe")
    args = p.parse_args()
    from reflexrl.teacher.qwen import QwenTeacher
    teacher = QwenTeacher(args.model, dtype=torch.float32, load_4bit=args.nf4, device_map="auto")

    frames, pos_truth, dist_truth = collect_with_oracles("health_gathering", args.frames, seed=777)
    rng = np.random.default_rng(0)
    perms4 = [np.arange(4)] + [rng.permutation(4) for _ in range(args.perms - 1)]
    perms3 = [np.arange(3)] + [rng.permutation(3) for _ in range(args.perms - 1)]

    blank = [[np.zeros_like(frames[0][0]) for _ in frames[0]]]
    pos = perception_probs(teacher, frames, "medkit", perms4)
    pos_prior = perception_probs(teacher, blank, "medkit", perms4)[0]
    pos_cal = calibrate(pos, pos_prior)
    pos_pred = [CLASSES[i] for i in pos.argmax(1)]
    pos_pred_cal = [CLASSES[i] for i in pos_cal.argmax(1)]
    pos_acc = float(np.mean([a == b for a, b in zip(pos_pred, pos_truth, strict=True)]))
    pos_acc_cal = float(np.mean([a == b for a, b in zip(pos_pred_cal, pos_truth, strict=True)]))
    pos_major = max(pos_truth.count(c) for c in CLASSES) / len(pos_truth)
    print(f"position: raw {pos_acc:.3f} -> calibrated {pos_acc_cal:.3f} (majority {pos_major:.3f})",
          flush=True)

    dist, dkeys = distance_probs(teacher, frames, perms3)
    dist_prior = distance_probs(teacher, blank, perms3)[0][0]
    dist_cal = calibrate(dist, dist_prior)
    dist_pred = [dkeys[i] for i in dist.argmax(1)]
    dist_pred_cal = [dkeys[i] for i in dist_cal.argmax(1)]
    dist_acc = float(np.mean([a == b for a, b in zip(dist_pred, dist_truth, strict=True)]))
    dist_acc_cal = float(np.mean([a == b for a, b in zip(dist_pred_cal, dist_truth, strict=True)]))
    dist_major = max(dist_truth.count(c) for c in set(dist_truth)) / len(dist_truth)

    res = {"meta": run_metadata(vars(args)),
           "position": {"accuracy_calibrated": pos_acc_cal, "accuracy": pos_acc, "majority_rate": pos_major,
                        "counts": {c: pos_truth.count(c) for c in CLASSES},
                        "pass": pos_acc >= pos_major + 0.20},
           "distance": {"accuracy_calibrated": dist_acc_cal, "accuracy": dist_acc, "majority_rate": dist_major,
                        "counts": {c: dist_truth.count(c) for c in set(dist_truth)}}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "hg_probe.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
