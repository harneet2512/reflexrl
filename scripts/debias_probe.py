"""Debias probe (experiments/configs/debias_probe.json): does Qwen-2B see anything?

Scores four ways of querying Qwen against an eval-only ground truth taken
from ViZDoom's labels buffer (never exposed to any policy).

    python scripts/debias_probe.py --frames 150 --out runs/probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.env.scenarios import get_scenario  # noqa: E402
from reflexrl.env.vizdoom_env import DoomEnv  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402
from reflexrl.teacher.prompts import LETTERS, build_prompt_text  # noqa: E402

N_PERM = 4
CLASSES = ("left", "center", "right", "none")
TARGET = {"defend_the_center": "monster", "health_gathering": "medkit"}
# Oracle class -> action names that move toward / act on the target.
CORRECT = {
    "defend_the_center": {"left": {"TURN_LEFT", "TURN_LEFT_FIRE"},
                          "right": {"TURN_RIGHT", "TURN_RIGHT_FIRE"},
                          "center": {"FIRE", "TURN_LEFT_FIRE", "TURN_RIGHT_FIRE"}},
    "health_gathering": {"left": {"TURN_LEFT", "FORWARD_TURN_LEFT"},
                         "right": {"TURN_RIGHT", "FORWARD_TURN_RIGHT"},
                         "center": {"FORWARD"}},
}
NOT_TARGET = ("DoomPlayer", "Blood", "Puff", "Bullet", "Clip", "Ammo", "Shell")


def is_target(scenario: str, name: str) -> bool:
    if scenario == "health_gathering":
        return "medi" in name.lower() or "stim" in name.lower()
    return not any(t.lower() in name.lower() for t in NOT_TARGET)


def oracle(env: DoomEnv, scenario: str) -> tuple[str, list[str]]:
    state = env.game.get_state()
    objs = [lab for lab in state.labels if is_target(scenario, lab.object_name)]
    names = [lab.object_name for lab in state.labels]
    if not objs:
        return "none", names
    near = max(objs, key=lambda lab: lab.height)
    cx = (near.x + near.width / 2) / env.game.get_screen_width()
    return ("left" if cx < 0.4 else "right" if cx > 0.6 else "center"), names


def collect(scenario: str, n: int, seed: int) -> tuple[list, list, dict]:
    rng = np.random.default_rng(seed)
    env = DoomEnv(scenario, seed=seed, keep_full_frames=True, eval_labels=True)
    env.reset()
    frames, classes, seen = [], [], {}
    while len(frames) < n:
        _, _, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
        if term or trunc:
            env.reset()
            continue
        if rng.random() < 0.25:
            cls, names = oracle(env, scenario)
            for nm in names:
                seen[nm] = seen.get(nm, 0) + 1
            frames.append(env.teacher_frames())
            classes.append(cls)
    env.close()
    return frames, classes, seen


def permuted_action_probs(teacher, spec, frames, perms) -> np.ndarray:
    """Average pi_T over option orders, mapped back to canonical action order."""
    names = spec.action_names
    total = np.zeros((len(frames), len(names)), np.float64)
    for perm in perms:
        order = [names[i] for i in perm]
        options = "\n".join(f"{LETTERS[k]}. {nm}" for k, nm in enumerate(order))
        base = build_prompt_text(spec)
        question = base[: base.index("Choose the best action")] + (
            f"Choose the best action right now:\n{options}\nAnswer with a single letter.")
        p = batched(teacher, frames, question, len(names))
        total[:, perm] += p
    return total / len(perms)


def batched(teacher, frames, question, n_opt, bs=8) -> np.ndarray:
    out = [teacher.choice_probs(frames[i:i + bs], question, n_opt) for i in range(0, len(frames), bs)]
    return np.concatenate(out)


def perception_probs(teacher, frames, target, perms) -> np.ndarray:
    total = np.zeros((len(frames), 4), np.float64)
    labels = {"left": f"on the left side", "center": "in the center, straight ahead",
              "right": "on the right side", "none": f"no {target} is visible"}
    for perm in perms:
        order = [CLASSES[i] for i in perm]
        options = "\n".join(f"{LETTERS[k]}. {labels[c]}" for k, c in enumerate(order))
        q = (f"This is a first-person view from the game Doom. Where is the nearest {target} "
             f"in the last image?\n{options}\nAnswer with a single letter.")
        total[:, perm] += batched(teacher, frames, q, 4)
    return total / len(perms)


def score_actions(spec, probs: np.ndarray, classes: list[str]) -> dict:
    names = spec.action_names
    corr = CORRECT[spec.name]
    mass, hit, uni = [], [], []
    for p, c in zip(probs, classes, strict=True):
        if c == "none":
            continue
        idx = [i for i, nm in enumerate(names) if nm in corr[c]]
        mass.append(float(p[idx].sum()))
        hit.append(float(int(np.argmax(p)) in idx))
        uni.append(len(idx) / len(names))
    return {"n": len(mass), "mass_on_correct": float(np.mean(mass)),
            "argmax_in_set": float(np.mean(hit)), "uniform_set_rate": float(np.mean(uni))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--out", default="runs/probe")
    args = p.parse_args()
    from reflexrl.teacher.qwen import QwenTeacher
    teacher = QwenTeacher()
    rng = np.random.default_rng(0)
    out = {"meta": run_metadata(vars(args)), "scenarios": {}}
    for sc in ("dtc", "hg"):
        spec = get_scenario(sc)
        frames, classes, seen = collect(spec.name, args.frames, seed=777)
        counts = {c: classes.count(c) for c in CLASSES}
        print(spec.name, "oracle classes", counts, "objects seen", dict(sorted(seen.items())), flush=True)
        assert counts["none"] < len(classes), "oracle found no targets: check object names"
        n_act = len(spec.actions)
        perms = [np.arange(n_act)] + [rng.permutation(n_act) for _ in range(N_PERM - 1)]
        a_orig = permuted_action_probs(teacher, spec, frames, perms[:1])
        b_perm = permuted_action_probs(teacher, spec, frames, perms)
        blank = [[np.zeros_like(frames[0][0]) for _ in frames[0]]]
        prior = permuted_action_probs(teacher, spec, blank, perms)[0]
        c_cal = b_perm / prior
        c_cal = c_cal / c_cal.sum(1, keepdims=True)
        p_perms = [np.arange(4)] + [rng.permutation(4) for _ in range(N_PERM - 1)]
        d = perception_probs(teacher, frames, TARGET[spec.name], p_perms)
        pred = [CLASSES[i] for i in d.argmax(1)]
        acc = float(np.mean([a == b for a, b in zip(pred, classes, strict=True)]))
        majority = max(counts.values()) / len(classes)
        res = {"oracle_counts": counts, "blank_prior": prior.tolist(),
               "A_original": score_actions(spec, a_orig, classes),
               "B_permuted": score_actions(spec, b_perm, classes),
               "C_calibrated": score_actions(spec, c_cal, classes),
               "D_perception": {"accuracy": acc, "majority_rate": majority,
                                "confusion": {c: {q: int(sum(1 for x, y in zip(classes, pred, strict=True) if x == c and y == q)) for q in CLASSES} for c in CLASSES}}}
        out["scenarios"][spec.name] = res
        print(json.dumps({k: v for k, v in res.items() if k != "blank_prior"}, indent=1), flush=True)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "probe_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
