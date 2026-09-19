"""Stage-1 small check for a candidate teacher (experiments/configs/bigger_teacher.json).

Synthetic red-disc localisation + answer-letter mass, then the pre-registered
debias probe variants on Defend-the-Center frames.

    python scripts/teacher_size_check.py --model Qwen/Qwen3-VL-4B-Instruct --dtype fp32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.debias_probe import (  # noqa: E402
    CLASSES, N_PERM, collect, perception_probs, permuted_action_probs, score_actions)
from scripts.diagnose_teacher import POS_Q, generate, next_token_report, synthetic  # noqa: E402
from reflexrl.env.scenarios import get_scenario  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", choices=["fp16", "fp32"], required=True)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--out", default="runs/size_check")
    args = p.parse_args()
    from reflexrl.teacher.qwen import QwenTeacher
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    teacher = QwenTeacher(args.model, dtype=dtype, device_map="auto")
    res = {"model": args.model, "dtype": args.dtype}

    rng = np.random.default_rng(0)
    correct, masses = 0, []
    for pos in ("left", "center", "right", None):
        for _ in range(8):
            rep = next_token_report(teacher, [synthetic(pos, rng)], POS_Q.format(obj="red circle"))
            masses.append(rep["letter_mass"])
            correct += max(rep["letters"], key=rep["letters"].get) == \
                {"left": "A", "center": "B", "right": "C", None: "D"}[pos]
    res["synthetic_accuracy"] = correct / 32
    res["mean_letter_mass"] = float(np.mean(masses))
    res["synthetic_generation"] = generate(teacher, [synthetic("left", rng)],
                                           "Describe this image in one sentence.")
    print(json.dumps(res, indent=1), flush=True)
    stage1_ok = res["synthetic_accuracy"] >= 0.9 and res["mean_letter_mass"] >= 0.9
    res["stage1_synthetic_pass"] = stage1_ok

    if stage1_ok:
        spec = get_scenario("dtc")
        frames, classes, _ = collect(spec.name, args.frames, seed=777)
        n = len(spec.actions)
        perms = [np.arange(n)] + [rng.permutation(n) for _ in range(N_PERM - 1)]
        a = permuted_action_probs(teacher, spec, frames, perms[:1])
        b = permuted_action_probs(teacher, spec, frames, perms)
        prior = permuted_action_probs(teacher, spec, [[np.zeros_like(frames[0][0])] * 2], perms)[0]
        c = b / prior
        c = c / c.sum(1, keepdims=True)
        d = perception_probs(teacher, frames, "monster", [np.arange(4)] + [rng.permutation(4) for _ in range(3)])
        pred = [CLASSES[i] for i in d.argmax(1)]
        res["probe"] = {"A": score_actions(spec, a, classes), "B": score_actions(spec, b, classes),
                        "C": score_actions(spec, c, classes),
                        "perception_accuracy": float(np.mean([x == y for x, y in zip(pred, classes, strict=True)]))}
        best = max(res["probe"][v]["argmax_in_set"] for v in "ABC")
        res["stage1_probe_pass"] = best >= res["probe"]["A"]["uniform_set_rate"] + 0.15
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1] + "_" + args.dtype
    (Path(args.out) / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "synthetic_generation"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
