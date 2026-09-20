"""Final report: every finished policy on fresh, never-used episodes.

Training used seeds 0-11 / 1000-1011 / 2000-2011, the in-training evaluation
used 900000+, and the adaptive handover read those scores. The numbers reported
in the write-up therefore come from this separate stream, which nothing has
seen.

    python scripts/final_eval.py --runs runs/train/defend_the_center --episodes 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.env.scenarios import get_scenario  # noqa: E402
from reflexrl.policy.actor_critic import ActorCritic  # noqa: E402
from reflexrl.rl.evaluate import evaluate_policy  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402

TEST_SEED_BASE = 7_000_000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/train/defend_the_center")
    p.add_argument("--scenario", default="dtc")
    p.add_argument("--teachers", default="runs/teachers")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/final_eval.json")
    args = p.parse_args()
    spec = get_scenario(args.scenario)
    n_act = len(spec.actions)
    out = {"meta": run_metadata(vars(args)), "test_seed_base": TEST_SEED_BASE,
           "episodes": args.episodes, "policies": {}}

    for run in sorted(Path(args.runs).iterdir()):
        ckpt = run / "ckpt_final.pt"
        if not ckpt.exists():
            continue
        policy = ActorCritic(n_act)
        policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
        ev = evaluate_policy(policy.to(args.device).eval(), spec.name, args.episodes,
                             args.device, seed_base=TEST_SEED_BASE)
        out["policies"][run.name] = {k: v for k, v in ev.items() if k != "returns"}
        print(f"{run.name:<24} {ev['return_mean']:6.2f} +- {ev['return_se']:.2f}", flush=True)

    pj = Path(args.teachers) / spec.name / "perception_jev.pt"
    if pj.exists():  # the distilled teacher (perception student + Jev), for reference
        from reflexrl.baselines.perception_bc import PerceptionJevPolicy, PerceptionNet
        blob = torch.load(pj, map_location="cpu", weights_only=False)
        net = PerceptionNet()
        net.load_state_dict(blob["perception"])
        ev = evaluate_policy(PerceptionJevPolicy(net, blob["J"]).to(args.device).eval(),
                             spec.name, args.episodes, args.device, seed_base=TEST_SEED_BASE)
        out["policies"]["distilled_perception_jev_teacher"] = {k: v for k, v in ev.items() if k != "returns"}
        print(f"{'perception+Jev teacher':<24} {ev['return_mean']:6.2f} +- {ev['return_se']:.2f}", flush=True)

    by = {k: v["return_mean"] for k, v in out["policies"].items()}
    out["summary"] = {"best": max(by, key=by.get) if by else None, "means": by}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
