"""Fit the perception student on the Qwen+Jev labels; Jev stays the decision maker.

    python scripts/build_perception_teacher.py --labels <dir> --out runs/teachers_pjev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.baselines.perception_bc import (  # noqa: E402
    PerceptionJevPolicy, perception_accuracy, recover_perception, train_perception_bc)
from reflexrl.env.scenarios import get_scenario  # noqa: E402
from reflexrl.rl.evaluate import evaluate_policy  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402
from reflexrl.teacher.dataset import load_labels  # noqa: E402
from reflexrl.teacher.perception_jev import CLASSES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="dtc")
    p.add_argument("--labels", required=True)
    p.add_argument("--jev-table", default="experiments/configs/jev_table_dtc.json")
    p.add_argument("--out", default="runs/teachers")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    spec = get_scenario(args.scenario)
    labels = load_labels(Path(args.labels) / spec.name)
    data = json.loads(Path(args.jev_table).read_text())
    J = np.array([data["table"][c] for c in CLASSES], np.float64)
    J /= J.sum(1, keepdims=True)
    net, fid = train_perception_bc(labels, J, device=args.device, epochs=args.epochs)
    policy = PerceptionJevPolicy(net, J).to(args.device).eval()
    ev = evaluate_policy(policy, spec.name, args.eval_episodes, args.device)

    q = recover_perception(labels["probs"].astype(np.float32), J)
    out = Path(args.out) / spec.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"perception": net.state_dict(), "J": J}, out / "perception_jev.pt")
    summary = {"meta": run_metadata(vars(args)), "fidelity": fid, "eval": ev,
               "n_labels": int(len(labels["obs"])),
               "teacher_perception_prior": dict(zip(CLASSES, q.mean(0).round(3).tolist(), strict=True))}
    (out / "perception_teacher.json").write_text(json.dumps(summary, indent=2))
    print(f"{spec.name}: perception student agrees {fid['val_top1_agree']:.3f} on held-out episodes; "
          f"with Jev it scores {ev['return_mean']:.2f} +- {ev['return_se']:.2f}", flush=True)


if __name__ == "__main__":
    main()
