"""Fit the BC network on Qwen labels for one scenario; it is both the BC
baseline and the frozen teacher proxy used by guided RL.

    python scripts/build_teacher.py --scenario dtc --labels runs/phase0/Qwen3-VL-2B-Instruct/labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.baselines.bc import train_bc  # noqa: E402
from reflexrl.env.scenarios import get_scenario  # noqa: E402
from reflexrl.rl.evaluate import evaluate_policy  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402
from reflexrl.teacher.dataset import load_labels  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--labels", required=True, help="dir containing <scenario_name>/shard_*.npz")
    p.add_argument("--out", default="runs/teachers")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    spec = get_scenario(args.scenario)
    labels = load_labels(Path(args.labels) / spec.name)
    policy, fid = train_bc(labels, len(spec.actions), device=args.device, epochs=args.epochs)
    ev = evaluate_policy(policy, spec.name, args.eval_episodes, args.device)
    out = Path(args.out) / spec.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), out / "proxy.pt")
    summary = {"meta": run_metadata(vars(args)), "fidelity": fid, "bc_eval": ev,
               "n_labels": int(len(labels["obs"]))}
    (out / "teacher.json").write_text(json.dumps(summary, indent=2))
    print(f"{spec.name}: BC/proxy eval {ev['return_mean']:.2f} +- {ev['return_se']:.2f}; "
          f"fidelity top1 {fid['val_top1_agree']:.3f} KL {fid['val_kl']:.3f}", flush=True)


if __name__ == "__main__":
    main()
