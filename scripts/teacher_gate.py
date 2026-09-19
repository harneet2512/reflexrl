"""Phase 0 teacher gate: Qwen3-VL (or uniform random) as the direct controller.

Runs exactly N episodes per scenario, records returns, teacher latency, and
saves every (student_obs, pi_T) pair as reusable teacher labels.

    python scripts/teacher_gate.py --policy qwen --scenarios dtc hg dc --episodes 30
    python scripts/teacher_gate.py --policy random --scenarios dtc hg dc --episodes 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.env.vizdoom_env import DoomEnv  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402
from reflexrl.teacher.dataset import LabelWriter  # noqa: E402

SEED_BASE = 50_000  # env seeds shared by teacher and random runs


def run_scenario(scenario: str, policy: str, episodes: int, n_envs: int, teacher,
                 rng: np.random.Generator, out_dir: Path, seed_offset: int = 0) -> dict:
    envs = [DoomEnv(scenario, seed=SEED_BASE + seed_offset + i, keep_full_frames=True)
            for i in range(n_envs)]
    spec = envs[0].scenario
    n_actions = len(spec.actions)
    writer = LabelWriter(out_dir / "labels" / spec.name) if teacher is not None else None

    obs = [e.reset()[0] for e in envs]
    active = [True] * n_envs
    started = n_envs if episodes >= n_envs else episodes
    for i in range(started, n_envs):
        active[i] = False
    ep_ids = list(range(n_envs))
    step_in_ep = [0] * n_envs
    returns, lengths = [], []
    t0 = time.time()

    while any(active):
        idx = [i for i in range(n_envs) if active[i]]
        if teacher is not None:
            probs = teacher.action_probs([envs[i].teacher_frames() for i in idx], spec)
        else:
            probs = np.full((len(idx), n_actions), 1.0 / n_actions, dtype=np.float32)
        for j, i in enumerate(idx):
            p = probs[j].astype(np.float64)
            a = int(rng.choice(n_actions, p=p / p.sum()))
            if writer is not None:
                writer.add(obs[i], probs[j], a, ep_ids[i], step_in_ep[i])
            obs[i], _, term, trunc, info = envs[i].step(a)
            step_in_ep[i] += 1
            if term or trunc:
                returns.append(info["episode"]["r"])
                lengths.append(info["episode"]["l"])
                print(f"  {spec.name} ep {len(returns)}/{episodes}: return "
                      f"{returns[-1]:.1f} len {lengths[-1]} ({time.time() - t0:.0f}s)", flush=True)
                if started < episodes:
                    obs[i] = envs[i].reset()[0]
                    ep_ids[i] = started
                    step_in_ep[i] = 0
                    started += 1
                else:
                    active[i] = False
    for e in envs:
        e.close()
    if writer is not None:
        writer.close()

    r = np.asarray(returns, dtype=np.float64)
    out = {
        "scenario": spec.name, "policy": policy, "episodes": len(r),
        "return_mean": float(r.mean()), "return_std": float(r.std(ddof=1)),
        "return_se": float(r.std(ddof=1) / np.sqrt(len(r))),
        "len_mean": float(np.mean(lengths)), "returns": r.tolist(), "lengths": lengths,
        "decisions": int(np.sum(lengths)), "wall_s": round(time.time() - t0, 1),
    }
    if teacher is not None:
        out["teacher_s_per_decision"] = teacher.seconds / max(teacher.samples, 1)
        out["teacher_nonfinite_retries"] = teacher.nonfinite_retries
        out["teacher_dtype"] = str(teacher.dtype)
        out["teacher_mean_letter_mass"] = teacher.letter_mass_sum / max(teacher.samples, 1)
    print(f"[{policy}] {spec.name}: {out['return_mean']:.2f} +- {out['return_se']:.2f} "
          f"(n={len(r)}, len {out['len_mean']:.0f}, {out['wall_s']}s)", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["qwen", "random"], required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--scenarios", nargs="+", default=["dtc", "hg", "dc"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/phase0")
    p.add_argument("--model-dtype", choices=["auto", "fp32"], default="auto")
    p.add_argument("--nf4", action="store_true", help="4-bit NF4 weights (compute stays fp32)")
    p.add_argument("--seed-offset", type=int, default=0, help="env seed offset for split runs")
    p.add_argument("--jev-table", default="experiments/configs/jev_table_dtc.json")
    p.add_argument("--variant", choices=["raw", "calibrated", "perception_jev"], default="calibrated",
                   help="calibrated = probe-validated variant C (debias_probe.json)")
    args = p.parse_args()

    tag = args.policy if args.policy == "random" else args.model.split("/")[-1]
    if args.policy == "qwen" and args.variant == "calibrated":
        tag += "_cal"
    if args.policy == "qwen" and args.variant == "perception_jev":
        tag += "_pjev"
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher = None
    if args.policy == "qwen":
        from reflexrl.teacher.qwen import QwenTeacher
        import torch
        teacher = QwenTeacher(args.model, load_4bit=args.nf4,
                              dtype=torch.float32 if (args.nf4 or args.model_dtype == "fp32") else None)
        if args.variant == "calibrated":
            from reflexrl.teacher.qwen import CalibratedQwenTeacher
            teacher = CalibratedQwenTeacher(teacher)
        elif args.variant == "perception_jev":
            from reflexrl.teacher.perception_jev import PerceptionJevTeacher
            teacher = PerceptionJevTeacher(teacher, args.jev_table)
    rng = np.random.default_rng(args.seed)

    res_path = out_dir / "gate_results.json"
    results = (json.loads(res_path.read_text()) if res_path.exists()
               else {"meta": run_metadata(vars(args)), "scenarios": {}})
    from reflexrl.env.scenarios import get_scenario
    for sc in args.scenarios:
        if get_scenario(sc).name in results["scenarios"]:
            print(f"skip {sc}: already in {res_path}", flush=True)
            continue
        label_dir = out_dir / "labels" / get_scenario(sc).name
        for stale in label_dir.glob("shard_*.npz"):
            stale.unlink()  # a partial scenario is rerun from scratch
        res = run_scenario(sc, args.policy, args.episodes, args.n_envs, teacher, rng, out_dir,
                           seed_offset=args.seed_offset)
        results["scenarios"][res["scenario"]] = res
        (out_dir / "gate_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'gate_results.json'}")


if __name__ == "__main__":
    main()
