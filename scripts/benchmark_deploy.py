"""Deployment benchmark: reflex policy vs Qwen3-VL as the controller.

Measures params, FLOPs/action, latency (mean/median/P95), actions/s, peak VRAM
and cost per 10K decisions on the same machine, on real rendered frames.

    python scripts/benchmark_deploy.py --scenario dtc --ckpt runs/train/defend_the_center/reflexrl_s0/ckpt_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.env.observations import STUDENT_H, STUDENT_STACK, STUDENT_W  # noqa: E402
from reflexrl.env.vizdoom_env import DoomEnv  # noqa: E402
from reflexrl.eval.latency import cost_per_10k, student_latency, teacher_latency  # noqa: E402
from reflexrl.policy.actor_critic import ActorCritic, n_params  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402


def collect_frames(scenario: str, n: int = 8) -> list[np.ndarray]:
    env = DoomEnv(scenario, seed=123, keep_full_frames=True)
    env.reset()
    frames = []
    for _ in range(n):
        env.step(env.action_space.sample())
        frames.append(env.teacher_frames()[-1])
    env.close()
    return frames


def student_flops(policy: ActorCritic) -> int:
    x = torch.zeros((1, 3 * STUDENT_STACK, STUDENT_H, STUDENT_W), dtype=torch.uint8)
    with FlopCounterMode(display=False) as fc:
        policy.cpu()(x)
    return int(fc.get_total_flops())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--usd-per-gpu-hour", type=float, default=0.59,
                   help="price applied to BOTH controllers (default: Modal T4 list price)")
    p.add_argument("--usd-per-cpu-core-hour", type=float, default=0.047)
    p.add_argument("--skip-teacher", action="store_true")
    p.add_argument("--out", default="runs/benchmark")
    args = p.parse_args()

    frames = collect_frames(args.scenario)
    env = DoomEnv(args.scenario)
    spec, n_act = env.scenario, env.action_space.n
    env.close()
    policy = ActorCritic(n_act)
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    res = {"meta": run_metadata(vars(args)), "scenario": spec.name}
    res["reflex"] = {"params": n_params(policy), "flops_per_action": student_flops(policy)}
    torch.set_num_threads(1)
    res["reflex"]["cpu_1core"] = student_latency(policy, frames, "cpu")
    res["reflex"]["cpu_1core"]["usd_per_10k"] = cost_per_10k(
        res["reflex"]["cpu_1core"]["ms_mean"], args.usd_per_cpu_core_hour)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        res["reflex"]["gpu"] = student_latency(policy, frames, "cuda")
        res["reflex"]["gpu"]["usd_per_10k"] = cost_per_10k(
            res["reflex"]["gpu"]["ms_mean"], args.usd_per_gpu_hour)

    if not args.skip_teacher:
        from reflexrl.teacher.qwen import QwenTeacher
        teacher = QwenTeacher(args.model)
        res["qwen"] = {"model": args.model,
                       "params": sum(p.numel() for p in teacher.model.parameters())}
        res["qwen"]["gpu"] = teacher_latency(teacher, frames, spec)
        res["qwen"]["gpu"]["usd_per_10k"] = cost_per_10k(
            res["qwen"]["gpu"]["ms_mean"], args.usd_per_gpu_hour)
        batch = teacher.processor(
            text=[teacher._prompt(spec, 2)],
            images=[__import__("PIL").Image.fromarray(f) for f in frames[-2:]],
            return_tensors="pt")
        batch = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in batch.items()}
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            teacher.model(**batch)
        res["qwen"]["flops_per_action"] = int(fc.get_total_flops())
        r, q = res["reflex"]["gpu"], res["qwen"]["gpu"]
        res["ratios_same_gpu"] = {
            "speedup_mean": q["ms_mean"] / r["ms_mean"],
            "speedup_p95": q["ms_p95"] / r["ms_p95"],
            "cost_ratio": q["usd_per_10k"] / r["usd_per_10k"],
            "flops_ratio": res["qwen"]["flops_per_action"] / res["reflex"]["flops_per_action"],
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{spec.name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=2))


if __name__ == "__main__":
    main()
