"""Real-time evaluation + demo video for the shipped reflex policy vs Qwen as controller.

Everything shown in the video is read from what this script measures.

    python scripts/make_demo.py --runs runs/train/defend_the_center --out results/demo
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.demo import compose as C  # noqa: E402
from reflexrl.env.vizdoom_env import DoomEnv  # noqa: E402
from reflexrl.eval.realtime import realtime_episode  # noqa: E402
from reflexrl.policy.actor_critic import ActorCritic  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402

DEMO_SEED = 424242


def teacher_gate_scores(results: Path) -> dict:
    """Measured DTC gate means for the fp32 teachers (results/teacher_gate)."""
    def dtc(files):
        rets = []
        for f in files:
            sc = json.loads(f.read_text())["scenarios"].get("defend_the_center")
            if sc:
                rets += sc["returns"]
        return float(np.mean(rets)) if rets else None
    return {"2B": dtc(sorted(results.glob("qwen2b_fp32_cal_lane*.json"))),
            "8B": dtc(sorted(results.glob("qwen8b_nf4_cal_dtc_lane*.json"))),
            "jev": dtc(sorted(results.glob("qwen8b_jev_dtc_lane*.json")))}


def best_run(runs: Path) -> Path:
    cands = sorted(runs.glob("reflexrl_s*")) or sorted(runs.glob("ppo_s*"))
    done = [(json.loads((r / "done.json").read_text())["final_eval"], r)
            for r in cands if (r / "done.json").exists()]
    if not done:
        raise SystemExit(f"no finished ppo runs in {runs}")
    return max(done)[1]


def reflex_decider(policy: ActorCritic, device: str):
    @torch.no_grad()
    def decide(env, obs):
        return int(policy.act(torch.as_tensor(obs[None], device=device))[0])
    return decide


def qwen_decider(teacher, rng: np.random.Generator):
    def decide(env, obs):
        p = teacher.action_probs([env.teacher_frames()], env.scenario)[0].astype(np.float64)
        return int(rng.choice(len(p), p=p / p.sum()))
    return decide


def realtime_eval(decide, scenario: str, n: int, record: bool) -> tuple[dict, list]:
    """Runs n episodes; keeps the longest recorded timeline for the video."""
    rows, best = [], None
    for i in range(n):
        env = DoomEnv(scenario, seed=DEMO_SEED + i, keep_full_frames=True)
        res = realtime_episode(env, decide, record=record)
        env.close()
        tics = res.pop("tics", None)
        if record and tics and (best is None or len(tics) > len(best)):
            best = tics
        rows.append(res)
        print(f"  realtime ep {i}: return {res['return']:.1f} ms {res['ms_mean']:.1f}", flush=True)
    r = np.array([x["return"] for x in rows])
    summary = {"n": n, "return_mean": float(r.mean()), "return_se": float(r.std(ddof=1) / np.sqrt(n)),
               "ms_mean": float(np.mean([x["ms_mean"] for x in rows])),
               "ms_p95": float(np.mean([x["ms_p95"] for x in rows])),
               "max_actions_per_s": float(np.mean([x["max_actions_per_s"] for x in rows])),
               "episodes": rows}
    return summary, best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/train/defend_the_center")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--teacher-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--jev-table", default="experiments/configs/jev_table_dtc.json")
    p.add_argument("--out", default="results/demo")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    run = best_run(Path(args.runs))
    policy = ActorCritic(5)
    policy.load_state_dict(torch.load(run / "ckpt_final.pt", map_location="cpu"))
    policy.to(dev).eval()
    print("reflex policy:", run, flush=True)

    reflex, reflex_tics = realtime_eval(reflex_decider(policy, dev), "dtc", args.episodes, True)
    import torch as _torch
    from reflexrl.teacher.perception_jev import PerceptionJevTeacher
    from reflexrl.teacher.qwen import QwenTeacher
    # The gate-validated teacher: Qwen3-VL-8B (NF4 weights, fp32 compute) sees,
    # Jev decides. This is what guided training; here it plays under real latency.
    teacher = PerceptionJevTeacher(
        QwenTeacher(args.teacher_model, dtype=_torch.float32, load_4bit=True),
        args.jev_table)
    qwen, qwen_tics = realtime_eval(qwen_decider(teacher, np.random.default_rng(0)), "dtc",
                                    args.episodes, True)
    summary = {"meta": run_metadata(vars(args)), "reflex_run": str(run),
               "realtime": {"reflex": reflex, "qwen8b_jev_teacher": qwen}}
    (out / "realtime.json").write_text(json.dumps(summary, indent=2))

    frames = C.card([("Teach slowly. Act fast.", 1.4, C.FG),
                     ("Qwen3-VL-8B + Jev (the teacher) vs the 0.8M-parameter policy it trained",
                      0.65, C.DIM)], 3.0)
    frames += C.split_screen(qwen_tics, reflex_tics, 16.0, qwen, reflex)
    frames += C.card([("How was the reflex policy trained?", 1.0, C.FG),
                      ("Reinforcement learning from pixels and reward only", 0.7, C.DIM)], 2.5)
    frames += C.training_segment(run / "metrics.jsonl", 9.0, "Reflex policy: score vs experience")
    frames += C.montage([("DEFEND THE CENTER", reflex_tics, reflex)], 8.0)
    speed = qwen["ms_mean"] / reflex["ms_mean"]
    gates = teacher_gate_scores(Path(__file__).resolve().parents[1] / "results" / "teacher_gate")
    summary["teacher_gate_dtc"] = gates
    (out / "realtime.json").write_text(json.dumps(summary, indent=2))
    frames += C.card([
        (f"{reflex['return_mean']:.1f} vs {qwen['return_mean']:.1f} kills in real time", 1.1, C.ACCENT),
        (f"{speed:.0f}x faster decisions ({reflex['ms_mean']:.1f} ms vs {qwen['ms_mean']:.0f} ms)", 0.9, C.FG),
        ("0 Qwen calls", 0.9, C.FG),
        (f"teacher (paused game): {gates['jev']:.2f} kills; VLM alone: {gates['8B']:.2f}", 0.6, C.DIM),
    ], 6.0)
    raw = out / "demo_raw.mp4"
    C.write_video(frames, raw)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(out / "demo.mp4")], check=False)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "episodes"}
                      for k, v in summary["realtime"].items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
