"""Rebuild the scoreboard (results/SCOREBOARD.md) from whatever results exist.

Every number is read from a file produced by a run; nothing is hard-coded, so
re-running this after new results refreshes the table.

    python scripts/scoreboard.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DTC = "defend_the_center"


def gate(pattern: str, scenario: str = DTC) -> tuple[float, int] | None:
    rets: list[float] = []
    for f in glob.glob(str(REPO / "results" / "teacher_gate" / pattern)):
        sc = json.loads(Path(f).read_text())["scenarios"].get(scenario)
        if sc:
            rets += sc["returns"]
    return (float(np.mean(rets)), len(rets)) if rets else None


def runs(pattern: str) -> list[tuple[str, np.ndarray, float]]:
    out = []
    for m in sorted(glob.glob(str(REPO / pattern))):
        p = Path(m)
        done = p.parent / "done.json"
        if not done.exists():
            continue
        ev = [json.loads(line) for line in p.read_text().splitlines() if '"eval"' in line]
        curve = np.array([[e["step"], e["return_mean"]] for e in ev if not e.get("final")])
        out.append((p.parent.name, curve, json.loads(done.read_text())["final_eval"]))
    return out


def steps_to(curve: np.ndarray, target: float, extra: int = 0) -> float | None:
    k = 3
    if len(curve) < k:
        return None
    trail = np.convolve(curve[:, 1], np.ones(k) / k, "valid")
    hit = np.flatnonzero(trail >= target)
    return float(curve[hit[0] + k - 1, 0] + extra) if len(hit) else None


def main() -> None:
    lines = ["# Scoreboard", "", "Every number below is read from a results file in this repo; "
             "`python scripts/scoreboard.py` rebuilds it.", ""]
    rand = gate("random_gate.json")
    label_steps = sum(json.loads(Path(f).read_text())["scenarios"][DTC]["decisions"]
                      for f in glob.glob(str(REPO / "results/teacher_gate/qwen8b_jev_dtc_lane*.json")))

    lines += ["## Controllers (game paused; 30-episode gates, seeds 50,000+)", "",
              "| controller | kills | episodes |", "|---|---|---|"]
    for label, pat in (("random", "random_gate.json"),
                       ("Qwen3-VL-2B alone (calibrated)", "qwen2b_fp32_cal_lane*.json"),
                       ("Qwen3-VL-8B alone (calibrated)", "qwen8b_nf4_cal_dtc_lane*.json"),
                       ("**Qwen3-VL-8B sees + Jev decides**", "qwen8b_jev_dtc_lane*.json")):
        g = gate(pat)
        if g:
            lines.append(f"| {label} | {g[0]:.2f} | {g[1]} |")

    collected = {
        "PPO from scratch": "runs/kaggle_ppo/runs/train/defend_the_center/ppo_s*/metrics.jsonl",
        "BC -> PPO": "kaggle/train_lane1/output3/runs/train/defend_the_center/bc_ppo_s*/metrics.jsonl",
        "ReflexRL (offline teacher)": "kaggle/train_lane0/output/runs/train/defend_the_center/reflexrl_s*/metrics.jsonl",
        "ReflexRL (fixed schedule)": "kaggle/train_lane1/output/runs/train/defend_the_center/fixed_s*/metrics.jsonl",
        "ReflexRL (live Qwen+Jev teacher)": "archive/kaggle/reflexrl-live-teacher/**/reflexrl_live_s*/metrics.jsonl",
        "ReflexRL (action-cloned teacher)": "archive/kaggle/reflexrl-ablation/**/reflexrl_actionclone_s*/metrics.jsonl",
    }
    ppo = runs(collected["PPO from scratch"])
    if ppo and rand:
        ppo_final = float(np.mean([f for _, _, f in ppo]))
        target = rand[0] + 0.8 * (ppo_final - rand[0])
        base = np.median([steps_to(c, target) or np.inf for _, c, _ in ppo])
        lines += ["", f"## Learned policies (1.5M steps; target R* = {target:.2f})", "",
                  f"Teacher-label collection ({label_steps:,} env steps) is charged to every "
                  "teacher-using method.", "",
                  "| method | final (per seed) | steps to R* | X |", "|---|---|---|---|"]
        for label, pat in collected.items():
            rs = runs(pat)
            if not rs:
                continue
            extra = 0 if label == "PPO from scratch" else label_steps
            hits = [steps_to(c, target, extra) for _, c, _ in rs]
            fins = [f for _, _, f in rs]
            med = np.median(hits) if all(h is not None for h in hits) else None
            x = f"{base / med:.2f}x" if med else "-"
            hs = ", ".join(f"{int(h / 1000)}K" if h else "never" for h in hits)
            lines.append(f"| {label} | {np.mean(fins):.2f} ({', '.join(f'{f:.1f}' for f in fins)}) "
                         f"| {hs} | {x} |")

    fe = REPO / "results" / "final_eval.json"
    if fe.exists():
        d = json.loads(fe.read_text())
        lines += ["", f"## Final evaluation on unseen episodes (seeds {d['test_seed_base']:,}+, "
                  f"{d['episodes']} episodes)", "", "| policy | kills |", "|---|---|"]
        for k, v in sorted(d["policies"].items(), key=lambda kv: -kv[1]["return_mean"]):
            lines.append(f"| {k} | {v['return_mean']:.2f} ± {v['return_se']:.2f} |")

    bench = sorted(glob.glob(str(REPO / "archive/kaggle/reflexrl-demo/**/benchmark/*.json"),
                             recursive=True)) or sorted(glob.glob(str(REPO / "kaggle/demo/output/results/benchmark/*.json")))
    if bench:
        b = json.loads(Path(bench[0]).read_text())
        lines += ["", "## Deployment (same T4)", "", "| | reflex policy | teacher model |",
                  "|---|---|---|",
                  f"| parameters | {b['reflex']['params']:,} | {b.get('qwen', {}).get('params', 0):,} |",
                  f"| FLOPs per action | {b['reflex']['flops_per_action']/1e6:.1f}M | "
                  f"{b.get('qwen', {}).get('flops_per_action', 0)/1e12:.2f}T |"]
        if "gpu" in b["reflex"]:
            lines.append(f"| latency (GPU) | {b['reflex']['gpu']['ms_mean']:.2f} ms | "
                         f"{b.get('qwen', {}).get('gpu', {}).get('ms_mean', float('nan')):.0f} ms |")
        if "ratios_same_gpu" in b:
            r = b["ratios_same_gpu"]
            lines.append(f"| ratio | **{r['speedup_mean']:.0f}x faster**, "
                         f"**{r['cost_ratio']:.0f}x cheaper** | 1x |")

    rt = sorted(glob.glob(str(REPO / "archive/kaggle/reflexrl-demo/**/demo/realtime.json"),
                          recursive=True)) or sorted(glob.glob(str(REPO / "kaggle/demo/output/results/demo/realtime.json")))
    if rt:
        d = json.loads(Path(rt[0]).read_text())["realtime"]
        lines += ["", "## Real time (the game does not wait for a slow controller)", "",
                  "| controller | kills | ms per decision |", "|---|---|---|"]
        for k, v in d.items():
            lines.append(f"| {k} | {v['return_mean']:.2f} ± {v['return_se']:.2f} | {v['ms_mean']:.1f} |")

    out = REPO / "results" / "SCOREBOARD.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
