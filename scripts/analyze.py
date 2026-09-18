"""Compute the pre-registered headline numbers and figures from run logs.

Implements experiments/configs/metrics_prereg.json exactly: R*, steps to
target (with Qwen-label interactions charged to teacher methods), X, W, and
learning / teacher-dependence curves.

    python scripts/analyze.py --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "runs" / "train"
PHASE0 = REPO / "runs" / "phase0"
TEACHER_METHODS = {"bc", "bc_ppo", "fixed", "reflexrl"}
TARGET_FRAC = 0.8
SMOOTH = 3


def load_evals(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    steps, rets, pt, final = [], [], [], None
    for line in (run / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") != "eval":
            continue
        if row.get("final"):
            final = row
            continue
        steps.append(row["step"])
        rets.append(row["return_mean"])
        pt.append(row["p_teacher"])
    return np.asarray(steps), np.asarray(rets), np.asarray(pt), final


def steps_to_target(steps: np.ndarray, rets: np.ndarray, target: float) -> float | None:
    if len(rets) < SMOOTH:
        return None
    trail = np.convolve(rets, np.ones(SMOOTH) / SMOOTH, mode="valid")
    hit = np.flatnonzero(trail >= target)
    return float(steps[hit[0] + SMOOTH - 1]) if len(hit) else None


def label_steps(scenario: str, teacher_tag: str) -> int:
    res = json.loads((PHASE0 / teacher_tag / "gate_results.json").read_text())
    return int(res["scenarios"][scenario]["decisions"])


def gate_mean(tag: str, scenario: str) -> float | None:
    f = PHASE0 / tag / "gate_results.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["scenarios"].get(scenario, {}).get("return_mean")


def analyze_scenario(sdir: Path, teacher_tag: str) -> dict:
    scenario = sdir.name
    runs: dict[str, list[dict]] = {}
    for run in sorted(sdir.iterdir()):
        if not (run / "done.json").exists():
            continue
        method, seed = run.name.rsplit("_s", 1)
        steps, rets, pt, final = load_evals(run)
        runs.setdefault(method, []).append({"seed": int(seed), "steps": steps, "rets": rets,
                                            "p": pt, "final": final["return_mean"]})
    rand = gate_mean("random", scenario)
    qwen = gate_mean(teacher_tag, scenario)
    out = {"scenario": scenario, "random": rand, "qwen": qwen, "methods": {}}
    if "ppo" not in runs or rand is None:
        return out
    ppo_final = float(np.mean([r["final"] for r in runs["ppo"]]))
    target = rand + TARGET_FRAC * (ppo_final - rand)
    out.update({"ppo_final": ppo_final, "R_star": target})
    charge = label_steps(scenario, teacher_tag) if qwen is not None else 0
    for method, rs in runs.items():
        n_hit = [steps_to_target(r["steps"], r["rets"], target) for r in rs]
        extra = charge if method in TEACHER_METHODS else 0
        finals = [r["final"] for r in rs]
        out["methods"][method] = {
            "seeds": [r["seed"] for r in rs],
            "final_per_seed": finals,
            "final_mean": float(np.mean(finals)),
            "final_se": float(np.std(finals, ddof=1) / np.sqrt(len(finals))) if len(finals) > 1 else 0.0,
            "steps_to_R_star_per_seed": n_hit,
            "label_steps_charged": extra,
            "W_vs_ppo_pct": 100.0 * float(np.mean(finals)) / ppo_final if ppo_final else None,
        }
    base = _median_hit(out["methods"]["ppo"]["steps_to_R_star_per_seed"], 0)
    for m in out["methods"].values():
        mine = _median_hit(m["steps_to_R_star_per_seed"], m["label_steps_charged"])
        mine_free = _median_hit(m["steps_to_R_star_per_seed"], 0)
        m["X_vs_ppo"] = base / mine if (base and mine) else None
        m["X_vs_ppo_uncharged"] = base / mine_free if (base and mine_free) else None
    return out


def _median_hit(hits: list[float | None], extra: int) -> float | None:
    if not hits or any(h is None for h in hits):
        return None  # a censored seed makes the median undefined at this budget
    return float(np.median([h + extra for h in hits]))


def plot(results: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for res in results:
        sdir = TRAIN / res["scenario"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        for method in sorted(res.get("methods", {})):
            curves = [load_evals(r) for r in sorted(sdir.glob(f"{method}_s*"))
                      if (r / "done.json").exists()]
            n = min(len(c[0]) for c in curves)
            steps = curves[0][0][:n]
            r = np.stack([c[1][:n] for c in curves])
            mean, se = r.mean(0), r.std(0, ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0 * r[0]
            ax1.plot(steps, mean, label=method)
            ax1.fill_between(steps, mean - se, mean + se, alpha=0.2)
            ax2.plot(steps, np.stack([c[2][:n] for c in curves]).mean(0), label=method)
        for key, style in (("random", ":"), ("qwen", "--"), ("R_star", "-.")):
            if res.get(key) is not None:
                ax1.axhline(res[key], ls=style, c="gray", lw=1)
                ax1.text(0, res[key], f" {key}", va="bottom", fontsize=8, color="gray")
        ax1.set_ylabel("student-only eval return")
        ax1.set_title(res["scenario"])
        ax1.legend(fontsize=8)
        ax2.set_ylabel("teacher p")
        ax2.set_xlabel("env steps")
        fig.tight_layout()
        fig.savefig(out / f"curve_{res['scenario']}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-tag", default="Qwen3-VL-2B-Instruct")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    results = [analyze_scenario(s, args.teacher_tag) for s in sorted(TRAIN.iterdir()) if s.is_dir()]
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    plot(results, out)
    for res in results:
        print(f"\n== {res['scenario']}  random {res.get('random')}  qwen {res.get('qwen')}  "
              f"R* {res.get('R_star')}")
        for m, v in sorted(res.get("methods", {}).items()):
            print(f"  {m:10s} final {v['final_mean']:8.2f} +-{v['final_se']:.2f}  "
                  f"W {v['W_vs_ppo_pct']:.0f}%  X {v['X_vs_ppo']}  hits {v['steps_to_R_star_per_seed']}")


if __name__ == "__main__":
    main()
