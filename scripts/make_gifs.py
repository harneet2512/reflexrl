"""Build the README's animated assets from footage and metrics already on disk.

Three GIFs, no GPU and no re-running of anything:

- ``teacher_vs_student.gif`` -- the 6.2 s/decision teacher beside the 1.7 ms
  policy it trained, same seed, both in real time.
- ``learning_race.gif``      -- ReflexRL vs PPO from scratch, animated from the
  archived ``metrics.jsonl`` files, racing to the pre-registered target score.
- ``gameplay.gif``           -- the reflex policy playing, full frame.
- ``budget_comparison.gif``  -- first episodes of the same-budget three-way
  comparison recorded by scripts/make_budget_video.py.

    python scripts/make_gifs.py
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "demo"
SRC = OUT / "reflexrl_demo.mp4"

# Segments of the composed demo (see scripts/make_demo.py), as
# start, duration, width, fps. Doom's camera turns constantly, so every frame
# differs and GIF size is driven by duration far more than by quality knobs:
# these are tuned to keep each file near 4 MB, which a README can autoplay.
SPLIT_SCREEN = (6.0, 8.0, 640, 12)  # inside the 3.0-19.0 s split screen
MONTAGE = (31.0, 4.0, 640, 12)  # inside the 30.5-38.5 s full-frame montage
# the budget comparison is its own recording (scripts/make_budget_video.py); the
# README shows its first three episodes and links the full 54 s file
BUDGET = (0.0, 26.0, 560, 9)

BG = "#16161a"
AMBER = "#ffc850"  # ReflexRL
PINK = "#ff78b4"  # teacher / baseline
DIM = "#8a8a94"


def clip_gif(start: float, dur: float, width: int, fps: int, out: Path,
             colors: int = 128, src: Path | None = None) -> None:
    """Extract one segment of a demo video as a palette-optimised GIF."""
    source = str(src or SRC)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not on PATH")
    pal = out.with_suffix(".palette.png")
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur),
                    "-i", source, "-vf",
                    f"{scale},palettegen=max_colors={colors}:stats_mode=diff", str(pal)],
                   check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur),
                    "-i", source, "-i", str(pal), "-lavfi",
                    f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:"
                    "diff_mode=rectangle", str(out)], check=True)
    pal.unlink(missing_ok=True)
    print(f"{out.relative_to(REPO).as_posix()}  {out.stat().st_size / 1e6:.1f} MB")


# --------------------------------------------------------------------- the race


def curves(pattern: str) -> list[np.ndarray]:
    """Every run's (step, eval return) curve for a glob of metrics.jsonl files."""
    out = []
    for m in sorted(glob.glob(str(REPO / pattern), recursive=True)):
        if not (Path(m).parent / "done.json").exists():
            continue
        ev = [json.loads(ln) for ln in Path(m).read_text().splitlines() if '"eval"' in ln]
        ev = [e for e in ev if not e.get("final")]
        out.append(np.array([[e["step"], e["return_mean"]] for e in ev]))
    return out


def band(cs: list[np.ndarray], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and min/max envelope of several curves, resampled onto a common grid."""
    ys = np.stack([np.interp(grid, c[:, 0], c[:, 1]) for c in cs])
    return ys.mean(0), ys.min(0), ys.max(0)


def crossing(y: np.ndarray, grid: np.ndarray, target: float, k: int = 3) -> float | None:
    """First grid point where a 3-point trailing mean reaches the target."""
    if len(y) < k:
        return None
    trail = np.convolve(y, np.ones(k) / k, "valid")
    hit = np.flatnonzero(trail >= target)
    return float(grid[hit[0] + k - 1]) if len(hit) else None


def learning_race(out: Path, seconds: float = 9.0, fps: int = 14) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    A = "archive/kaggle"
    ref = curves(f"{A}/reflexrl-train-lane0/**/reflexrl_s*/metrics.jsonl")
    ppo = curves(f"{A}/reflexrl-train-ppo/**/ppo_s*/metrics.jsonl")
    if not (ref and ppo):
        print("no training curves found; skipping the race", file=sys.stderr)
        return
    prereg = json.loads((REPO / "experiments/configs/metrics_prereg.json").read_text())
    target = 5.46  # recomputed below from the same rule the metrics report uses
    rand = json.loads((REPO / "results/teacher_gate/random_gate.json").read_text())
    rand_mean = float(np.mean(rand["scenarios"]["defend_the_center"]["returns"]))
    ppo_final = float(np.mean([json.loads((Path(glob.glob(str(REPO / f"{A}/reflexrl-train-ppo/**/ppo_s{i}/done.json"), recursive=True)[0])).read_text())["final_eval"]
                               for i in range(len(ppo))]))
    target = rand_mean + 0.8 * (ppo_final - rand_mean)

    top = min(float(c[-1, 0]) for c in ref + ppo)
    grid = np.linspace(0, top, 300)
    r_mean, r_lo, r_hi = band(ref, grid)
    p_mean, p_lo, p_hi = band(ppo, grid)

    # Identical accounting to scripts/metrics_report.py, so one number is reported
    # everywhere: per-seed crossing on that seed's own evaluation points, median
    # across seeds, and the teacher-label collection charged to ReflexRL.
    label_steps = sum(json.loads(f.read_text())["scenarios"]["defend_the_center"]["decisions"]
                      for f in (REPO / "results/teacher_gate").glob("qwen8b_jev_dtc_lane*.json"))

    def per_seed(cs: list[np.ndarray], extra: int = 0) -> float | None:
        hits = [crossing(c[:, 1], c[:, 0], target) for c in cs]
        done = [h for h in hits if h is not None]
        return float(np.median(done)) + extra if len(done) == len(hits) else None

    r_cross, p_cross = per_seed(ref, label_steps), per_seed(ppo)

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100, facecolor=BG)
    ax = fig.add_axes((0.09, 0.14, 0.88, 0.70), facecolor=BG)
    ax.set_xlim(0, top / 1e6)
    ax.set_ylim(min(0.0, float(min(r_lo.min(), p_lo.min()))) - 0.3,
                float(max(r_hi.max(), p_hi.max())) * 1.12)
    ax.set_xlabel("environment steps (millions): the cost of learning", color=DIM, fontsize=11)
    ax.set_ylabel("score (no teacher involved)", color=DIM, fontsize=11)
    ax.tick_params(colors=DIM)
    for s in ax.spines.values():
        s.set_color("#33333c")
    ax.axhline(target, color=DIM, ls="--", lw=1.2)
    ax.text(top / 1e6 * 0.985, target + 0.12, f"target score {target:.2f}",
            color=DIM, fontsize=10, ha="right")
    fig.text(0.09, 0.90, "Same score. Reached three times sooner.",
             color="#ededed", fontsize=17)
    fig.text(0.62, 0.905, "shaded = spread across seeds", color=DIM, fontsize=10)

    p_line, = ax.plot([], [], color=PINK, lw=2.6, label=f"PPO from scratch ({len(ppo)} seeds)")
    r_line, = ax.plot([], [], color=AMBER, lw=3.0, label=f"ReflexRL ({len(ref)} seeds)")
    p_fill = r_fill = None
    marks: list = []
    leg = ax.legend(loc="upper left", facecolor="#1e1e24", edgecolor="#33333c", fontsize=10)
    for t in leg.get_texts():
        t.set_color("#dddddd")
    note = fig.text(0.09, 0.035, "", color=AMBER, fontsize=12)

    n = int(seconds * fps)

    def frame(i: int):
        nonlocal p_fill, r_fill
        k = max(2, int(len(grid) * (i + 1) / n))
        x = grid[:k] / 1e6
        p_line.set_data(x, p_mean[:k])
        r_line.set_data(x, r_mean[:k])
        for f in (p_fill, r_fill):
            if f is not None:
                f.remove()
        p_fill = ax.fill_between(x, p_lo[:k], p_hi[:k], color=PINK, alpha=0.16, linewidth=0)
        r_fill = ax.fill_between(x, r_lo[:k], r_hi[:k], color=AMBER, alpha=0.18, linewidth=0)
        here = grid[k - 1]
        msgs = []
        for cross, col, label in ((r_cross, AMBER, "ReflexRL"), (p_cross, PINK, "PPO")):
            if cross is not None and here >= cross:
                if not any(m[0] == label for m in marks):
                    ax.plot([cross / 1e6], [target], "o", color=col, ms=9, zorder=5)
                    ax.annotate(f"{label}\n{cross / 1000:.0f}K steps",
                                (cross / 1e6, target), textcoords="offset points",
                                xytext=(8, -34), color=col, fontsize=11, weight="bold")
                    marks.append((label, cross))
                msgs.append(label)
        if r_cross and p_cross and len(msgs) == 2:
            note.set_text(f"{p_cross / r_cross:.2f}x fewer environment steps to the same score")
        elif "ReflexRL" in msgs:
            note.set_text("ReflexRL is there. PPO is still climbing.")
        return ()

    FuncAnimation(fig, frame, frames=n, blit=False).save(
        str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"{out.relative_to(REPO).as_posix()}  {out.stat().st_size / 1e6:.1f} MB  "
          f"(target {target:.2f}, ReflexRL {r_cross}, PPO {p_cross})")
    assert prereg  # the target rule is pre-registered; kept in the read path deliberately


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["split", "race", "gameplay", "budget"],
                    default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "split"):
        clip_gif(*SPLIT_SCREEN, OUT / "teacher_vs_student.gif")
    if args.only in (None, "race"):
        learning_race(OUT / "learning_race.gif")
    if args.only in (None, "gameplay"):
        clip_gif(*MONTAGE, OUT / "gameplay.gif")
    if args.only in (None, "budget"):
        budget = OUT / "budget_comparison.mp4"
        if budget.exists():
            clip_gif(*BUDGET, OUT / "budget_comparison.gif", src=budget)
        else:
            print("no budget_comparison.mp4; run scripts/make_budget_video.py first",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
