"""Demo video: 1280x720 at 35 fps (Doom's native tic rate, so gameplay is real time).

Segments: title -> split-screen Qwen vs ReflexRL (both latency-aware, same
seed) -> "HOW?" -> training (student return rising while teacher dependence
falls, from a real metrics.jsonl) -> ReflexRL montage -> headline card.
Every number drawn comes from a results file passed in; nothing is typed in.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 35
BG = (18, 18, 22)
FG = (235, 235, 235)
DIM = (150, 150, 160)
ACCENT = (80, 200, 255)  # BGR amber-ish for ReflexRL
QWEN = (180, 120, 255)  # BGR pink-ish for Qwen
FONT = cv2.FONT_HERSHEY_DUPLEX


def canvas() -> np.ndarray:
    return np.full((H, W, 3), BG, np.uint8)


def text(img, s, xy, scale=1.0, color=FG, thick=1, center=False):
    if center:
        (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
        xy = (xy[0] - tw // 2, xy[1])
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def card(lines: list[tuple[str, float, tuple]], seconds: float) -> list[np.ndarray]:
    img = canvas()
    total = sum(int(60 * sc) + 20 for _, sc, _ in lines)
    y = (H - total) // 2 + 40
    for s, sc, col in lines:
        text(img, s, (W // 2, y), sc, col, 2 if sc >= 1.2 else 1, center=True)
        y += int(60 * sc) + 20
    return [img] * int(seconds * FPS)


def _panel(frame_rgb: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    if frame_rgb is None:
        return np.zeros((size[1], size[0], 3), np.uint8)
    return cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), size,
                      interpolation=cv2.INTER_NEAREST)


def split_screen(qwen_tics: list, reflex_tics: list, seconds: float,
                 qwen_stats: dict, reflex_stats: dict) -> list[np.ndarray]:
    """Both tic lists are 35 fps game timelines from realtime_episode(record=True)."""
    n = int(seconds * FPS)
    pw, ph = 616, 347
    frames, q_calls, last_q, last_r = [], 0, None, None
    for t in range(n):
        img = canvas()
        q = qwen_tics[min(t, len(qwen_tics) - 1)]
        r = reflex_tics[min(t, len(reflex_tics) - 1)]
        if t < len(qwen_tics) and q.fresh:
            q_calls += 1
        last_q = q.frame if q.frame is not None else last_q
        last_r = r.frame if r.frame is not None else last_r
        for x0, fr, name, col in ((16, last_q, "QWEN3-VL-8B + JEV", QWEN),
                                  (648, last_r, "REFLEXRL", ACCENT)):
            text(img, name, (x0, 50), 1.1, col, 2)
            img[70:70 + ph, x0:x0 + pw] = _panel(fr, (pw, ph))
        rows = (
            ("DECISION LATENCY", f"{qwen_stats['ms_mean']:.0f} ms",
             f"{reflex_stats['ms_mean']:.1f} ms"),
            ("MAX ACTION RATE", f"{qwen_stats['max_actions_per_s']:.1f} Hz",
             f"{reflex_stats['max_actions_per_s']:.0f} Hz"),
            ("MODEL CALLS", f"{q_calls}", "0"),
            ("SCORE", f"{q.score:.0f}", f"{r.score:.0f}"),
        )
        y = 470
        for label, qv, rv in rows:
            text(img, label, (16, y), 0.6, DIM)
            text(img, qv, (16, y + 34), 1.0, QWEN, 2)
            text(img, label, (648, y), 0.6, DIM)
            text(img, rv, (648, y + 34), 1.0, ACCENT, 2)
            y += 62 if label != "QWEN CALLS" else 62
        frames.append(img)
    return frames


def training_segment(metrics_path: Path, seconds: float, title: str) -> list[np.ndarray]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = [json.loads(line) for line in Path(metrics_path).read_text().splitlines()]
    ev = [e for e in ev if e.get("kind") == "eval" and not e.get("final")]
    steps = np.array([e["step"] for e in ev]) / 1e6
    ret = np.array([e["return_mean"] for e in ev])
    pt = np.array([e["p_teacher"] for e in ev])
    n = int(seconds * FPS)
    frames = []
    for t in range(n):
        k = max(2, int(len(steps) * (t + 1) / n))
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#16161a")
        ax = fig.add_axes([0.08, 0.36, 0.6, 0.52], facecolor="#16161a")
        ax.plot(steps[:k], ret[:k], color="#ffc850", lw=3)
        ax.set_xlim(0, steps[-1])
        ax.set_ylim(min(ret.min(), 0) * 1.05, ret.max() * 1.1)
        ax.set_xlabel("environment steps (millions)", color="#aaa")
        ax.set_ylabel("student score (no teacher)", color="#aaa")
        ax.tick_params(colors="#aaa")
        for s in ax.spines.values():
            s.set_color("#444")
        fig.text(0.08, 0.92, title, color="#eee", fontsize=22)
        guided = bool(pt.max() > 0)
        if not guided:
            fig.text(0.74, 0.84, "NO TEACHER", color="#ff78b4", fontsize=18)
            fig.text(0.74, 0.78, "pixels + reward only", color="#aaa", fontsize=13)
        for i, level in enumerate((1.0, 0.5, 0.25, 0.1, 0.0) if guided else ()):
            y = 0.74 - i * 0.075
            cur = pt[k - 1]
            fig.text(0.74, y, f"{int(level * 100):>3d}%", color="#aaa", fontsize=14, family="monospace")
            bar = fig.add_axes([0.80, y - 0.005, 0.16, 0.035])
            bar.barh([0], [level], color="#ff78b4" if abs(cur - level) < 1e-6 else "#44343c")
            bar.set_xlim(0, 1)
            bar.axis("off")
        if guided:
            fig.text(0.74, 0.30, f"now: {pt[k - 1] * 100:.0f}% of actions from Qwen", color="#eee", fontsize=13)
        fig.text(0.08, 0.22, f"step {steps[k - 1]:.2f}M   score {ret[k - 1]:.1f}", color="#ffc850", fontsize=16)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        frames.append(cv2.resize(img, (W, H)))
    return frames


def montage(clips: list[tuple[str, list, dict]], seconds_each: float) -> list[np.ndarray]:
    frames = []
    for name, tics, stats in clips:
        last = None
        for t in range(int(seconds_each * FPS)):
            tic = tics[min(t, len(tics) - 1)]
            last = tic.frame if tic.frame is not None else last
            img = _panel(last, (W, H))
            cv2.rectangle(img, (0, H - 70), (W, H), BG, -1)
            text(img, f"REFLEXRL  |  {name}", (24, H - 28), 0.9, ACCENT, 2)
            text(img, f"score {tic.score:.0f}   {stats['ms_mean']:.1f} ms/decision   0 Qwen calls",
                 (W - 620, H - 28), 0.75, FG)
            frames.append(img)
    return frames


def write_video(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
