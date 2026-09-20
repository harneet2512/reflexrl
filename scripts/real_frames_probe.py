"""Does the sees/decides pipeline work on real gameplay footage?

Same perception question and same Jev decision layer as the Doom teacher, run on
real Valorant frames that ship with enemy bounding boxes, so the answer is scored
against ground truth rather than eyeballed.

    python scripts/real_frames_probe.py --root /kaggle/input/valorant-object-detection-dataset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.rl.dagger import WHERE  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402
from reflexrl.teacher.perception_jev import CLASSES, calibrate  # noqa: E402
from reflexrl.teacher.prompts import LETTERS  # noqa: E402

COPY = re.compile(r"( - Copy( \(\d+\))?)+", re.I)  # the dataset repeats frames as " - Copy (2)..."


def _is_num(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def dedup_key(p: Path) -> str:
    return COPY.sub("", p.stem).lower()


def class_ids(root: Path) -> tuple[list[int], list[int]]:
    """(enemy ids, other-player ids). Teammates are players too: a vision model
    cannot know which team a character belongs to, so frames whose only players
    are teammates are reported separately instead of being scored as 'nobody'."""
    for y in root.rglob("data.yaml"):
        names = re.findall(r"names:\s*\[(.*?)\]", y.read_text(), re.S)
        if names:
            items = [n.strip().strip("'\"") for n in names[0].split(",")]
            print("classes:", items, flush=True)
            enemy = [i for i, n in enumerate(items) if n.lower() in ("enemy", "enemy head")]
            mate = [i for i, n in enumerate(items) if "team" in n.lower()]
            return enemy, mate
    return [1], []


def oracle(label_file: Path, enemy_ids: list[int]) -> tuple[str, float]:
    """Nearest enemy (largest box) -> left/center/right/none, plus its box area."""
    best, best_area = None, 0.0
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or int(parts[0]) not in enemy_ids:
            continue
        cx, _, w, h = (float(x) for x in parts[1:5])
        if w * h > best_area:
            best, best_area = cx, w * h
    if best is None:
        return "none", 0.0
    return ("left" if best < 0.4 else "right" if best > 0.6 else "center"), best_area


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--perms", type=int, default=2)
    p.add_argument("--jev-table", default="experiments/configs/jev_table_dtc.json")
    p.add_argument("--width", type=int, default=320, help="resize width; 0 keeps full resolution")
    p.add_argument("--out", default="runs/real_frames")
    args = p.parse_args()
    root = Path(args.root)
    enemy_ids, mate_ids = class_ids(root)

    def is_yolo(f: Path) -> bool:
        try:
            first = next((ln for ln in f.read_text().splitlines() if ln.strip()), "")
        except Exception:
            return False
        parts = first.split()
        return len(parts) >= 5 and all(_is_num(x) for x in parts[:5])

    labels, images = {}, {}
    for f in root.rglob("*"):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            images.setdefault(dedup_key(f), f)
        elif f.suffix.lower() == ".txt" and not f.name.lower().startswith("readme") and is_yolo(f):
            labels.setdefault(dedup_key(f), f)
    pairs = [(images[k], labels[k]) for k in sorted(labels) if k in images]
    print(f"{len(images)} unique images, {len(labels)} label files, {len(pairs)} paired", flush=True)
    if not pairs:
        for d in sorted(root.glob("*")):
            print("  contents:", d, flush=True)
        raise SystemExit("no image/label pairs found under " + str(root))
    random.Random(0).shuffle(pairs)

    truth, chosen, has_mate = [], [], []
    for img, lab in pairs:
        cls, _ = oracle(lab, enemy_ids)
        mate, _ = oracle(lab, mate_ids) if mate_ids else ("none", 0.0)
        truth.append(cls)
        has_mate.append(mate != "none")
        chosen.append(img)
        if len(chosen) >= args.frames:
            break
    counts = Counter(truth)
    print("oracle classes:", dict(counts), flush=True)

    from PIL import Image

    from reflexrl.teacher.jev import JevClient
    from reflexrl.teacher.qwen import QwenTeacher
    teacher = QwenTeacher(args.model, dtype=torch.float32, load_4bit=True, device_map="auto")
    rng = np.random.default_rng(0)
    perms = [np.arange(4)] + [rng.permutation(4) for _ in range(args.perms - 1)]

    def load(f):
        im = Image.open(f).convert("RGB")
        if args.width:
            im = im.resize((args.width, round(im.height * args.width / im.width)))
        return [np.asarray(im)]

    frames = [load(f) for f in chosen]
    print(f"frame size fed to the model: {frames[0][0].shape}", flush=True)
    def question_for(perm) -> str:
        options = "\n".join(
            f"{LETTERS[k]}. {WHERE[CLASSES[i]].replace('not visible', 'no enemy is visible')}"
            for k, i in enumerate(perm))
        return ("This is a screenshot from the first-person shooter Valorant. Where is the "
                f"nearest enemy player?\n{options}\nAnswer with a single letter.")

    q = np.zeros((len(frames), 4))
    for perm in perms:
        out = [teacher.choice_probs(frames[i:i + 4], question_for(perm), 4)
               for i in range(0, len(frames), 4)]
        q[:, perm] += np.concatenate(out)
    q /= len(perms)
    blank = [np.zeros_like(frames[0][0])]
    prior = np.zeros(4)
    for perm in perms:
        prior[perm] += teacher.choice_probs([blank], question_for(perm), 4)[0]
    prior /= len(perms)
    q_cal = calibrate(q, prior)
    pred = [CLASSES[i] for i in q.argmax(1)]
    pred_cal = [CLASSES[i] for i in q_cal.argmax(1)]
    acc = float(np.mean([a == b for a, b in zip(pred, truth, strict=True)]))
    acc_cal = float(np.mean([a == b for a, b in zip(pred_cal, truth, strict=True)]))
    major = max(counts.values()) / len(truth)
    print(f"raw {acc:.3f} -> calibrated {acc_cal:.3f} (majority {major:.3f})", flush=True)
    # frames where the only visible character is a teammate are ambiguous for a
    # model that was only asked "where is the nearest enemy"
    keep = [i for i, (t, m) in enumerate(zip(truth, has_mate, strict=True)) if not (t == "none" and m)]
    acc_clean = float(np.mean([pred_cal[i] == truth[i] for i in keep])) if keep else float("nan")
    major_clean = (max(Counter(truth[i] for i in keep).values()) / len(keep)) if keep else float("nan")

    jev = JevClient(args.jev_table)
    J = jev.table_probs(CLASSES)
    actions = [jev.actions[int(i)] for i in (q @ J).argmax(1)]

    res = {"meta": run_metadata(vars(args)), "n_frames": len(truth),
           "oracle_counts": dict(counts),
           "perception_accuracy_calibrated": acc_cal,
           "perception_accuracy": acc, "majority_rate": major,
           "beats_majority_by": acc - major,
           "frames_with_teammate_only": int(sum(1 for t, m in zip(truth, has_mate, strict=True)
                                                if t == "none" and m)),
           "accuracy_excluding_teammate_only": acc_clean,
           "majority_excluding_teammate_only": major_clean,
           "confusion": {c: dict(Counter(p_ for t, p_ in zip(truth, pred, strict=True) if t == c))
                         for c in CLASSES},
           "example_decisions": [{"truth": t, "qwen": p_, "jev_action": a}
                                 for t, p_, a in list(zip(truth, pred, actions, strict=True))[:12]]}
    os.makedirs(args.out, exist_ok=True)
    res["frame_width"] = args.width or "full"
    Path(args.out, f"real_frames_w{args.width or 'full'}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
