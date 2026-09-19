"""Root-cause diagnosis for the Qwen3-VL-2B teacher failure.

H1 pipeline      synthetic control: red disc left/center/right on grey
H2 readout       unrestricted top-10 next tokens + free-text generation
H3 precision     fp16 vs fp32 on the same inputs
H4 two frames    1-frame vs 2-frame prompt
H5 domain gap    free-text descriptions of Doom frames with a known target
H6 batching      same frame alone vs inside a batch of 8

    python scripts/diagnose_teacher.py --out runs/diagnose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.debias_probe import collect  # noqa: E402

POS_Q = ("Where is the {obj} in the image?\nA. on the left side\nB. in the center\n"
         "C. on the right side\nD. there is no {obj}\nAnswer with a single letter.")


def synthetic(pos: str | None, rng) -> np.ndarray:
    img = np.full((180, 320, 3), 128, np.uint8)
    if pos is not None:
        import cv2
        cx = {"left": 55, "center": 160, "right": 265}[pos] + int(rng.integers(-15, 16))
        cv2.circle(img, (cx, 90 + int(rng.integers(-30, 31))), 28, (220, 20, 20), -1)
    return img


def prompt_for(teacher, n_images: int, question: str) -> str:
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": question}]
    return teacher.processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def next_token_report(teacher, images: list[np.ndarray], question: str, k: int = 10) -> dict:
    from PIL import Image
    batch = teacher.processor(text=[prompt_for(teacher, len(images), question)],
                              images=[Image.fromarray(i) for i in images], return_tensors="pt")
    batch = {n: v.to(teacher.device) if hasattr(v, "to") else v for n, v in batch.items()}
    logits = teacher.model(**batch).logits[0, -1].float()
    probs = torch.softmax(logits, -1)
    top = torch.topk(probs, k)
    tok = teacher.processor.tokenizer
    letters = {c: float(probs[tok.encode(c, add_special_tokens=False)[0]]) for c in "ABCD"}
    return {"top": [(tok.decode([int(i)]), round(float(p), 4)) for p, i in zip(top.values, top.indices, strict=True)],
            "letter_mass": round(sum(letters.values()), 4), "letters": letters,
            "prompt_tail": repr(prompt_for(teacher, len(images), question)[-80:])}


@torch.no_grad()
def generate(teacher, images: list[np.ndarray], question: str, max_new: int = 48) -> str:
    from PIL import Image
    batch = teacher.processor(text=[prompt_for(teacher, len(images), question)],
                              images=[Image.fromarray(i) for i in images], return_tensors="pt")
    batch = {n: v.to(teacher.device) if hasattr(v, "to") else v for n, v in batch.items()}
    out = teacher.model.generate(**batch, max_new_tokens=max_new, do_sample=False)
    return teacher.processor.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run(dtype_name: str, out: dict) -> None:
    from reflexrl.teacher.qwen import QwenTeacher
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    teacher = QwenTeacher(dtype=dtype)
    rng = np.random.default_rng(0)
    res = {}

    # H1: synthetic control, 8 per class
    correct, rows = 0, []
    for pos in ("left", "center", "right", None):
        for _ in range(8):
            rep = next_token_report(teacher, [synthetic(pos, rng)], POS_Q.format(obj="red circle"))
            pred = max(rep["letters"], key=rep["letters"].get)
            truth = {"left": "A", "center": "B", "right": "C", None: "D"}[pos]
            correct += pred == truth
            rows.append((pos, pred, rep["letter_mass"]))
    res["H1_synthetic_accuracy"] = correct / 32
    res["H1_rows"] = rows
    res["H1_example_generation"] = generate(teacher, [synthetic("left", rng)],
                                            "Describe this image in one sentence.")

    # H2/H4/H5 on real Doom frames with known targets
    doom = {}
    for scen, obj in (("defend_the_center", "monster"), ("health_gathering", "medkit")):
        frames, classes, _ = collect(scen, 60, seed=4242)
        picks = [i for i, c in enumerate(classes) if c != "none"][:6] + \
                [i for i, c in enumerate(classes) if c == "none"][:2]
        items = []
        for i in picks:
            fr = frames[i]
            items.append({
                "oracle": classes[i],
                "describe_1frame": generate(teacher, fr[-1:], "Describe this image in one sentence."),
                "is_there": generate(teacher, fr[-1:], f"Is there a {obj} in this image? Answer yes or no, then say where."),
                "pos_1frame": next_token_report(teacher, fr[-1:], POS_Q.format(obj=obj)),
                "pos_2frame": next_token_report(teacher, fr, POS_Q.format(obj=obj)),
            })
        doom[scen] = items
    res["doom"] = doom

    # H6: batch consistency on one Doom frame via the production scorer
    from reflexrl.env.scenarios import get_scenario
    spec = get_scenario("dtc")
    fr = collect("defend_the_center", 4, seed=9)[0][0]
    alone = teacher.action_probs([fr], spec)[0]
    inbatch = teacher.action_probs([fr] * 8, spec)[0]
    res["H6_batch_max_abs_diff"] = float(np.abs(alone - inbatch).max())
    out[dtype_name] = res
    del teacher
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/diagnose")
    p.add_argument("--dtypes", nargs="+", default=["fp16", "fp32"])
    args = p.parse_args()
    out: dict = {}
    for dt in args.dtypes:
        run(dt, out)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "diagnose.json").write_text(json.dumps(out, indent=2, default=str))
        print(dt, "H1 synthetic acc", out[dt]["H1_synthetic_accuracy"], flush=True)


if __name__ == "__main__":
    main()
