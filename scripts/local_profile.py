"""Local GPU profile: equivalence gate + per-exit latency on the RTX 2060.

Free replacement for the Modal GPU profile (blocked by workspace billing).
Downloads the model to the local HF cache once, then reuses it.
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from reflexrl.models.qwen_vl import EXIT_LAYERS, QwenVLBackbone


def main():
    t0 = time.time()
    backbone = QwenVLBackbone(device="cuda")
    print(f"loaded in {time.time() - t0:.1f}s  dtype={backbone.dtype}")
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB "
          f"(reserved {torch.cuda.memory_reserved() / 1e9:.2f})")

    rng = np.random.default_rng(0)
    obs = rng.integers(0, 255, (2, 2, 180, 320, 3), dtype=np.uint8)

    # ---- equivalence gate -------------------------------------------------
    h_all = backbone.features_collect(obs)
    eq = {}
    for d in range(h_all.size(1)):
        eq[d] = float((backbone.features_to(obs, d) - h_all[:, d]).abs().max())
    print(f"truncated-vs-collected max|diff| per exit: {eq}")
    assert max(eq.values()) < 1e-3, "equivalence gate FAILED"

    # ---- per-exit latency --------------------------------------------------
    # warmup
    for d in range(3):
        backbone.features_to(obs, d)
    torch.cuda.synchronize()

    lat = {}
    reps = 10
    for d in range(3):
        t = time.perf_counter()
        for _ in range(reps):
            backbone.features_to(obs, d)
        torch.cuda.synchronize()
        lat[d] = (time.perf_counter() - t) / reps * 1000
    for d, ms in lat.items():
        print(f"exit {d} (layer {EXIT_LAYERS[d]}): {ms:.1f} ms/decision  "
              f"({1000 / ms:.1f} decisions/s)")

    from reflexrl.eval.compute import default_model
    cm = default_model()
    for row in cm.table():
        print(f"exit {row['exit']}: {row['gflops']:.0f} GFLOP  "
              f"frac_of_deep={row['frac_of_deep']:.3f}")

    ms = {d: lat[d] for d in lat}
    print(f"\nspeedup reflex/deep: {ms[2] / ms[0]:.2f}x  "
          f"fast/deep: {ms[2] / ms[1]:.2f}x")


if __name__ == "__main__":
    main()
