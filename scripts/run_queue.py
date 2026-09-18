"""Resumable sequential run queue (one training process at a time).

Reads a plan (JSON list of {"method", "scenario", "steps", "seed"}), skips
runs whose done.json exists, and runs the rest one by one. Safe to kill and
restart: at most the in-flight run is repeated.

    python scripts/run_queue.py experiments/configs/plan_main.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from reflexrl_paths import REPO, run_dir  # noqa: F401  (local helper below)


def main() -> None:
    plan = json.loads(Path(sys.argv[1]).read_text())
    extra = sys.argv[2:]
    for i, r in enumerate(plan):
        wd = run_dir(r)
        if (wd / "done.json").exists():
            print(f"[{i + 1}/{len(plan)}] skip {wd} (done)", flush=True)
            continue
        cmd = [sys.executable, "-u", str(REPO / "scripts" / "train.py"),
               "--method", r["method"], "--scenario", r["scenario"],
               "--steps", str(r["steps"]), "--seed", str(r["seed"]), *extra]
        for key in ("init_ckpt", "tag"):
            if r.get(key):
                cmd += [f"--{key.replace('_', '-')}", str(r[key])]
        if (wd / "metrics.jsonl").exists():
            (wd / "metrics.jsonl").unlink()  # restart the interrupted run cleanly
        print(f"[{i + 1}/{len(plan)}] start {wd}", flush=True)
        t0 = time.time()
        wd.mkdir(parents=True, exist_ok=True)
        with (wd / "stdout.log").open("w") as fh:
            rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"[{i + 1}/{len(plan)}] {status} {wd} in {(time.time() - t0) / 60:.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
