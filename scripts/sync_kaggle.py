"""Pull every Kaggle job's output into the repo so nothing lives only in the cloud.

Small files (metrics, results, configs, logs) land under archive/kaggle/<job>/ and
are committed; checkpoints go to the backup directory, which stays out of git.

    python scripts/sync_kaggle.py [--jobs reflexrl-demo ...] [--backup D:/reflexrl_backup]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO.parent / ".kaggle-venv" / "Scripts" / "kaggle.exe"
JOBS = [
    "reflexrl-qwen-gate", "reflexrl-gate-8b", "reflexrl-gate-pjev", "reflexrl-debias-probe",
    "reflexrl-size-check", "reflexrl-diagnose", "reflexrl-hg-probe",
    "reflexrl-teachers-pjev", "reflexrl-perception-teacher",
    "reflexrl-train-ppo", "reflexrl-train-lane0", "reflexrl-train-lane1",
    "reflexrl-pilot-pjev", "reflexrl-live-teacher", "reflexrl-ablation",
    "reflexrl-heldout-a", "reflexrl-heldout-b", "reflexrl-demo", "reflexrl-final-eval",
    "reflexrl-bench-cpu", "reflexrl-bench-gpu",
]
BIG = {".pt", ".mp4", ".npz"}


def pull(job: str, tmp: Path) -> bool:
    dest = tmp / job
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(KAGGLE), "kernels", "output", f"harneetb/{job}", "-p", str(dest)],
                       capture_output=True, text=True)
    ok = any(dest.rglob("*"))
    print(f"{job:<28} {'ok' if ok else 'no output'} {r.stderr.strip()[:60]}", flush=True)
    return ok


def split_files(job_dir: Path, small_root: Path, backup_root: Path) -> dict:
    counts = {"small": 0, "big": 0, "bytes_big": 0}
    for f in job_dir.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(job_dir)
        if f.suffix in BIG:
            out = backup_root / job_dir.name / rel
            counts["big"] += 1
            counts["bytes_big"] += f.stat().st_size
        else:
            out = small_root / job_dir.name / rel
            counts["small"] += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", nargs="*", default=JOBS)
    p.add_argument("--backup", default="D:/reflexrl_backup")
    p.add_argument("--tmp", default="D:/tmp/reflexrl_sync")
    args = p.parse_args()
    tmp, small, backup = Path(args.tmp), REPO / "archive" / "kaggle", Path(args.backup)
    if tmp.exists():
        shutil.rmtree(tmp)
    summary = {}
    for job in args.jobs:
        if pull(job, tmp):
            summary[job] = split_files(tmp / job, small, backup)
    (small / "sync_summary.json").write_text(json.dumps(summary, indent=2))
    mb = sum(v["bytes_big"] for v in summary.values()) / 2**20
    print(f"\n{len(summary)} jobs archived | small files in {small} (git) | "
          f"{mb:.0f} MB of checkpoints/video in {backup}")
    if not summary:
        sys.exit("nothing pulled: check the Kaggle CLI and job names")


if __name__ == "__main__":
    main()
