"""Provenance stamped into every run: config, git commit, host, versions."""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_metadata(config: dict) -> dict:
    import torch
    return {
        "config": config,
        "git_commit": git_commit(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_unix": time.time(),
    }
