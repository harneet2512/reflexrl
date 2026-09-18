"""Canonical run-directory layout shared by the queue and the analysis."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reflexrl.env.scenarios import get_scenario  # noqa: E402

TRAIN_ROOT = REPO / "runs" / "train"


def run_dir(r: dict) -> Path:
    return TRAIN_ROOT / get_scenario(r["scenario"]).name / f"{r.get('tag') or r['method']}_s{r['seed']}"
