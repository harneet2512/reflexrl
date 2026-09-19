"""A run paused at the session deadline must resume and finish without
duplicated or lost metric rows (Kaggle sessions are capped at 12 h)."""

import json
import time

import pytest

from reflexrl.rl.ppo import Paused, PPOConfig, train


@pytest.mark.env
def test_pause_then_resume_completes(tmp_path, monkeypatch):
    cfg = PPOConfig(scenario="dtc", total_steps=1024, n_envs=2, n_steps=64, device="cpu",
                    eval_every=256, eval_episodes=1, ckpt_every=10**9)
    monkeypatch.setenv("REFLEXRL_DEADLINE", str(time.time() - 1))  # already past
    with pytest.raises(Paused):
        train(cfg, tmp_path)
    assert (tmp_path / "resume.pt").exists()
    assert not (tmp_path / "done.json").exists()

    monkeypatch.setenv("REFLEXRL_DEADLINE", "inf")
    train(cfg, tmp_path)
    assert (tmp_path / "done.json").exists()
    assert not (tmp_path / "resume.pt").exists()

    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    train_steps = [r["step"] for r in rows if r["kind"] == "train"]
    assert train_steps == sorted(set(train_steps)), "no duplicated train rows"
    assert train_steps == list(range(128, 1024 + 1, 128))
    evals = [r["step"] for r in rows if r["kind"] == "eval" and not r.get("final")]
    assert evals == sorted(set(evals))
