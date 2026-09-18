"""CPU-only Modal app for ReflexRL: test gates + budget ledger status.

Separated from modal_app.py because Modal validates GPU functions at
deploy time — a mixed app can't deploy at all when the workspace lacks a
payment method for GPUs. Same app name family and same volumes, so the
ledger is shared across both apps.

Usage:
    modal run training/modal_cpu.py::run_tests
    modal run training/modal_cpu.py::budget_status
"""

from __future__ import annotations

import json

import modal

APP_NAME = "reflexrl-cpu"
RUNS = "/runs"
HF = "/hf"

_BASE_DEPS = [
    "vizdoom==1.3.0", "gymnasium>=1.0", "numpy>=1.26",
    "pytest>=8.0", "opencv-python-headless>=4.8",
]


def _locals(image):
    """Local package mounts — must be the last image build steps."""
    return (
        image
        .add_local_dir("reflexrl", remote_path="/root/reflexrl_pkg/reflexrl")
        .add_local_dir("tests", remote_path="/root/reflexrl_pkg/tests")
        .add_local_dir("scripts", remote_path="/root/reflexrl_pkg/scripts")
        .add_local_file("pyproject.toml", remote_path="/root/reflexrl_pkg/pyproject.toml")
    )


cpu_image = _locals(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_BASE_DEPS)
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .env({"PYTHONPATH": "/root/reflexrl_pkg"})
)

# Real-model CPU image: same + transformers. Qwen3-VL-2B in fp32 is ~9GB —
# slow but correct, and correctness checks (exit equivalence, real-feature
# PPO smoke) cost cents here instead of GPU dollars.
model_image = _locals(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_BASE_DEPS)
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "transformers>=4.57", "accelerate>=1.0", "sentencepiece", "protobuf",
        "LevDoom>=1.0", "pillow>=10.0",
    )
    .env({"PYTHONPATH": "/root/reflexrl_pkg", "HF_HOME": HF,
          "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("reflexrl-runs", create_if_missing=True)
hf_volume = modal.Volume.from_name("reflexrl-hf", create_if_missing=True)

MAX_CONTAINERS = 4


@app.function(image=cpu_image, volumes={RUNS: volume}, timeout=60 * 60,
              cpu=4.0, max_containers=MAX_CONTAINERS)
def budget_status() -> dict:
    """Show the spend ledger: recorded spend, remaining headroom, events."""
    from reflexrl.cloud_budget import CAP_USD, _read, remaining_usd, spent_usd

    volume.reload()
    led = _read()
    out = {
        "cap_usd": CAP_USD,
        "spent_usd": spent_usd(),
        "remaining_usd": remaining_usd(),
        "n_events": len(led["events"]),
        "events": led["events"][-20:],
    }
    print(json.dumps(out, indent=2))
    return out


@app.function(image=cpu_image, volumes={RUNS: volume}, timeout=60 * 60,
              cpu=4.0, max_containers=MAX_CONTAINERS)
def run_tests() -> str:
    """CI gate: all gates run in the deployment image, CPU-only."""
    import subprocess
    import sys
    import time

    from reflexrl.cloud_budget import RATES_PER_HR, record_spend, spend_gate

    volume.reload()
    spend_gate(0.5, "run_tests")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "/root/reflexrl_pkg/tests"],
        cwd="/root/reflexrl_pkg", capture_output=True, text=True,
    )
    print(proc.stdout[-8000:])
    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR["cpu"] * 4, "run_tests")
    volume.commit()
    print(f"[budget] run_tests cost ~${usd:.3f}")
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise RuntimeError("gates failed")
    return proc.stdout[-2000:]


@app.function(image=model_image, volumes={RUNS: volume, HF: hf_volume},
              timeout=60 * 60, cpu=8.0, memory=16384,
              max_containers=MAX_CONTAINERS)
def probe_hidden() -> dict:
    """Diagnose hidden_states tuple layout: len + per-index alignment.

    Answers exactly which index in a truncated run corresponds to 'after
    layer n', and whether a post-norm entry is appended.
    """
    import time

    import numpy as np

    from reflexrl.cloud_budget import RATES_PER_HR, record_spend, spend_gate
    from reflexrl.models.qwen_vl import QwenVLBackbone

    volume.reload()
    spend_gate(0.1, "probe_hidden")
    t0 = time.time()
    backbone = QwenVLBackbone(device="cpu")
    obs = np.random.default_rng(0).integers(0, 255, (2, 2, 180, 320, 3), dtype=np.uint8)
    batch = backbone.encode(obs)

    full = backbone._inner_forward(batch)
    n_full = len(full)
    info = {"n_full": n_full, "full_norms": [float(h[:, -1].norm()) for h in full]}

    with backbone._truncated_layers(9):
        tr = backbone._inner_forward(batch)
    info["n_trunc9"] = len(tr)
    # alignment: for each index in truncated run, closest match in full run
    al = []
    for i, h in enumerate(tr):
        diffs = [float((h - full[j]).abs().max()) for j in range(n_full)]
        al.append((i, int(np.argmin(diffs)), min(diffs)))
    info["trunc9_alignment"] = al
    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR["cpu"] * 8, "probe_hidden")
    volume.commit()
    info["recorded_usd"] = usd
    import json

    return json.dumps(info)


@app.function(image=cpu_image, volumes={RUNS: volume},
              schedule=modal.Period(minutes=5),
              max_containers=1)
def budget_police() -> None:
    """Wind-down enforcer. Deployed (`modal deploy`), fires every 5 min.

    If the ledger reaches the cap, list running apps and AppStop every
    reflexrl app — including itself last. Best-effort: in-container client
    may lack workspace scope, in which case it logs and the in-run
    watchdogs + spend gates remain the enforcement.
    """
    from reflexrl.cloud_budget import CAP_USD, spent_usd

    volume.reload()
    usd = spent_usd()
    print(f"[police] ledger spend ${usd:.4f} / cap ${CAP_USD:.2f}")
    if usd < CAP_USD:
        return

    import asyncio

    import modal as _m
    from modal_proto import api_pb2

    async def _kill_all():
        client = await _m.client._Client.from_env()
        apps = await client.stub.AppList(api_pb2.AppListRequest())
        killed = []
        for a in getattr(apps, "apps", []):
            name = getattr(a, "name", "") or getattr(a, "description", "")
            if "reflexrl" not in name:
                continue
            try:
                await client.stub.AppStop(api_pb2.AppStopRequest(
                    app_id=a.app_id, source=api_pb2.APP_STOP_SOURCE_CLI))
                killed.append((a.app_id, name))
            except Exception as e:
                print(f"[police] AppStop {a.app_id} failed: {e}")
        return killed

    try:
        killed = asyncio.run(_kill_all())
        print(f"[police] cap reached — stopped: {killed}")
    except Exception as e:
        print(f"[police] kill sweep failed ({e}); in-run watchdogs still enforce")


@app.function(image=cpu_image, volumes={RUNS: volume},
              timeout=60 * 60 * 6, cpu=8.0, memory=8192,
              max_containers=MAX_CONTAINERS)
def train_cnn(tag: str = "cnn", total_decisions: int = 150_000, seed: int = 0,
              n_envs: int = 8, rollout_steps: int = 256) -> str:
    """CNN specialist baseline — the one matrix row that trains fine on CPU.

    Small convnet, forward ~ms; ViZDoom stepping dominates. Same trainer,
    same decision budget as the Qwen pilot for a fair comparison.
    """
    import pathlib
    import time

    from reflexrl.baselines.cnn_policy import CNNSpecialist
    from reflexrl.cloud_budget import (
        RATES_PER_HR,
        BudgetWatchdog,
        record_spend,
        spend_gate,
    )
    from reflexrl.rl.trainer import TrainConfig
    from reflexrl.rl.trainer import train as _train

    est = 1.5
    volume.reload()
    spend_gate(est, f"train_cnn:{tag}")
    t0 = time.time()
    watchdog = BudgetWatchdog(RATES_PER_HR["cpu"] * 8, f"train_cnn:{tag}",
                            reload=volume.reload)
    agent = CNNSpecialist(n_actions=6)
    cfg = TrainConfig(
        total_decisions=total_decisions, seed=seed, n_envs=n_envs,
        rollout_steps=rollout_steps, device="cpu", tag=tag, lam=0.0,
        ckpt_at=tuple(x for x in (10_000, 50_000, 150_000, 300_000)
                      if x <= total_decisions),
    )
    wd = pathlib.Path(f"{RUNS}/{tag}_s{seed}")
    history = _train(agent, cfg, wd, budget_check=watchdog.check)
    import json

    summary = {"tag": tag, "seed": seed, "workdir": str(wd),
               "ep_returns_tail": history["ep_returns"][-5:],
               "mean_depth_tail": history["mean_depth"][-5:]}
    pathlib.Path(wd, "summary.json").write_text(json.dumps(summary, indent=2))
    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR["cpu"] * 8, f"train_cnn:{tag}")
    summary["recorded_usd"] = usd
    volume.commit()
    return json.dumps(summary)


@app.function(image=model_image, volumes={RUNS: volume, HF: hf_volume},
              timeout=60 * 60 * 3, cpu=8.0, memory=16384,
              max_containers=MAX_CONTAINERS)
def e2e_qwen(total_decisions: int = 64, n_envs: int = 2,
             rollout_steps: int = 16) -> str:
    """Real-model pipeline check on CPU — cents, not GPU dollars.

    1) Equivalence gate: features_to(d) must equal features_collect[:,d]
       for every exit on real Qwen3-VL weights.
    2) A tiny training smoke: real features through rollout -> PPO ->
       checkpoints + metrics, proving the loop end-to-end.
    """
    import pathlib
    import time

    import numpy as np

    from reflexrl.cloud_budget import RATES_PER_HR, record_spend, spend_gate
    from reflexrl.models.heads import ExitPolicy
    from reflexrl.models.qwen_vl import QwenVLBackbone
    from reflexrl.rl.agent import ExitAgent
    from reflexrl.rl.trainer import TrainConfig
    from reflexrl.rl.trainer import train as _train

    volume.reload()
    spend_gate(0.4, "e2e_qwen")
    t0 = time.time()

    backbone = QwenVLBackbone(device="cpu")  # fp32
    obs = np.random.default_rng(0).integers(0, 255, (2, 2, 180, 320, 3), dtype=np.uint8)
    h_all = backbone.features_collect(obs)
    eq = {}
    for d in range(h_all.size(1)):
        eq[d] = float((backbone.features_to(obs, d) - h_all[:, d]).abs().max())
    assert max(eq.values()) < 1e-4, f"truncated != collected on real model: {eq}"

    agent = ExitAgent(backbone, ExitPolicy(n_actions=6))
    cfg = TrainConfig(
        total_decisions=total_decisions, n_envs=n_envs,
        rollout_steps=rollout_steps, ckpt_at=(total_decisions,),
        device="cpu", tag="e2e_qwen",
    )
    wd = pathlib.Path(f"{RUNS}/e2e_qwen")
    wd.mkdir(parents=True, exist_ok=True)
    history = _train(agent, cfg, wd)

    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR["cpu"] * 8, "e2e_qwen")
    volume.commit()
    import json

    return json.dumps({
        "trunc_equiv_max_abs": eq,
        "iterations": history["iterations"][-1],
        "ep_return_mean": history["ep_returns"][-1] if history["ep_returns"] else None,
        "mean_depth": history["mean_depth"][-1],
        "recorded_usd": usd,
    })
