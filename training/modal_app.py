"""Modal deployment for ReflexRL.

Everything runs in one container per run: ViZDoom envs on the container's
vCPUs, the Qwen backbone + heads on its GPU. No remote env farm, no
network hop per decision — the same economics as sps-rl.

Volumes:
  reflexrl-runs   workdirs: config.json, metrics.jsonl, ckpts, evals
  reflexrl-hf     HF_HOME cache so Qwen3-VL-2B downloads once ever

Usage:
    modal run training/modal_app.py::run_tests
    modal run training/modal_app.py::profile --gpu T4
    modal run training/modal_app.py::train --tag pilot --total-decisions 150000
    modal run training/modal_app.py::evaluate --run-dir /runs/<wd> --mode router
"""

from __future__ import annotations

import json

import modal

APP_NAME = "reflexrl"
RUNS = "/runs"
HF = "/hf"

_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vizdoom==1.3.0", "gymnasium>=1.0", "numpy>=1.26",
        "pytest>=8.0", "opencv-python-headless>=4.8",
    )
)

cpu_image = (
    _base
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .env({"PYTHONPATH": "/root/reflexrl_pkg"})
    .add_local_dir("reflexrl", remote_path="/root/reflexrl_pkg/reflexrl")
    .add_local_dir("tests", remote_path="/root/reflexrl_pkg/tests")
    .add_local_dir("scripts", remote_path="/root/reflexrl_pkg/scripts")
    .add_local_file("pyproject.toml", remote_path="/root/reflexrl_pkg/pyproject.toml")
)

gpu_image = (
    _base
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1")  # PyPI build ships CUDA wheels for Modal GPUs
    .pip_install("torchvision==0.20.1")
    .pip_install(
        "transformers>=4.57", "accelerate>=1.0", "sentencepiece", "protobuf",
        "LevDoom>=1.0", "matplotlib>=3.8", "pillow>=10.0",
    )
    .env({"HF_HOME": HF, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
          "PYTHONPATH": "/root/reflexrl_pkg"})
    .add_local_dir("reflexrl", remote_path="/root/reflexrl_pkg/reflexrl")
    .add_local_dir("tests", remote_path="/root/reflexrl_pkg/tests")
    .add_local_dir("scripts", remote_path="/root/reflexrl_pkg/scripts")
    .add_local_file("pyproject.toml", remote_path="/root/reflexrl_pkg/pyproject.toml")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("reflexrl-runs", create_if_missing=True)
hf_volume = modal.Volume.from_name("reflexrl-hf", create_if_missing=True)

# Hard concurrency ceiling: no function may ever run more than 4
# containers, and the $20 ledger cap in reflexrl.cloud_budget gates spend.
MAX_CONTAINERS = 4

# sm75 (T4) has no bf16 tensor cores -> fp16. Ampere+ -> bf16.
def _model_dtype(gpu_label: str):
    import torch
    return torch.bfloat16 if gpu_label in ("A10G", "L4", "L40S", "A100", "H100") else torch.float16


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


@app.function(image=gpu_image, gpu="T4", volumes={RUNS: volume, HF: hf_volume},
              timeout=60 * 60, max_containers=MAX_CONTAINERS)
def profile(gpu_label: str = "T4") -> dict:
    """Per-exit latency + per-decision cost on the real backbone.

    The number that re-prices every later phase. Runs once per GPU type.
    """
    import time

    import numpy as np
    import torch

    from reflexrl.cloud_budget import RATES_PER_HR, record_spend, spend_gate
    from reflexrl.eval.compute import default_model
    from reflexrl.eval.latency import profile_all
    from reflexrl.models.qwen_vl import QwenVLBackbone

    volume.reload()
    spend_gate(0.6, "profile")
    t0 = time.time()
    backbone = QwenVLBackbone(device="cuda", dtype=_model_dtype(gpu_label))
    obs = np.random.default_rng(0).integers(0, 255, (8, 2, 180, 320, 3), dtype=np.uint8)

    # Equivalence gate: truncated forward must reproduce features_collect.
    # If this fails, exit representations are contaminated by later layers
    # and neither latency nor oracle numbers can be trusted.
    h_all = backbone.features_collect(obs)
    eq = {}
    for d in range(h_all.size(1)):
        h_d = backbone.features_to(obs, d)
        eq[d] = float((h_d - h_all[:, d]).abs().max())
    assert max(eq.values()) < 1e-3, f"truncated != collected: {eq}"

    rows = profile_all(backbone, obs)
    cm = default_model(n_frames=2)
    out = {
        "gpu": gpu_label,
        "torch": torch.__version__,
        "flop_table": cm.table(),
        "trunc_equiv_max_abs": eq,
        "latency": rows,
        "ms_per_decision_deep": rows[-2]["ms_mean"] if len(rows) > 1 else rows[-1]["ms_mean"],
    }
    wd = f"{RUNS}/profile_{gpu_label}"
    import pathlib
    pathlib.Path(wd).mkdir(parents=True, exist_ok=True)
    pathlib.Path(wd, "profile.json").write_text(json.dumps(out, indent=2))
    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR[gpu_label], f"profile:{gpu_label}")
    volume.commit()
    out["recorded_usd"] = usd
    return out


@app.function(image=gpu_image, gpu="T4", volumes={RUNS: volume, HF: hf_volume},
              timeout=60 * 60 * 12, cpu=8.0, max_containers=MAX_CONTAINERS)
def train(tag: str = "pilot", total_decisions: int = 150_000, seed: int = 0,
          gpu_label: str = "T4",
          lam: float = 0.1, forced_d: int | None = None, random_depth: bool = False,
          spinal: bool = False,
          n_envs: int = 8, rollout_steps: int = 256,
          ckpt_at: str = "") -> dict:
    """One training run. forced_d=None -> ReflexRL; int -> fixed baseline."""
    import pathlib
    import time

    from reflexrl.cloud_budget import (
        RATES_PER_HR,
        BudgetWatchdog,
        estimate_train_usd,
        record_spend,
        spend_gate,
    )
    from reflexrl.models.heads import ExitPolicy
    from reflexrl.models.qwen_vl import QwenVLBackbone
    from reflexrl.rl.agent import ExitAgent
    from reflexrl.rl.trainer import TrainConfig
    from reflexrl.rl.trainer import train as _train

    est = estimate_train_usd(total_decisions, gpu="T4")
    volume.reload()
    spend_gate(est, f"train:{tag}")
    t0 = time.time()
    watchdog = BudgetWatchdog(RATES_PER_HR[gpu_label], f"train:{tag}",
                          reload=volume.reload)
    backbone = QwenVLBackbone(device="cuda", dtype=_model_dtype(gpu_label))
    if spinal:
        from reflexrl.models.spinal import ReflexPolicy
        from reflexrl.rl.agent import ReflexAgent
        agent = ReflexAgent(backbone, ReflexPolicy(n_actions=6)).cuda()
    else:
        agent = ExitAgent(backbone, ExitPolicy(n_actions=6)).cuda()
    cfg = TrainConfig(
        total_decisions=total_decisions, seed=seed, lam=lam, forced_d=forced_d,
        random_depth=random_depth, spinal=spinal,
        n_envs=n_envs, rollout_steps=rollout_steps,
        ckpt_at=(tuple(int(x) for x in ckpt_at.split(",") if x.strip())
                 or (10_000, 50_000, 150_000, 300_000)),
        device="cuda", tag=tag,
    )
    wd = pathlib.Path(f"{RUNS}/{tag}_s{seed}")
    history = _train(agent, cfg, wd, budget_check=watchdog.check)
    volume.commit()
    summary = {"tag": tag, "seed": seed, "workdir": str(wd),
               "ep_returns_tail": history["ep_returns"][-5:],
               "mean_depth_tail": history["mean_depth"][-5:]}
    pathlib.Path(wd, "summary.json").write_text(json.dumps(summary, indent=2))
    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR[gpu_label], f"train:{tag}")
    summary["recorded_usd"] = usd
    summary["est_usd"] = est
    volume.commit()
    return summary


@app.function(image=gpu_image, gpu="T4", volumes={RUNS: volume, HF: hf_volume},
              timeout=60 * 60 * 4, max_containers=MAX_CONTAINERS)
def evaluate(run_dir: str, mode: str = "router", n_episodes: int = 10,
             gpu_label: str = "T4",
             seed: int = 10_000, out_name: str | None = None,
             tau: float | None = None, ckpt: str = "ckpt_final.pt") -> dict:
    """Greedy eval of a checkpoint: returns, depth stats, oracle data."""
    import pathlib
    import time

    from reflexrl.cloud_budget import (
        RATES_PER_HR,
        BudgetWatchdog,
        estimate_train_usd,
        record_spend,
        spend_gate,
    )
    from reflexrl.eval.automaticity import load_agent
    from reflexrl.eval.evaluate import evaluate as _eval
    from reflexrl.models.qwen_vl import QwenVLBackbone

    # ~600 decisions/episode at worst-case latency + overhead
    est = estimate_train_usd(n_episodes * 600, gpu="T4", overhead=1.5)
    volume.reload()
    spend_gate(est, f"evaluate:{mode}")
    t0 = time.time()
    watchdog = BudgetWatchdog(RATES_PER_HR[gpu_label], f"evaluate:{mode}",
                          reload=volume.reload)
    backbone = QwenVLBackbone(device="cuda", dtype=_model_dtype(gpu_label))
    agent, _ = load_agent(backbone, f"{run_dir}/{ckpt}")
    if mode in ("router", "random"):
        m = mode
    elif mode == "confidence":
        if tau is None:
            raise ValueError("confidence mode requires --tau")
        m = ("confidence", tau)
    else:
        m = int(mode)
    res = _eval(agent, n_episodes=n_episodes, mode=m, seed=seed,
                budget_check=watchdog.check)
    out = {
        "run_dir": run_dir, "mode": mode, "seed": seed,
        "episode_returns": res.episode_returns, "episode_lens": res.episode_lens,
        "mean_depth": res.mean_depth, "depth_hist": res.depth_hist,
        "mean_oracle_depth": res.mean_oracle_depth,
        "steps": [s.__dict__ for s in res.steps],
    }
    name = out_name or f"eval_{mode}.json"
    pathlib.Path(run_dir, name).write_text(json.dumps(out))
    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR[gpu_label], f"evaluate:{mode}")
    volume.commit()
    res_out = {k: v for k, v in out.items() if k != "steps"}
    res_out["recorded_usd"] = usd
    return res_out


@app.function(image=gpu_image, gpu="T4", volumes={RUNS: volume, HF: hf_volume},
              timeout=60 * 60 * 4, max_containers=MAX_CONTAINERS)
def automaticity(run_dir: str, episodes_per_ckpt: int = 5,
                 gpu_label: str = "T4") -> dict:
    """The automaticity curve: replay every ckpt_*.pt, record depth vs return."""
    import pathlib
    import time

    from reflexrl.cloud_budget import (
        RATES_PER_HR,
        estimate_train_usd,
        record_spend,
        spend_gate,
    )
    from reflexrl.eval.automaticity import automaticity_curve
    from reflexrl.models.qwen_vl import QwenVLBackbone

    n_ckpts = len(list(pathlib.Path(run_dir).glob("ckpt_*.pt")))
    est = estimate_train_usd(n_ckpts * episodes_per_ckpt * 600, gpu="T4", overhead=1.5)
    volume.reload()
    spend_gate(est, "automaticity")
    t0 = time.time()
    backbone = QwenVLBackbone(device="cuda", dtype=_model_dtype(gpu_label))

    out = automaticity_curve(backbone, run_dir, n_episodes=episodes_per_ckpt)
    pathlib.Path(run_dir, "automaticity.json").write_text(json.dumps(out, indent=2))
    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR[gpu_label], "automaticity")
    volume.commit()
    return {"n_checkpoints": len(out["curve"]), "recorded_usd": usd,
            "curve": out["curve"]}


@app.function(image=gpu_image, gpu="T4", volumes={RUNS: volume, HF: hf_volume},
              timeout=60 * 60 * 2, max_containers=MAX_CONTAINERS)
def novelty(run_dir: str, level: int = 2, n_episodes: int = 5,
            gpu_label: str = "T4") -> dict:
    """Novelty probe: router depth on train visuals vs a LevDoom level."""
    import pathlib
    import time

    from reflexrl.cloud_budget import (
        RATES_PER_HR,
        estimate_train_usd,
        record_spend,
        spend_gate,
    )
    from reflexrl.eval.automaticity import load_agent
    from reflexrl.eval.novelty import novelty_probe
    from reflexrl.models.qwen_vl import QwenVLBackbone

    est = estimate_train_usd(2 * n_episodes * 600, gpu="T4", overhead=1.5)
    volume.reload()
    spend_gate(est, "novelty")
    t0 = time.time()
    backbone = QwenVLBackbone(device="cuda", dtype=_model_dtype(gpu_label))
    agent, _ = load_agent(backbone, f"{run_dir}/ckpt_final.pt")

    out = novelty_probe(agent, n_episodes=n_episodes, level=level)
    pathlib.Path(run_dir, f"novelty_L{level}.json").write_text(json.dumps(out, indent=2))
    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR[gpu_label], "novelty")
    volume.commit()
    out["recorded_usd"] = usd
    return out


@app.function(image=gpu_image, gpu="T4", volumes={RUNS: volume, HF: hf_volume},
              timeout=60 * 60 * 2, max_containers=MAX_CONTAINERS)
def demo(run_dir: str, n_steps: int = 400, gpu_label: str = "T4",
         ckpt: str = "ckpt_final.pt", out_name: str = "demo.mp4") -> dict:
    """Offline mp4: gameplay + compute-bar overlay, the PRD section-18 visual."""
    import time

    from reflexrl.cloud_budget import (
        RATES_PER_HR,
        estimate_train_usd,
        record_spend,
        spend_gate,
    )
    from reflexrl.demo.compose import (
        add_thoughts,
        compose_video,
        record_rollout,
        write_trace,
    )
    from reflexrl.eval.automaticity import load_agent
    from reflexrl.eval.compute import default_model
    from reflexrl.models.qwen_vl import QwenVLBackbone

    est = estimate_train_usd(n_steps, gpu="T4", overhead=1.5)
    volume.reload()
    spend_gate(est, "demo")
    t0 = time.time()
    backbone = QwenVLBackbone(device="cuda", dtype=_model_dtype(gpu_label))
    agent, _ = load_agent(backbone, f"{run_dir}/{ckpt}")

    rows = record_rollout(agent, n_steps=n_steps)
    rows = add_thoughts(backbone, rows, deepest=len(agent.policy.depths) - 1)
    model = default_model(n_frames=2)
    cost = model.cost_vector()
    if getattr(agent, "stores", "features") == "hybrid":
        from reflexrl.models.spinal import spinal_flops_per_decision
        deep = model.flops(len(model.exit_layers) - 1)
        cost = [spinal_flops_per_decision(2) / deep] + cost
    out_path = f"{run_dir}/{out_name}"
    compose_video(rows, out_path, cost_vector=cost)
    write_trace(rows, f"{run_dir}/trace.jsonl")
    volume.reload()
    usd = record_spend(time.time() - t0, RATES_PER_HR[gpu_label], "demo")
    volume.commit()
    return {"video": out_path, "steps": len(rows), "recorded_usd": usd}
