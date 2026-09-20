"""Train one method on one scenario and seed.

Methods:
  ppo        PPO from scratch, no teacher
  bc_ppo     PPO initialised from the BC network (Qwen imitation), then no teacher
  fixed      intervention + distillation, predetermined anneal of teacher dependence
  reflexrl   intervention + distillation, dependence handed over when the student
             matches the teacher (adaptive), forced to zero by the horizon

    python scripts/train.py --method ppo --scenario dtc --steps 1000000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.baselines.bc import ProxyTeacher  # noqa: E402
from reflexrl.env.scenarios import get_scenario  # noqa: E402
from reflexrl.policy.actor_critic import ActorCritic  # noqa: E402
from reflexrl.rl.ppo import PAUSED_EXIT_CODE, Paused, PPOConfig, train  # noqa: E402
from reflexrl.rl.teacher_guidance import AdaptiveSchedule, FixedSchedule  # noqa: E402
from reflexrl.runinfo import run_metadata  # noqa: E402

METHODS = ("ppo", "bc_ppo", "fixed", "reflexrl")


def load_proxy(teacher_dir: Path, n_actions: int, device: str, want_actor_critic: bool = False):
    """Perception student + Jev table (the Qwen-sees/Jev-decides teacher) for guided RL;
    the action-cloned network when the caller needs an initialisable policy (bc_ppo)."""
    pj = teacher_dir / "perception_jev.pt"
    if pj.exists() and not want_actor_critic:
        from reflexrl.baselines.perception_bc import PerceptionJevPolicy, PerceptionNet
        blob = torch.load(pj, map_location="cpu", weights_only=False)
        net = PerceptionNet()
        net.load_state_dict(blob["perception"])
        info = json.loads((teacher_dir / "perception_teacher.json").read_text())
        return PerceptionJevPolicy(net, blob["J"]), {"bc_eval": info["eval"], "kind": "perception_jev"}
    net = ActorCritic(n_actions)
    net.load_state_dict(torch.load(teacher_dir / "proxy.pt", map_location="cpu"))
    return net, json.loads((teacher_dir / "teacher.json").read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--teacher-horizon", type=float, default=0.4,
                   help="fraction of total steps after which teacher dependence is 0")
    p.add_argument("--teachers", default="runs/teachers")
    p.add_argument("--out", default="runs/train")
    p.add_argument("--device", default="cuda")
    p.add_argument("--init-ckpt", default=None,
                   help="start the student from this checkpoint (held-out adaptation)")
    p.add_argument("--tag", default=None, help="run-name override, e.g. ppo_ft")
    args = p.parse_args()

    spec = get_scenario(args.scenario)
    n_act = len(spec.actions)
    cfg = PPOConfig(scenario=spec.name, total_steps=args.steps, n_envs=args.n_envs,
                    seed=args.seed, device=args.device)
    teacher, schedule, policy = None, None, None
    horizon = int(args.steps * args.teacher_horizon)
    if args.method != "ppo":
        proxy, tinfo = load_proxy(Path(args.teachers) / spec.name, n_act, args.device,
                                  want_actor_critic=args.method == "bc_ppo")
        if args.method == "bc_ppo":
            if not isinstance(proxy, ActorCritic):
                raise SystemExit("bc_ppo needs an action-cloned proxy (proxy.pt)")
            policy = proxy  # fine-tune the imitation network with plain PPO
        else:
            teacher = ProxyTeacher(proxy, args.device)
            cfg.intervene = cfg.distill = True
            schedule = (FixedSchedule(horizon) if args.method == "fixed" else
                        AdaptiveSchedule(teacher_return=tinfo["bc_eval"]["return_mean"],
                                         horizon=horizon))
    if args.init_ckpt:
        if policy is not None:
            raise SystemExit("--init-ckpt cannot be combined with bc_ppo")
        policy = ActorCritic(n_act)
        policy.load_state_dict(torch.load(args.init_ckpt, map_location="cpu"))
    workdir = Path(args.out) / spec.name / f"{args.tag or args.method}_s{args.seed}"
    try:
        train(cfg, workdir, teacher=teacher, schedule=schedule, policy=policy,
              meta=run_metadata(vars(args)))
    except Paused as e:
        print(f"paused: {e}", flush=True)
        sys.exit(PAUSED_EXIT_CODE)
    if isinstance(schedule, AdaptiveSchedule):
        (workdir / "handover.json").write_text(json.dumps(schedule.history))
    print(f"done -> {workdir}", flush=True)


if __name__ == "__main__":
    main()
