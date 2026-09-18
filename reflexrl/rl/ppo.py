"""PPO with optional teacher guidance (intervention and/or distillation).

With guidance off this is plain PPO (the scratch baseline). With intervention
on, teacher-executed steps are corrected with decoupled PPO: the clipped
ratio is taken against the student's own old policy, and each sample is
weighted by the truncated importance weight w = min(1, pi_old(a) / b(a)),
where b is the behaviour distribution that actually chose the action.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reflexrl.env.vizdoom_env import make_vec_env
from reflexrl.policy.actor_critic import ActorCritic
from reflexrl.rl.evaluate import evaluate_policy


@dataclass
class PPOConfig:
    scenario: str = "dtc"
    total_steps: int = 1_000_000
    n_envs: int = 16
    n_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "cuda"
    eval_every: int = 50_000
    eval_episodes: int = 16
    ckpt_every: int = 250_000
    intervene: bool = False
    distill: bool = False
    alpha0: float = 1.0  # distillation weight at p = 1


class RunningStd:
    """Running std of the discounted return, for reward scaling."""

    def __init__(self, n_envs: int, gamma: float):
        self.ret = np.zeros(n_envs)
        self.gamma = gamma
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def scale(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + rewards
        b_mean, b_var, n = self.ret.mean(), self.ret.var(), len(self.ret)
        delta, tot = b_mean - self.mean, self.count + n
        self.mean += delta * n / tot
        self.var = (self.var * self.count + b_var * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot
        self.ret[dones] = 0.0
        return rewards / np.sqrt(self.var + 1e-8)


def _gae(rew, val, done, last_val, gamma, lam):
    adv = torch.zeros_like(rew)
    gae = torch.zeros_like(last_val)
    for t in reversed(range(rew.shape[0])):
        nxt = last_val if t == rew.shape[0] - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv


def train(cfg: PPOConfig, workdir: Path, teacher=None, schedule=None,
          policy: ActorCritic | None = None, meta: dict | None = None) -> ActorCritic:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = torch.device(cfg.device)
    envs = make_vec_env(cfg.scenario, cfg.n_envs, seed=cfg.seed)
    n_act = envs.single_action_space.n
    policy = (policy or ActorCritic(n_act)).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    (workdir / "config.json").write_text(json.dumps(
        {"ppo": asdict(cfg), "meta": meta or {},
         "schedule": type(schedule).__name__ if schedule else None}, indent=2))
    log = (workdir / "metrics.jsonl").open("a")

    T, N = cfg.n_steps, cfg.n_envs
    obs_buf = torch.zeros((T, N, *envs.single_observation_space.shape), dtype=torch.uint8, device=dev)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=dev)
    logp_old = torch.zeros((T, N), device=dev)
    logp_beh = torch.zeros((T, N), device=dev)
    val_buf = torch.zeros((T, N), device=dev)
    rew_buf = torch.zeros((T, N), device=dev)
    done_buf = torch.zeros((T, N), device=dev)
    tprobs_buf = torch.zeros((T, N, n_act), device=dev)
    scaler = RunningStd(N, cfg.gamma)

    obs, _ = envs.reset(seed=cfg.seed)
    ep_ret, ep_len = np.zeros(N), np.zeros(N, dtype=int)
    step, next_eval, next_ckpt, t0 = 0, 0, cfg.ckpt_every, time.time()
    recent, teacher_steps = [], 0
    use_teacher = teacher is not None and (cfg.intervene or cfg.distill)

    while step < cfg.total_steps:
        if step >= next_eval:
            ev = evaluate_policy(policy, cfg.scenario, cfg.eval_episodes, cfg.device)
            if schedule is not None:
                schedule.on_eval(step, ev["return_mean"])
            p_now = schedule.p(step) if schedule else 0.0
            log.write(json.dumps({"kind": "eval", "step": step, "p_teacher": p_now,
                                  "teacher_steps": teacher_steps, **ev}) + "\n")
            log.flush()
            print(f"[{cfg.scenario} s{cfg.seed}] step {step:>9,} eval {ev['return_mean']:8.2f} "
                  f"+-{ev['return_se']:.2f} p_T {p_now:.2f} sps {step / max(time.time() - t0, 1e-9):.0f}",
                  flush=True)
            next_eval += cfg.eval_every
        p = schedule.p(step) if schedule is not None else 0.0
        frac = 1.0 - (step / cfg.total_steps)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * frac

        for t in range(T):
            o = torch.as_tensor(obs, device=dev)
            with torch.no_grad():
                logits, v = policy(o)
            probs_s = torch.softmax(logits, -1)
            a = torch.distributions.Categorical(probs=probs_s).sample()
            behav = probs_s
            if use_teacher and p > 0:
                tp = torch.as_tensor(teacher.probs(obs), device=dev)
                tprobs_buf[t] = tp
                if cfg.intervene:
                    mask = torch.as_tensor(rng.random(N) < p, device=dev)
                    a_t = torch.distributions.Categorical(probs=tp).sample()
                    a = torch.where(mask, a_t, a)
                    behav = torch.where(mask[:, None], tp, probs_s)
                    teacher_steps += int(mask.sum())
            obs_buf[t], act_buf[t], val_buf[t] = o, a, v
            logp_old[t] = torch.log(probs_s.gather(1, a[:, None]).squeeze(1) + 1e-8)
            logp_beh[t] = torch.log(behav.gather(1, a[:, None]).squeeze(1) + 1e-8)
            obs, r, term, trunc, _ = envs.step(a.cpu().numpy())
            done = np.logical_or(term, trunc)
            ep_ret += r
            ep_len += 1
            for i in np.flatnonzero(done):
                recent.append(ep_ret[i])
                ep_ret[i], ep_len[i] = 0.0, 0
            rew_buf[t] = torch.as_tensor(scaler.scale(r, done), device=dev, dtype=torch.float32)
            done_buf[t] = torch.as_tensor(done, device=dev, dtype=torch.float32)
            step += N

        with torch.no_grad():
            last_v = policy(torch.as_tensor(obs, device=dev))[1]
        adv = _gae(rew_buf, val_buf, done_buf, last_v, cfg.gamma, cfg.gae_lambda)
        ret = adv + val_buf
        stats = _update(policy, opt, cfg, obs_buf, act_buf, logp_old, logp_beh, adv, ret,
                        tprobs_buf if (cfg.distill and use_teacher and p > 0) else None,
                        cfg.alpha0 * p)
        log.write(json.dumps({"kind": "train", "step": step, "p_teacher": p,
                              "train_return": float(np.mean(recent[-20:])) if recent else None,
                              "teacher_steps": teacher_steps, **stats}) + "\n")
        if step >= next_ckpt:
            torch.save(policy.state_dict(), workdir / f"ckpt_{step}.pt")
            next_ckpt += cfg.ckpt_every

    ev = evaluate_policy(policy, cfg.scenario, cfg.eval_episodes * 2, cfg.device)
    log.write(json.dumps({"kind": "eval", "step": step, "final": True, "p_teacher": 0.0,
                          "teacher_steps": teacher_steps, **ev}) + "\n")
    log.close()
    torch.save(policy.state_dict(), workdir / "ckpt_final.pt")
    (workdir / "done.json").write_text(json.dumps(
        {"steps": step, "final_eval": ev["return_mean"], "wall_s": time.time() - t0,
         "teacher_steps": teacher_steps}))
    envs.close()
    return policy


def _update(policy, opt, cfg, obs_b, act_b, logp_old, logp_beh, adv, ret, tprobs, alpha):
    T, N = act_b.shape
    B = T * N
    flat = lambda x: x.reshape(B, *x.shape[2:])  # noqa: E731
    obs_f, act_f, lo_f, lb_f = flat(obs_b), flat(act_b), flat(logp_old), flat(logp_beh)
    adv_f, ret_f = flat(adv), flat(ret)
    is_w = torch.clamp(torch.exp(lo_f - lb_f), max=1.0)  # 1 for student-chosen steps
    tp_f = flat(tprobs) if tprobs is not None else None
    mb = B // cfg.minibatches
    out = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl_teacher": 0.0, "is_w": float(is_w.mean())}
    for _ in range(cfg.epochs):
        perm = torch.randperm(B, device=adv.device)
        for k in range(cfg.minibatches):
            idx = perm[k * mb:(k + 1) * mb]
            logits, v = policy(obs_f[idx])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, act_f[idx, None]).squeeze(1)
            a = adv_f[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = torch.exp(logp - lo_f[idx])
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a)
            pg = -(is_w[idx] * surr).mean()
            vf = F.mse_loss(v, ret_f[idx])
            ent = -(logp_all.exp() * logp_all).sum(-1).mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
            if tp_f is not None and alpha > 0:
                tp = tp_f[idx]
                kl = (tp * (torch.log(tp + 1e-8) - logp_all)).sum(-1).mean()
                loss = loss + alpha * kl
                out["kl_teacher"] += float(kl) / (cfg.epochs * cfg.minibatches)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()
            out["pg"] += float(pg) / (cfg.epochs * cfg.minibatches)
            out["vf"] += float(vf) / (cfg.epochs * cfg.minibatches)
            out["ent"] += float(ent) / (cfg.epochs * cfg.minibatches)
    return out
