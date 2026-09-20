"""Is the trained policy reacting to the frame, or running an open-loop routine?

The sceptical reading of a fixed arena is that nothing was learned about
*seeing*: a policy that simply spins at a steady rate and fires on a rhythm
would collect kills because monsters reliably walk into its line of fire. If
that were what happened, the sample-efficiency result would be a timing trick
rather than transferred perception.

Two tests, both using ViZDoom's labels buffer as ground truth. The buffer is
enabled for evaluation only and never reaches the policy, which sees pixels.

BLINDFOLD: replay the same episodes with the observation degraded.
    real        the frames the episode actually produces
    blank       all zeros; the policy is blind
    frozen      the first frame, held forever; a static view
    mismatched  real Doom frames from a DIFFERENT episode, so the input has
                the right statistics and the wrong content
An open-loop routine scores the same in all four. A policy that is genuinely
closed-loop on vision collapses toward random when the frame stops matching
the world.

ALIGNMENT: play normally, and at every decision record what was really on
screen (monster left/centre/right/none) against the action chosen. A policy
that sees fires when a monster is actually in front of it and holds fire when
the view is empty. An open-loop routine fires at the same rate regardless.

    python scripts/perception_check.py --episodes 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.env.vizdoom_env import DoomEnv  # noqa: E402
from reflexrl.policy.actor_critic import ActorCritic  # noqa: E402

SEED = 606060  # disjoint from training, validation, test, labelling and footage
NOT_TARGET = ("DoomPlayer", "Blood", "Clip", "Bullet", "Puff", "Shell")
FIRE_ACTIONS = {2, 3, 4}  # FIRE, TURN_LEFT_FIRE, TURN_RIGHT_FIRE
LEFT_ACTIONS, RIGHT_ACTIONS = {0, 3}, {1, 4}

ARMS = {
    "ReflexRL": "reflexrl-train-lane0/runs/train/defend_the_center/reflexrl_s0",
    "PPO from scratch": "reflexrl-train-ppo/runs/train/defend_the_center/ppo_s0",
    "Shuffled-teacher control":
        "reflexrl-ablation/runs/train/defend_the_center/reflexrl_shuffled_s0",
}


def load(ckpt: Path, n_act: int = 5) -> ActorCritic:
    policy = ActorCritic(n_act)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.eval()
    return policy


@torch.no_grad()
def act(policy: ActorCritic, obs: np.ndarray) -> int:
    return int(policy.act(torch.as_tensor(obs[None]))[0])


def truth(env: DoomEnv) -> str:
    """Ground-truth direction of the nearest monster, from the labels buffer."""
    state = env.game.get_state()
    if state is None:
        return "none"
    objs = [lab for lab in state.labels
            if not any(t.lower() in lab.object_name.lower() for t in NOT_TARGET)]
    if not objs:
        return "none"
    near = max(objs, key=lambda lab: lab.height)
    cx = (near.x + near.width / 2) / env.game.get_screen_width()
    return "left" if cx < 0.4 else "right" if cx > 0.6 else "center"


def donor_frames(scenario: str, seed: int, n: int) -> list[np.ndarray]:
    """Observations from a different episode, for the mismatched condition."""
    env = DoomEnv(scenario, seed=seed)
    obs, _ = env.reset()
    out = [obs.copy()]
    rng = np.random.default_rng(seed)
    try:
        while len(out) < n:
            obs, _, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
            out.append(obs.copy())
            if term or trunc:
                obs, _ = env.reset()
    finally:
        env.close()
    return out


def blindfold(policy: ActorCritic, scenario: str, episodes: int,
              donors: list[np.ndarray]) -> dict[str, dict]:
    """Same episodes, four observation conditions."""
    results: dict[str, dict] = {}
    for mode in ("real", "blank", "frozen", "mismatched"):
        rets = []
        for i in range(episodes):
            torch.manual_seed(SEED + i)
            env = DoomEnv(scenario, seed=SEED + i)
            obs, _ = env.reset()
            first, total, k, done = obs.copy(), 0.0, 0, False
            try:
                while not done:
                    if mode == "real":
                        view = obs
                    elif mode == "blank":
                        view = np.zeros_like(obs)
                    elif mode == "frozen":
                        view = first
                    else:
                        view = donors[(i * 997 + k) % len(donors)]
                    obs, r, term, trunc, _ = env.step(act(policy, view))
                    total += r
                    k += 1
                    done = term or trunc
            finally:
                env.close()
            rets.append(total)
        a = np.array(rets)
        results[mode] = {"mean": float(a.mean()),
                         "se": float(a.std(ddof=1) / np.sqrt(len(a))),
                         "episodes": len(a)}
        print(f"    {mode:<11} {a.mean():5.2f} +- {a.std(ddof=1) / np.sqrt(len(a)):.2f}",
              flush=True)
    return results


def alignment(policy: ActorCritic, scenario: str, episodes: int) -> dict:
    """What was really on screen, against what the policy did about it."""
    table: Counter = Counter()
    for i in range(episodes):
        torch.manual_seed(SEED + i)
        env = DoomEnv(scenario, seed=SEED + i, eval_labels=True)
        obs, _ = env.reset()
        done = False
        try:
            while not done:
                cls = truth(env)  # state the policy is about to act on
                a = act(policy, obs)
                table[(cls, a)] += 1
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
        finally:
            env.close()

    out: dict = {}
    for cls in ("left", "center", "right", "none"):
        n = sum(v for (c, _), v in table.items() if c == cls)
        if not n:
            continue
        fire = sum(v for (c, a), v in table.items() if c == cls and a in FIRE_ACTIONS)
        left = sum(v for (c, a), v in table.items() if c == cls and a in LEFT_ACTIONS)
        right = sum(v for (c, a), v in table.items() if c == cls and a in RIGHT_ACTIONS)
        out[cls] = {"decisions": n, "p_fire": fire / n,
                    "p_turn_left": left / n, "p_turn_right": right / n}
        print(f"    truth={cls:<7} n={n:<6} fire {fire / n:.2f}  "
              f"left {left / n:.2f}  right {right / n:.2f}", flush=True)

    # Does the turn decision depend on which side the monster is on at all?
    # P(turn left | monster left) - P(turn left | monster right). Zero means the
    # policy turns the same way whatever it is looking at; higher means its
    # steering tracks the thing it is supposed to be steering toward.
    if "left" in out and "right" in out:
        out["direction_sensitivity"] = (out["left"]["p_turn_left"]
                                        - out["right"]["p_turn_left"])
        if "center" in out and "none" in out:
            # the same question for shooting: does firing depend on anything
            # being there? near zero means it just holds the trigger down
            out["fire_sensitivity"] = out["center"]["p_fire"] - out["none"]["p_fire"]
        print(f"    direction sensitivity {out['direction_sensitivity']:+.2f}", flush=True)
    return out


def chart(summary: dict, out: Path) -> None:
    """Two panels: how much each policy needs to see, and how well it steers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, DIM = "#16161a", "#8a8a94"
    colour = {"ReflexRL": "#ffc850", "PPO from scratch": "#ff78b4",
              "Shuffled-teacher control": "#7f8fa6"}
    arms = [a for a in colour if a in summary["arms"]]
    if not arms:
        return
    fig = plt.figure(figsize=(11.2, 4.6), dpi=110, facecolor=BG)

    ax = fig.add_axes((0.065, 0.20, 0.52, 0.60), facecolor=BG)
    modes = ["real", "blank", "frozen", "mismatched"]
    labels = ["real\nframes", "blind\n(zeros)", "frozen\nframe", "wrong\nepisode"]
    w = 0.26
    for j, arm in enumerate(arms):
        vals = [summary["arms"][arm]["blindfold"][m]["mean"] for m in modes]
        errs = [summary["arms"][arm]["blindfold"][m]["se"] for m in modes]
        ax.bar([i + (j - 1) * w for i in range(len(modes))], vals, w, yerr=errs,
               color=colour[arm], label=arm, ecolor="#55555f", capsize=2)
    floor = summary.get("random_floor")
    if floor is not None:
        ax.axhline(floor, color=DIM, ls="--", lw=1.2)
        ax.text(-0.42, floor + 0.22, f"random actions ({floor:.2f})",
                color=DIM, fontsize=8.5, ha="left")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(labels, color=DIM, fontsize=9)
    ax.set_ylabel("kills", color=DIM, fontsize=10)
    ax.tick_params(colors=DIM)
    for sp in ax.spines.values():
        sp.set_color("#33333c")
    ax.set_title("Break the view, and the policy stops working",
                 color="#ededed", fontsize=12, loc="left", pad=10)
    leg = ax.legend(facecolor="#1e1e24", edgecolor="#33333c", fontsize=8.5)
    for t in leg.get_texts():
        t.set_color("#dddddd")

    ax2 = fig.add_axes((0.68, 0.20, 0.29, 0.60), facecolor=BG)
    vals = [summary["arms"][a]["alignment"].get("direction_sensitivity", 0.0) for a in arms]
    ax2.barh(range(len(arms)), vals, 0.55, color=[colour[a] for a in arms])
    for i, v in enumerate(vals):
        ax2.text(v + 0.008, i, f"{v:+.2f}", color="#ededed", fontsize=10, va="center")
    ax2.set_yticks(range(len(arms)))
    ax2.set_yticklabels([a.replace(" from scratch", "").replace("-teacher control", " ctrl")
                         for a in arms], color=DIM, fontsize=9)
    ax2.set_xlim(0, max(vals) * 1.35)
    ax2.set_xlabel("P(turn left | monster left)\n minus P(turn left | monster right)",
                   color=DIM, fontsize=9)
    ax2.tick_params(colors=DIM)
    for sp in ax2.spines.values():
        sp.set_color("#33333c")
    ax2.set_title("Does steering track the monster?", color="#ededed", fontsize=12,
                  loc="left", pad=10)
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", default="/mnt/d/reflexrl_backup")
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="results/perception_check.json")
    args = ap.parse_args()

    donors = donor_frames(args.scenario, SEED + 500_000, 400)
    summary: dict = {"seed_base": SEED, "scenario": args.scenario,
                     "episodes": args.episodes, "arms": {}}
    for name, rel in ARMS.items():
        ckpt = Path(args.backup) / rel / "ckpt_final.pt"
        if not ckpt.exists():
            print(f"missing {ckpt}", file=sys.stderr)
            continue
        policy = load(ckpt)
        print(f"\n{name}: blindfold", flush=True)
        bf = blindfold(policy, args.scenario, args.episodes, donors)
        print(f"{name}: alignment", flush=True)
        al = alignment(policy, args.scenario, args.episodes)
        summary["arms"][name] = {"blindfold": bf, "alignment": al}

    # a random-action floor, so "collapsed" has a number rather than a feeling
    rets = []
    for i in range(args.episodes):
        rng = np.random.default_rng(SEED + i)
        env = DoomEnv(args.scenario, seed=SEED + i)
        env.reset()
        total, done = 0.0, False
        try:
            while not done:
                _, r, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
                total += r
                done = term or trunc
        finally:
            env.close()
        rets.append(total)
    summary["random_floor"] = float(np.mean(rets))
    print(f"random actions: {np.mean(rets):.2f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}", flush=True)
    chart(summary, Path("results/demo/perception_check.png"))


if __name__ == "__main__":
    main()
