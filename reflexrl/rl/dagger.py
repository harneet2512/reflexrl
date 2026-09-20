"""Live teacher in the training loop: Qwen sees, Jev decides, on the student's own states.

Between rollout phases the current student plays a short stretch in a dedicated
environment that keeps full-resolution frames. Qwen3-VL is queried live on those
frames, Jev turns each perception into an action distribution, and the pairs join
the label set. The perception student is then refit, so the fast online teacher
improves during training on exactly the states the student reaches (DAgger,
rather than one offline label dump).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from reflexrl.baselines.perception_bc import (
    PerceptionJevPolicy,
    PerceptionNet,
    train_perception_bc,
)
from reflexrl.env.scenarios import get_scenario
from reflexrl.env.vizdoom_env import DoomEnv
from reflexrl.teacher.perception_jev import CLASSES

DAGGER_SEED = 31337
WHERE = {"left": "on the left side", "center": "in the center, straight ahead",
         "right": "on the right side", "none": "not visible"}


def jev_state(scenario, q: np.ndarray, prev_action: str | None) -> tuple[str, str]:
    """Typed state for Jev, plus a cache key (class, confidence bucket, previous action)."""
    cls = CLASSES[int(np.argmax(q))]
    conf = "high" if q.max() >= 0.8 else ("medium" if q.max() >= 0.5 else "low")
    state = (f"Doom, scenario '{scenario.name}'. {scenario.briefing} "
             f"Perception of the current frame (from a vision model): the nearest monster is "
             f"{WHERE[cls]} (confidence {conf}). Previous action: {prev_action or 'none'}.")
    return state, f"{cls}|{conf}|{prev_action}"


class DaggerTeacher:
    """Fast distilled teacher for rollouts, improved by live Qwen+Jev labelling rounds."""

    def __init__(self, net: PerceptionNet, J: np.ndarray, jev, jev_table_path, scenario: str,
                 labels: dict, device: str = "cuda", every: int = 100_000,
                 steps_per_round: int = 300, max_rounds: int = 4, qwen_model: str =
                 "Qwen/Qwen3-VL-8B-Instruct"):
        self.scenario = get_scenario(scenario)
        self.J = J
        self.jev = jev
        self.jev_table_path = jev_table_path
        self.qwen_model = qwen_model
        self.device = device
        self.every = every
        self.steps_per_round = steps_per_round
        self.max_rounds = max_rounds
        self.live_jev = getattr(jev, "live", False)
        self.labels = {k: np.asarray(v) for k, v in labels.items()}
        self.policy = PerceptionJevPolicy(net, J).to(device).eval()
        self.teacher = None  # Qwen perception, loaded at the first round
        self.rounds = 0
        self.next_round = 0
        self.calls = 0
        self.log: list[dict] = []

    # ---- rollout side ----------------------------------------------------

    @torch.no_grad()
    def probs(self, obs_u8: np.ndarray) -> np.ndarray:
        self.calls += len(obs_u8)
        return self.policy.action_probs(torch.as_tensor(obs_u8, device=self.device)).cpu().numpy()

    # ---- DAgger side -----------------------------------------------------

    def _perception_teacher(self):
        if self.teacher is None:
            from reflexrl.teacher.perception_jev import PerceptionJevTeacher
            from reflexrl.teacher.qwen import QwenTeacher
            base = QwenTeacher(self.qwen_model, dtype=torch.float32, load_4bit=True,
                               device_map="auto")
            self.teacher = PerceptionJevTeacher(base, self.jev_table_path, n_perm=1)
        return self.teacher

    def _label_round(self, student) -> tuple[np.ndarray, np.ndarray]:
        """Student drives the states; Qwen perceives; Jev decides. Returns (obs, pi_T)."""
        teacher = self._perception_teacher()
        env = DoomEnv(self.scenario.name, seed=DAGGER_SEED + self.rounds, keep_full_frames=True)
        obs, _ = env.reset()
        prev_action, obs_batch, pi_batch = None, [], []
        for _ in range(self.steps_per_round):
            q = teacher.perceive([env.teacher_frames()])[0]
            pi = q @ self.J
            if self.live_jev:
                state, key = jev_state(self.scenario, q, prev_action)
                try:
                    pi = self.jev.decide(state, cache_key=key)
                except Exception:
                    pass  # keep the offline-table decision for this state
            obs_batch.append(obs)
            pi_batch.append(pi)
            with torch.no_grad():
                a = int(student.act(torch.as_tensor(obs[None], device=self.device))[0])
            prev_action = self.scenario.action_names[a]
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset()
                prev_action = None
        env.close()
        return np.stack(obs_batch), np.stack(pi_batch)

    def maybe_round(self, step: int, student) -> dict | None:
        """Called at eval points; runs a labelling round when one is due."""
        if step < self.next_round or self.rounds >= self.max_rounds:
            return None
        t0 = time.time()
        new_obs, new_pi = self._label_round(student)
        n_before = len(self.labels["obs"])
        ep = np.full(len(new_obs), 900_000 + self.rounds, np.int32)
        self.labels = {
            "obs": np.concatenate([self.labels["obs"], new_obs]),
            "probs": np.concatenate([self.labels["probs"], new_pi.astype(np.float16)]),
            "action": np.concatenate([self.labels["action"], np.zeros(len(ep), np.int16)]),
            "episode": np.concatenate([self.labels["episode"], ep]),
            "step": np.concatenate([self.labels["step"], np.arange(len(ep), dtype=np.int32)]),
        }
        net, fid = train_perception_bc(self.labels, self.J, device=self.device, epochs=20)
        self.policy = PerceptionJevPolicy(net, self.J).to(self.device).eval()
        self.rounds += 1
        self.next_round = step + self.every
        info = {"kind": "dagger", "step": step, "round": self.rounds,
                "labels": int(len(self.labels["obs"])),
                "added": int(len(self.labels["obs"]) - n_before),
                "val_top1_agree": fid["val_top1_agree"], "jev_live": self.live_jev,
                "jev_calls": getattr(self.jev, "calls", 0),
                "qwen_calls": getattr(getattr(self.teacher, "qwen", None), "samples", 0),
                "seconds": round(time.time() - t0, 1)}
        self.log.append(info)
        print(f"[dagger] round {self.rounds} at {step:,}: +{info['added']} labels on student "
              f"states, perception agreement {fid['val_top1_agree']:.3f}, "
              f"jev_live={self.live_jev}, {info['seconds']}s", flush=True)
        return info
