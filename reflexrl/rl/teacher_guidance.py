"""How the teacher participates in RL: schedules and the teacher interface.

Two mechanisms, both driven by one dependence level ``p`` in [0, 1]:

- intervention: each step, with probability p the teacher's action is
  executed instead of the student's (the behaviour policy is recorded so the
  PPO update can correct for it);
- distillation: an auxiliary KL(pi_T || pi_theta) term weighted by alpha0 * p.

Schedules decide how p falls to zero:

- ``FixedSchedule``: predetermined piecewise-linear anneal over a step horizon.
- ``AdaptiveSchedule`` (ReflexRL): p steps down the ladder only when the
  student's own evaluation return reaches the teacher's measured return
  (hand over control once the student can do what the teacher does), with a
  hard horizon after which p is forced to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

LADDER = (1.0, 0.5, 0.25, 0.1, 0.0)


class Teacher(Protocol):
    def probs(self, obs_u8: np.ndarray) -> np.ndarray:  # (B, C, H, W) -> (B, A)
        ...


@dataclass
class FixedSchedule:
    horizon: int  # env steps over which p goes 1.0 -> 0 through LADDER

    def p(self, step: int) -> float:
        if step >= self.horizon:
            return 0.0
        x = step / self.horizon * (len(LADDER) - 1)
        i = int(x)
        return float(LADDER[i] + (LADDER[i + 1] - LADDER[i]) * (x - i))

    def on_eval(self, step: int, student_return: float) -> None:
        pass


@dataclass
class AdaptiveSchedule:
    teacher_return: float  # teacher's measured student-free return (its own play)
    horizon: int  # hard cap: p = 0 after this many env steps regardless
    min_steps_per_rung: int = 20_000
    drop_ratio: float = 1.5  # student this far above the teacher -> stop guiding at once
    rung: int = 0
    _last_change: int = 0
    history: list = field(default_factory=list)

    def p(self, step: int) -> float:
        return 0.0 if step >= self.horizon else LADDER[self.rung]

    def on_eval(self, step: int, student_return: float) -> None:
        """Hand over control as soon as the student matches the teacher.

        A student that already outscores the teacher has nothing to gain from it
        (and is actively dragged down by intervention), so a student far above the
        teacher skips the ladder entirely. This is what a fixed schedule cannot do.
        """
        if student_return < self.teacher_return or self.rung >= len(LADDER) - 1:
            return
        if student_return >= self.drop_ratio * self.teacher_return:
            self.rung = len(LADDER) - 1  # teacher has nothing left to teach
        elif not self.history or step - self._last_change >= self.min_steps_per_rung:
            self.rung += 1
        else:
            return
        self._last_change = step
        self.history.append((step, LADDER[self.rung], student_return))


@dataclass
class NoTeacher:
    def p(self, step: int) -> float:
        return 0.0

    def on_eval(self, step: int, student_return: float) -> None:
        pass
