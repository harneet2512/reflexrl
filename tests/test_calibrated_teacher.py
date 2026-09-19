"""The calibrated teacher must undo option-position bias exactly."""

import numpy as np

from reflexrl.env.scenarios import get_scenario
from reflexrl.teacher.qwen import CalibratedQwenTeacher


class FakeQwen:
    """Prefers option letter A (position bias) plus a true preference for one action."""

    def __init__(self, true_action: str):
        self.true_action = true_action
        self.calls = self.samples = 0
        self.seconds = 0.0

    def choice_probs(self, frames, question, n):
        # Read back the option order from the question text.
        lines = [ln for ln in question.splitlines() if len(ln) > 2 and ln[1:3] == ". "]
        names = [ln[3:] for ln in lines]
        blank = frames[0][0].max() == 0
        logits = np.zeros(n)
        logits[0] += 2.0  # position bias toward "A"
        if not blank:
            logits[names.index(self.true_action)] += 1.5
        p = np.exp(logits) / np.exp(logits).sum()
        return np.tile(p, (len(frames), 1)).astype(np.float32)


def test_calibration_removes_position_bias():
    spec = get_scenario("dtc")
    frame = np.full((180, 320, 3), 50, np.uint8)
    for true_action in spec.action_names:
        t = CalibratedQwenTeacher(FakeQwen(true_action), n_perm=4, seed=0)
        p = t.action_probs([[frame, frame]], spec)[0]
        assert spec.action_names[int(np.argmax(p))] == true_action
        assert abs(p.sum() - 1) < 1e-5
