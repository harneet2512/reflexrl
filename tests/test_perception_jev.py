"""pi_T must be Jev's decision averaged under Qwen's perception."""

import json

import numpy as np

from reflexrl.env.scenarios import get_scenario
from reflexrl.teacher.perception_jev import CLASSES, PerceptionJevTeacher


class FakeQwen:
    """Perceives 'right' with certainty, whatever the option order."""

    def __init__(self):
        self.calls = self.samples = 0
        self.seconds = 0.0

    def choice_probs(self, frames, question, n):
        lines = [ln for ln in question.splitlines() if len(ln) > 2 and ln[1:3] == ". "]
        p = np.array([1.0 if "right side" in ln else 0.0 for ln in lines])
        return np.tile(p, (len(frames), 1))


def test_expected_jev_decision_under_perception(tmp_path):
    spec = get_scenario("dtc")
    table = {c: [0.0] * 5 for c in CLASSES}
    table["right"] = [0.0, 0.1, 0.0, 0.0, 0.9]
    table["left"] = [0.1, 0.0, 0.0, 0.9, 0.0]
    table["center"] = [0.0, 0.0, 1.0, 0.0, 0.0]
    table["none"] = [0.5, 0.5, 0.0, 0.0, 0.0]
    path = tmp_path / "jev.json"
    path.write_text(json.dumps({"actions": spec.action_names, "table": table}))
    t = PerceptionJevTeacher(FakeQwen(), path)
    frame = np.zeros((180, 320, 3), np.uint8)
    p = t.action_probs([[frame, frame]], spec)[0]
    np.testing.assert_allclose(p, table["right"], atol=1e-6)
