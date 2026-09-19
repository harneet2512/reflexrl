"""Perception targets must be recoverable from the stored teacher labels."""

import json

import numpy as np

from reflexrl.baselines.perception_bc import recover_perception
from reflexrl.teacher.perception_jev import CLASSES


def test_recover_perception_inverts_the_jev_table():
    data = json.load(open("experiments/configs/jev_table_dtc.json"))
    J = np.array([data["table"][c] for c in CLASSES], np.float64)
    J /= J.sum(1, keepdims=True)
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(4), size=200)
    recovered = recover_perception((q @ J).astype(np.float32), J)
    np.testing.assert_allclose(recovered, q, atol=1e-3)
