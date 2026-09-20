"""Probe labels must describe the frames the model is actually shown.

An earlier Health-Gathering probe generated distance labels from a second,
independent rollout, so the reported accuracy compared answers about one set of
frames with ground truth from another.
"""

import pytest

from scripts.hg_probe import collect_with_oracles


@pytest.mark.env
def test_frames_and_both_label_sets_come_from_the_same_states():
    frames, pos, dist = collect_with_oracles("health_gathering", 12, seed=777)
    assert len(frames) == len(pos) == len(dist) == 12
    # a frame with no medkit must be 'none' in both label sets, and vice versa
    for p, d in zip(pos, dist, strict=True):
        assert (p == "none") == (d == "none")
    assert set(pos) <= {"left", "center", "right", "none"}
    assert set(dist) <= {"close", "far", "none"}


@pytest.mark.env
def test_repeat_call_is_deterministic():
    a = collect_with_oracles("health_gathering", 8, seed=777)[1]
    b = collect_with_oracles("health_gathering", 8, seed=777)[1]
    assert a == b, "same seed must give the same labelled states"
