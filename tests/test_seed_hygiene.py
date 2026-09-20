"""Training, validation and final-test episodes must come from disjoint seed streams."""

from reflexrl.rl.dagger import DAGGER_SEED
from reflexrl.rl.evaluate import EVAL_SEED_BASE
from scripts.final_eval import TEST_SEED_BASE

TRAIN = {s * 1000 + i for s in range(3) for i in range(16)}
VALID = {EVAL_SEED_BASE + i for i in range(32)}
TEST = {TEST_SEED_BASE + i for i in range(64)}
GATE = {50_000 + off + i for off in (0, 100) for i in range(8)}
DAGGER = {DAGGER_SEED + r for r in range(8)}
PROBE = {777}
DEMO = {424_242 + i for i in range(16)}


def test_streams_are_disjoint():
    groups = {"train": TRAIN, "valid": VALID, "test": TEST, "gate/labels": GATE,
              "dagger": DAGGER, "probe": PROBE, "demo": DEMO}
    for a, sa in groups.items():
        for b, sb in groups.items():
            if a < b:
                assert not sa & sb, f"{a} and {b} share seeds: {sorted(sa & sb)[:5]}"


def test_final_numbers_use_unseen_seeds():
    assert not TEST & VALID, "the final test must not reuse the validation episodes"
    assert not TEST & TRAIN
