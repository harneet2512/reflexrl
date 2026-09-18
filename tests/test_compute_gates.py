"""Gates for eval/, analysis/ — oracle correctness, FLOP table, pareto."""
import json

import numpy as np
import pytest
import torch

from reflexrl.analysis.pareto import (
    build_frontier,
    frontier_point,
    is_dominated,
    summarize_eval_json,
)
from reflexrl.eval.compute import default_model
from reflexrl.eval.evaluate import confidence_tau_for_target, evaluate
from reflexrl.models.heads import ExitPolicy
from reflexrl.models.qwen_vl import StubEncoder
from reflexrl.rl.agent import ExitAgent


@pytest.fixture()
def agent():
    torch.manual_seed(0)
    return ExitAgent(StubEncoder(), ExitPolicy(n_actions=6))


class TestComputeModel:
    def test_table_monotone_and_deep_is_full_cost(self):
        rows = default_model().table()
        assert len(rows) == 3
        fracs = [r["frac_of_deep"] for r in rows]
        assert fracs[-1] == pytest.approx(1.0)
        assert np.all(np.diff(fracs) > 0)

    def test_vit_floor_means_exits_save_a_fraction(self):
        """The whole thesis' honest bound: exits shave ~25-55%, not 90%."""
        rows = default_model().table()
        saving = 1.0 - rows[0]["frac_of_deep"]
        assert 0.10 < saving < 0.70

    def test_cost_vector_normalizes_to_one(self):
        cv = default_model().cost_vector()
        assert cv[-1] == pytest.approx(1.0)
        assert cv[0] < cv[1] < 1.0


class TestEvaluate:
    def test_fixed_depth_uses_only_that_depth(self, agent):
        res = evaluate(agent, n_episodes=2, mode=2, max_steps=50)
        assert all(s.depth == 2 for s in res.steps)
        assert len(res.episode_returns) >= 1

    def test_router_mode_produces_valid_depths(self, agent):
        res = evaluate(agent, n_episodes=3, mode="router", max_steps=100)
        # greedy argmax may collapse at init; valid index is what matters
        assert set(s.depth for s in res.steps) <= {0, 1, 2}
        assert len(res.steps) > 0

    def test_oracle_depth_bounds(self, agent):
        res = evaluate(agent, n_episodes=2, mode="router", max_steps=50)
        assert all(0 <= s.oracle_depth <= 2 for s in res.steps)
        # oracle is cheapest exit agreeing with DEEP -> never deeper than 2
        assert res.mean_oracle_depth >= 0.0

    def test_confidence_mode_escalates_to_deep(self, agent):
        res = evaluate(agent, n_episodes=2, mode=("confidence", -1e9),
                       max_steps=50)
        assert all(s.depth == 2 for s in res.steps)

    def test_random_mode_uses_all_depths(self, agent):
        res = evaluate(agent, n_episodes=3, mode="random", max_steps=200)
        assert len(set(s.depth for s in res.steps)) == 3

    def test_steps_carry_debug_state(self, agent):
        res = evaluate(agent, n_episodes=1, mode="router", max_steps=30)
        assert isinstance(res.steps[0].debug, dict)


class TestConfidenceTau:
    def test_tau_within_entropy_range(self, agent):
        cv = default_model().cost_vector()
        tau = confidence_tau_for_target(agent, 0.7, cv, n_episodes=2, seed=7)
        res = evaluate(agent, n_episodes=2, mode=("confidence", tau),
                       max_steps=100)
        # confidence mode only ever picks REFLEX or DEEP
        assert set(s.depth for s in res.steps) <= {0, 2}


class TestPareto:
    def test_dominance(self):
        pts = [
            frontier_point("cheap", 0.9, 0.3),
            frontier_point("bad", 0.8, 0.5),   # worse AND costlier -> dominated
            frontier_point("deep", 0.95, 0.9),  # best but priciest -> survives
        ]
        dom = is_dominated(pts)
        assert dom == [False, True, False]

    def test_build_frontier_sorts_and_marks(self, tmp_path):
        rows = [
            frontier_point("b", 0.5, 0.9),
            frontier_point("a", 0.9, 0.3),
        ]
        out = tmp_path / "frontier.json"
        res = build_frontier(rows, out=out)
        assert res[0]["name"] == "a"
        assert res[0]["dominated"] is False and res[1]["dominated"] is True
        assert json.loads(out.read_text())[0]["name"] == "a"

    def test_summarize_eval_json(self, tmp_path):
        p = tmp_path / "eval.json"
        p.write_text(json.dumps({"episode_returns": [0.9, 1.0],
                                 "mean_depth": 0.3, "depth_hist": [3, 5, 2]}))
        s = summarize_eval_json(p)
        assert s["mean_return"] == pytest.approx(0.95)
        assert s["n_episodes"] == 2
