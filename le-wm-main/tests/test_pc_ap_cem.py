import numpy as np
import pytest

from experiments.spectral_reachability.pc_ap_cem import plan_ap_cem, plan_ap_decision
from experiments.spectral_reachability.pc_ap_observation_memory import ObservationMemory


def test_paper_budget_and_clipping():
    batches = []
    def predictor(actions):
        batches.append(len(actions))
        assert np.all(actions >= -0.5) and np.all(actions <= 0.5)
        return ((actions - 0.2) ** 2).sum(axis=(1, 2))

    result = plan_ap_cem(predictor, np.full((5, 2), -0.5), np.full((5, 2), 0.5), seed=3)
    assert result.predicted_blocks == 9002
    assert result.prediction_batches == 31
    assert batches == [300] * 30 + [2]
    assert result.score < 0.4


def test_strict_tie_uses_initial_and_is_reproducible():
    def predictor(actions):
        return np.zeros(len(actions))
    args = (predictor, np.full((2, 1), -1.), np.full((2, 1), 1.))
    a = plan_ap_cem(*args, seed=9, population=4, iterations=2, elites=2)
    b = plan_ap_cem(*args, seed=9, population=4, iterations=2, elites=2)
    np.testing.assert_array_equal(a.normalized_block, np.zeros((2, 1)))
    np.testing.assert_array_equal(a.normalized_block, b.normalized_block)


def test_invalid_predictor_result_fails():
    with pytest.raises(ValueError, match="finite"):
        plan_ap_cem(lambda x: np.full(len(x), np.nan),
                    np.full((1, 1), -1.), np.full((1, 1), 1.),
                    seed=0, population=2, iterations=1, elites=1)


def test_decision_scores_observed_target_not_final_goal():
    memory = ObservationMemory({"m": np.arange(8, dtype=float)[:, None]}, ["query"], block=2)
    def predictor(actions):
        return actions.sum(axis=1)
    decision = plan_ap_decision(memory, np.array([0.]), np.array([4.]), 4, 0,
                                predictor, np.full((2, 1), -2.), np.full((2, 1), 2.),
                                seed=3, population=40, iterations=5, elites=5)
    np.testing.assert_array_equal(decision.retrieval.target, [2.])
    assert abs(decision.plan.normalized_block.sum() - 2.) < 0.2
    assert decision.plan.predicted_blocks == 202


def test_decision_rejects_block_target_time_mismatch():
    memory = ObservationMemory({"m": np.arange(8, dtype=float)[:, None]}, ["q"], block=2)
    with pytest.raises(ValueError, match="block length"):
        plan_ap_decision(memory, np.array([0.]), np.array([4.]), 4, 0,
                         lambda x: x.sum(axis=1), np.full((1, 1), -1.),
                         np.full((1, 1), 1.), seed=1)
