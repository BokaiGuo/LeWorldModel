import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as base
from experiments import pc_response_diverse_fixed_terminal_exact_dev_20261103_codex as runner
from experiments.spectral_reachability import pc_response_diverse_compiler_v1 as original
from experiments.spectral_reachability import pc_response_diverse_feedback_v1 as feedback


def test_first_decision_matches_same_core():
    task = base.panel.tasks()[0]
    meter_a, meter_b = base.Budget(), base.Budget()
    a = original.decide("response_diverse", 0, task["goal"], lambda x: meter_a.predict(task, x), meter_a)
    b = feedback.decide("response_diverse", 0, 0, task["goal"], lambda x: meter_b.predict(task, x), meter_b)
    np.testing.assert_array_equal(a["selected"]["chain"], b["selected"]["chain"])
    assert a["selected"]["terminal_score"] == b["selected"]["terminal_score"]
    assert meter_a.token_samples == meter_b.token_samples


def test_remaining_horizons_carry_selection_and_budget():
    task = base.panel.tasks()[0]
    for arm in feedback.ARMS:
        for step in (1, 2):
            horizon = 15 - 5 * step
            carry = np.zeros((horizon, 2))
            meter = base.Budget()
            result = feedback.decide(arm, 0, step, task["goal"], lambda x: meter.predict(task, x), meter, carry)
            assert result["horizon"] == horizon
            assert len(result["selected"]["chain"]) == horizon
            assert meter.token_samples <= 480
            endpoint = base.exact.endpoints(task, carry[None].astype(np.float32))[0]
            assert result["carry_score"] == np.linalg.norm(endpoint - task["goal"])
            assert result["precomparison"]["terminal_score"] <= result["carry_score"]
            if arm != "full_cem":
                assert result["selected"]["terminal_score"] == min(result["precomparison"]["terminal_score"],
                                                                   result["comparison_score"])
                assert len(result["first_bank"]) == 5


def test_reject_invalid_tails():
    meter = base.Budget()
    with pytest.raises(ValueError):
        feedback.decide("response_diverse", 0, 1, [0, 0], lambda x: meter.predict(base.panel.tasks()[0], x),
                        meter, np.zeros((5, 2)))


def test_segmented_dispatch_matches_complete_exact_rollout():
    task = base.panel.tasks()[0]
    actions = np.random.default_rng(11).uniform(-1, 1, (15, 2)).astype(np.float32)
    state = task["start"]
    for start in (0, 5, 10):
        state, _ = runner.advance(state, actions[start:start + 5])
    padded = np.pad(actions.astype(float), ((0, 1), (0, 0))) * base.exact.wall.HIGH
    trajectory, _ = base.exact.wall.rollout(task["start"], padded[None])
    np.testing.assert_allclose(state, trajectory[0, 15], atol=1e-12, rtol=0)
