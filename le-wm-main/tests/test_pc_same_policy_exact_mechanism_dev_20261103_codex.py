import json

import numpy as np
import pytest

from experiments import pc_same_policy_exact_mechanism_dev_20261103_codex as experiment
from experiments.spectral_reachability import pc_exact_unicycle_development as reference


def test_exact_batch_rollout_matches_reference_scaled_actions():
    rng = np.random.default_rng(4)
    actions = rng.uniform(-1, 1, (7, 15, 2))
    actual = experiment.rollout(actions)
    expected = []
    for chain in actions:
        state = np.zeros(3)
        for action in chain:
            state = reference.step(state, [0.1 * action[0], 0.3 * action[1]])
        expected.append(state)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_reject_invalid_actions_and_enforce_predict_token_meter():
    with pytest.raises(ValueError):
        experiment.rollout(np.full((1, 5, 2), 1.01))
    meter = experiment.Meter(2)
    predict = experiment.predictor(np.array([.9, 0.]), meter)
    positions, scores = predict(np.zeros((2, 5, 2)))
    np.testing.assert_allclose(positions, 0)
    np.testing.assert_allclose(scores, .9)
    with pytest.raises(RuntimeError):
        predict(np.zeros((1, 5, 2)))


def test_sealed_intervention_when_present():
    if not experiment.INTERVENTION.exists():
        pytest.skip("development panel not yet run")
    plans = json.loads(experiment.PLANS.read_text())
    data = json.loads(experiment.INTERVENTION.read_text())
    assert data["plans_sha256"] == experiment.sha(experiment.PLANS)
    assert plans["dependencies"] == experiment.dependencies()
    assert len(data["rows"]) == len(plans["rows"]) == 12
    for row, original in zip(data["rows"], plans["rows"], strict=True):
        chain = np.asarray(row["chain"])
        np.testing.assert_array_equal(chain, original["methods"]["svd"]["decision"]["selected"]["chain"])
        controls = row["controls"]
        np.testing.assert_array_equal(controls["selected"]["prefix"], chain[:5])
        np.testing.assert_array_equal(controls["greedy"]["prefix"], original["same_bank_greedy_prefix"])
        goal = np.asarray(original["goal"])
        for control in controls.values():
            prefix = np.asarray(control["prefix"])
            values = np.asarray(control["common_suffix_distances"])
            actual = experiment.rollout(np.concatenate((prefix, chain[5:]))[None])[0, :2]
            np.testing.assert_allclose(values[-1], np.linalg.norm(actual - goal))
            np.testing.assert_allclose(control["active_changes"], values - values[0])
            assert len(control["own_scores"]) == 120
            assert control["own_optimized_distance"] == min(control["own_scores"])
        assert row["limited_witness"] == (
            row["nonprogress"] and row["active_response"] and row["own_plan_advantage"]
        )
