import numpy as np
import pytest

from experiments.spectral_reachability import p1_second_feedback_replan_20261102 as subject


def test_context_uses_real_observation_order_and_normalizes_actions_once():
    task = {"latent_dim": 2, "action_mean": [1, 2], "action_std": [2, 4],
            "root_context": {"latent_history": [[0, 0], [7, 8]]}}
    packets = np.zeros((2, 5, 2), np.float32)
    packets[1] = [1, -1]
    states, actions = subject.context(task, [9, 10], [11, 12], packets)
    np.testing.assert_array_equal(states, [[7, 8], [9, 10], [11, 12]])
    np.testing.assert_allclose(actions[0].reshape(5, 2), np.tile([-.5, -.5], (5, 1)))
    np.testing.assert_allclose(actions[1].reshape(5, 2), np.tile([0, -.75], (5, 1)))


@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_invalid_normalization_fails_closed(bad):
    task = {"latent_dim": 2, "action_mean": [0, 0], "action_std": [bad, 1],
            "root_context": {"latent_history": [[0, 0]]}}
    with pytest.raises(ValueError):
        subject.context(task, [0, 0], [0, 0], np.zeros((2, 5, 2)))


def test_frozen_development_protocol():
    spec = subject.protocol()
    assert len(spec["rows"]) == 8
    assert spec["budget_per_arm"]["predict_samples"] == 3200
    assert not spec["physical_execution"] and not spec["P1_admission"]
