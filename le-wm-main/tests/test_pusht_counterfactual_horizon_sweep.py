import numpy as np

from experiments.run_pusht_counterfactual import (
    PUBLIC_ACTION_MEAN,
    PUBLIC_ACTION_STD,
    normalized_action_sequence,
)
from experiments.run_pusht_counterfactual_horizon_sweep import bootstrap


def test_counterfactual_horizon_bootstrap_is_finite_and_episode_level():
    result = bootstrap(np.asarray([1.0, 2.0, 3.0, 4.0]), seed=7, draws=100)
    assert result["n"] == 4
    assert np.isfinite(result["estimate"])
    assert result["ci95_low"] <= result["estimate"] <= result["ci95_high"]


def test_counterfactual_action_arms_normalize_each_live_action_independently():
    action = np.asarray([0.55, -0.2], dtype=np.float32)
    plus = normalized_action_sequence(action, 2)
    minus = normalized_action_sequence(-action, 2)
    zero = normalized_action_sequence(np.zeros(2, dtype=np.float32), 2)
    assert plus.shape == (2, 10)
    np.testing.assert_allclose(plus[0, :2], (action / 100.0 - PUBLIC_ACTION_MEAN) / PUBLIC_ACTION_STD)
    np.testing.assert_allclose(minus[0, :2], (-action / 100.0 - PUBLIC_ACTION_MEAN) / PUBLIC_ACTION_STD)
    np.testing.assert_allclose(zero[0, :2], (-PUBLIC_ACTION_MEAN) / PUBLIC_ACTION_STD)
    assert not np.allclose(minus, -plus)


def test_counterfactual_normalization_matches_pusht_internal_action_scale():
    action = np.asarray([0.85, -0.4], dtype=np.float32)
    normalized = normalized_action_sequence(action, 1)[0, :2]
    expected = (action / 100.0 - PUBLIC_ACTION_MEAN) / PUBLIC_ACTION_STD
    np.testing.assert_allclose(normalized, expected)
