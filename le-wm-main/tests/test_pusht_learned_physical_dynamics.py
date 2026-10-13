import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from experiments.run_pusht_learned_physical_dynamics import (
    bootstrap,
    candidate_sequences,
    rollout_model,
    state_features,
)


def test_state_features_encode_orientation_without_wrap_discontinuity():
    state = np.array([1.0, 2.0, 3.0, 4.0, np.pi, 5.0, 6.0])
    features = state_features(state)
    assert features.shape == (8,)
    assert np.isclose(features[4], 0.0, atol=1e-7)
    assert np.isclose(features[5], -1.0, atol=1e-7)


def test_bootstrap_and_candidate_protocol_are_finite_and_include_zero():
    values = bootstrap(np.array([1.0, 2.0, 3.0]), seed=7, draws=100)
    assert values["n"] == 3
    assert values["ci95_low"] <= values["estimate"] <= values["ci95_high"]
    candidates = candidate_sequences(np.random.default_rng(7), 8, 2)
    assert candidates.shape == (8, 2, 2)
    assert np.allclose(candidates[0], 0.0)


def test_learned_rollout_uses_action_conditioned_transition():
    states = np.array([[0.0] * 8, [1.0] * 8, [2.0] * 8, [3.0] * 8])
    actions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    targets = np.concatenate([actions, np.zeros((4, 6))], axis=1)
    model = ExtraTreesRegressor(n_estimators=16, random_state=3, min_samples_leaf=1)
    model.fit(np.concatenate([states, actions], axis=1), targets)
    plus = rollout_model(model, states[1], np.array([1.0, 0.0]), 2)
    zero = rollout_model(model, states[1], np.array([0.0, 0.0]), 2)
    assert np.isfinite(plus).all() and np.isfinite(zero).all()
    assert plus[0] != zero[0]
