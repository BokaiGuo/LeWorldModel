from __future__ import annotations

import numpy as np

from experiments.empirical_spectral_audit.run_external_action_algebra_audit import (
    _apply_composed_map,
    _apply_map,
    _design,
    _fit_horizon_map,
    _fit_map,
    _r2,
)


def test_action_conditioned_map_recovers_a_synthetic_transition():
    rng = np.random.default_rng(17)
    states = rng.normal(size=(12, 9, 4))
    actions = rng.normal(size=(12, 8, 2))
    matrix = np.diag([0.9, 0.8, 0.7, 0.6])
    action_matrix = rng.normal(scale=0.1, size=(2, 4))
    bias = rng.normal(scale=0.02, size=4)
    for t in range(8):
        states[:, t + 1] = states[:, t] @ matrix + actions[:, t] @ action_matrix + bias
    fit = _fit_map(_design(states[:, :-1].reshape(-1, 4), actions.reshape(-1, 2)), states[:, 1:].reshape(-1, 4), ridge=1e-7)
    prediction = _apply_map(fit, _design(states[:, :-1].reshape(-1, 4), actions.reshape(-1, 2)))
    assert _r2(prediction, states[:, 1:].reshape(-1, 4)) > 0.999


def test_composition_and_direct_horizon_map_are_finite():
    rng = np.random.default_rng(19)
    states = rng.normal(size=(10, 9, 5))
    actions = rng.normal(size=(10, 8, 2))
    one_step = _fit_map(
        _design(states[:, :-1].reshape(-1, 5), actions.reshape(-1, 2)),
        states[:, 1:].reshape(-1, 5),
        ridge=1e-2,
    )
    direct = _fit_horizon_map(states, actions, horizon=4, ridge=1e-2)
    composed = _apply_composed_map(one_step, states[0, 0], actions[0, :4], 5)
    direct_prediction = _apply_map(direct, _design(states[:1, :1].reshape(-1, 5), actions[:1, :4].reshape(1, -1)))
    assert composed.shape == (5,)
    assert direct_prediction.shape == (1, 5)
    assert np.isfinite(composed).all()
    assert np.isfinite(direct_prediction).all()


def test_direct_horizon_map_preserves_window_start_alignment():
    """The direct-map action sequence must use (window, start) row order."""
    rng = np.random.default_rng(23)
    windows, steps, state_dim, action_dim = 32, 7, 3, 2
    actions = rng.normal(size=(windows, steps - 1, action_dim))
    states = np.zeros((windows, steps, state_dim), dtype=np.float64)
    states[:, 0] = rng.normal(size=(windows, state_dim))
    transition = np.diag([0.82, 0.71, 0.63])
    action_matrix = rng.normal(scale=0.25, size=(action_dim, state_dim))
    bias = rng.normal(scale=0.03, size=state_dim)
    for step in range(steps - 1):
        states[:, step + 1] = (
            states[:, step] @ transition
            + actions[:, step] @ action_matrix
            + bias
        )

    horizon = 3
    fit = _fit_horizon_map(states, actions, horizon=horizon, ridge=1e-8)
    source = states[:, :-horizon].reshape(-1, state_dim)
    action_sequences = np.stack(
        [
            actions[:, start : start + horizon].reshape(-1, horizon * action_dim)
            for start in range(steps - horizon)
        ],
        axis=1,
    ).reshape(-1, horizon * action_dim)
    target = states[:, horizon:].reshape(-1, state_dim)
    prediction = _apply_map(fit, _design(source, action_sequences))

    assert _r2(prediction, target) > 0.995
