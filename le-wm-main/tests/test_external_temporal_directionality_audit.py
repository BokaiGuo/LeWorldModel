import numpy as np
import torch

from experiments.empirical_spectral_audit.run_external_temporal_directionality_audit import (
    _action_features,
    _derangement,
    _episode_bootstrap,
)


def test_action_features_use_matched_summary_shape_and_lag_control():
    actions = np.arange(1 * 20 * 2, dtype=np.float64).reshape(1, 20, 2)
    future = _action_features(actions, 0, start_observation=0, horizon=2, lag=0)
    past = _action_features(actions, 0, start_observation=0, horizon=2, lag=1)
    assert future.shape == (6,)
    assert past.shape == (6,)
    assert np.isfinite(future).all()
    assert np.isfinite(past).all()
    assert not np.allclose(future, past)


def test_within_episode_derangement_has_no_fixed_points():
    rows = [
        {"episode_id": 0},
        {"episode_id": 0},
        {"episode_id": 1},
        {"episode_id": 1},
    ]
    permutation = _derangement(rows)
    assert np.array_equal(permutation, np.asarray([1, 0, 3, 2]))
    assert not np.any(permutation == np.arange(len(rows)))


def test_episode_bootstrap_preserves_clustered_estimand():
    values = np.asarray([1.0, 3.0, 10.0, 12.0])
    groups = np.asarray([0, 0, 1, 1])
    report = _episode_bootstrap(values, groups, seed=7)
    assert report["n_windows"] == 4
    assert report["n_episodes"] == 2
    assert report["mean"] == 6.5
    assert len(report["ci95"]) == 2
    assert all(np.isfinite(report["ci95"]))


def test_normalization_parameter_input_is_tensor_compatible():
    # Keep this small smoke check close to the directionality helpers: action
    # features must accept the same tensor-derived arrays used by the runner.
    actions = torch.zeros((2, 20, 2), dtype=torch.float32).numpy()
    result = _action_features(actions, 1, start_observation=1, horizon=1, lag=0)
    assert result.shape == (6,)
