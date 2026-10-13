from __future__ import annotations

import numpy as np

from experiments.empirical_spectral_audit.run_external_conditional_operator_geometry import (
    _crossfit,
    _direction_scores,
    _labels_from_thresholds,
    _quantile_labels,
    _load_trajectory_cache,
    _save_trajectory_cache,
    _state_matched_permutation_labels,
)


def test_heldout_quantile_labels_are_clipped_instead_of_dropped():
    labels, thresholds = _quantile_labels(np.arange(10, dtype=float), 2)
    heldout = _labels_from_thresholds(np.asarray([-10.0, 100.0]), thresholds, 2)
    assert labels.min() == 0 and labels.max() == 1
    assert np.array_equal(heldout, np.asarray([0, 1]))


def test_direction_scores_are_finite_and_scale_invariant():
    actions = np.asarray([[2.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [-2.0, 0.0]])
    scores = _direction_scores(actions, actions)
    scaled = _direction_scores(actions * 7.0, actions)
    assert np.isfinite(scores).all()
    assert np.allclose(scores, scaled)


def test_state_matched_null_preserves_condition_counts_within_state_strata():
    train_states = np.arange(80, dtype=float).reshape(40, 2)
    test_states = np.arange(40, dtype=float).reshape(20, 2)
    train_labels = np.tile([0, 1], 20)
    test_labels = np.tile([0, 1], 10)
    shuffled_train, shuffled_test = _state_matched_permutation_labels(
        train_labels, test_labels, train_states, test_states, seed=4, num_strata=4
    )
    train_strata, thresholds = _quantile_labels(train_states[:, 0], 4)
    test_strata = _labels_from_thresholds(test_states[:, 0], thresholds, 4)
    for stratum in range(4):
        assert np.array_equal(
            np.bincount(shuffled_train[train_strata == stratum], minlength=2),
            np.bincount(train_labels[train_strata == stratum], minlength=2),
        )
        assert np.array_equal(
            np.bincount(shuffled_test[test_strata == stratum], minlength=2),
            np.bincount(test_labels[test_strata == stratum], minlength=2),
        )


def test_crossfit_reports_magnitude_and_direction_geometry():
    rng = np.random.default_rng(23)
    transition = np.diag([0.9, 0.75, 0.6, 0.45])
    rows = []
    for episode_id in range(6):
        for start in range(2):
            initial = rng.normal(size=4)
            true = np.stack(
                [initial @ np.linalg.matrix_power(transition, step).T for step in range(9)]
            )
            predicted = true + rng.normal(scale=0.01, size=true.shape)
            actions = rng.normal(size=(8, 2))
            rows.append(
                {
                    "episode_id": episode_id,
                    "true": true,
                    "predicted": predicted,
                    "action_features": actions,
                }
            )
    report = _crossfit(
        rows, pca_dimension=3, ridge=1e-2, num_bins=2, min_samples=8, state_strata=4
    )
    for source in ("true", "predicted"):
        for condition in ("magnitude", "direction"):
            summary = report["summary"]["by_source"][source][condition]
            assert np.isfinite(summary["conditional_minus_random/commutator"]["mean"])
            assert np.isfinite(
                summary["conditional_minus_random/operator_heterogeneity"]["mean"]
            )
            assert np.isfinite(
                summary["conditional_minus_state_matched/operator_heterogeneity"]["mean"]
            )
            assert summary["conditional_minus_state_matched/operator_heterogeneity"][
                "bootstrap_repeats"
            ] == 2000
            assert summary["conditional/covered_test_rows"]["mean"] == 16.0
            assert summary["random_partition/covered_test_rows"]["mean"] == 16.0
            assert summary["state_matched_partition/covered_test_rows"]["mean"] == 16.0


def test_trajectory_cache_round_trips_protocol_metadata(tmp_path):
    rows = [
        {
            "episode_id": 2,
            "start_observation": 7,
            "true": np.zeros((9, 3)),
            "predicted": np.ones((9, 3)),
            "action_features": np.zeros((8, 2)),
        }
    ]
    path = tmp_path / "trajectory_cache.npz"
    _save_trajectory_cache(rows, path, "public", windows_per_episode=2, max_horizon=8)
    restored = _load_trajectory_cache(path, "public", windows_per_episode=2, max_horizon=8)
    assert restored[0]["episode_id"] == 2
    assert np.allclose(restored[0]["predicted"], rows[0]["predicted"])
