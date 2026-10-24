from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "empirical_spectral_audit"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from run_enriched_action_task_bridge import (  # noqa: E402
    _attach_observed_zero_effect,
    _candidate_operators,
    _joint_geometry,
    _metric_matrices,
    _orthogonal,
    _ridge,
)


def test_orthogonal_conjugation_preserves_spectrum() -> None:
    rng = np.random.default_rng(7)
    anchor = rng.normal(size=(5, 5))
    q = _orthogonal(5, 11)
    observed = np.sort_complex(np.linalg.eigvals(q @ anchor @ q.T))
    expected = np.sort_complex(np.linalg.eigvals(anchor))
    assert np.max(np.abs(observed - expected)) < 1e-10


def test_external_pullback_metrics_are_finite_positive_definite() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(24, 6))
    proprio = rng.normal(size=(24, 4))
    actions = rng.normal(size=(24, 2))
    delta = rng.normal(size=(24, 7))
    metrics = _metric_matrices(x, proprio, actions, delta, floor=0.05, ridge=0.05)
    for matrix in metrics.values():
        assert np.isfinite(matrix).all()
        assert np.allclose(matrix, matrix.T, atol=1e-10)
        assert np.min(np.linalg.eigvalsh(matrix)) > 0.0


def test_action_residual_fit_does_not_change_visual_anchor() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(30, 4))
    actions = rng.normal(size=(30, 2))
    target = x @ rng.normal(size=(4, 4)) + actions @ rng.normal(size=(2, 4))
    anchor = _ridge(x, target, 0.05)
    residual_fit = _ridge(actions, target - x @ anchor, 0.05)
    assert np.allclose(anchor, _ridge(x, target, 0.05))
    assert np.isfinite(residual_fit).all()


def test_observed_zero_effect_is_paired_and_exposed_to_metric_fit() -> None:
    observed = {
        "0:0": {
            "episode_id": 0,
            "latent": np.asarray([1.0, 2.0]),
            "pred": np.asarray([1.5, 2.5]),
            "target": np.asarray([1.7, 2.7]),
            "pred_downstream": np.asarray([1.6, 2.6]),
            "target_downstream": np.asarray([1.8, 2.8]),
            "action_sequence": np.asarray([[0.2, -0.1]]),
            "predictor_action_sequence": np.asarray([[0.2, -0.1]]),
        }
    }
    zero = {
        "0:0": {
            "episode_id": 0,
            "latent": np.asarray([1.0, 2.0]),
            "pred": np.asarray([1.1, 2.2]),
            "target": np.asarray([1.7, 2.7]),
            "pred_downstream": np.asarray([1.2, 2.3]),
            "target_downstream": np.asarray([1.8, 2.8]),
            "action_sequence": np.asarray([[0.2, -0.1]]),
            "predictor_action_sequence": np.zeros((1, 2)),
        }
    }
    audit = _attach_observed_zero_effect(observed, zero)
    assert audit["matched_samples"] == 1
    assert np.allclose(observed["0:0"]["latent_action_effect"], [0.4, 0.3])
    x = np.asarray([[1.0, 2.0], [1.2, 2.2], [0.8, 1.8]])
    proprio = np.zeros((3, 1))
    action = np.zeros((3, 1))
    delta = np.asarray([[0.1], [0.2], [0.0]])
    effect = np.tile(observed["0:0"]["latent_action_effect"], (3, 1))
    metrics = _metric_matrices(x, proprio, action, delta, floor=0.05, ridge=0.05, latent_action_effect=effect)
    assert "observed_zero_action_effect_pullback" in metrics


def test_observed_zero_effect_rejects_undeclared_or_nonzero_predictor_control() -> None:
    observed = {
        "0:0": {
            "episode_id": 0, "latent": np.asarray([1.0, 2.0]),
            "pred": np.asarray([1.5, 2.5]), "target": np.asarray([1.7, 2.7]),
            "pred_downstream": np.asarray([1.6, 2.6]), "target_downstream": np.asarray([1.8, 2.8]),
            "action_sequence": np.asarray([[0.2, -0.1]]), "predictor_action_sequence": np.asarray([[0.2, -0.1]]),
        }
    }
    control = {
        "0:0": {
            "episode_id": 0, "latent": np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
            "pred": np.asarray([1.1, 2.2]), "target": np.asarray([1.7, 2.7]),
            "pred_downstream": np.asarray([1.2, 2.3]), "target_downstream": np.asarray([1.8, 2.8]),
            "action_sequence": np.asarray([[0.2, -0.1]]), "predictor_action_sequence": np.asarray([[0.1, 0.0]]),
        }
    }
    import pytest
    with pytest.raises(ValueError, match="non-zero predictor actions"):
        _attach_observed_zero_effect(observed, control)
    del control["0:0"]["predictor_action_sequence"]
    with pytest.raises(ValueError, match="lacks predictor_action_blocks_normalized"):
        _attach_observed_zero_effect(observed, control)


def test_local_candidate_family_preserves_spectrum_and_augmented_geometry_is_finite() -> None:
    rng = np.random.default_rng(10)
    anchor = rng.normal(size=(5, 5))
    operators = _candidate_operators(anchor, candidates=9, seed=3, family="local")
    expected = np.sort_complex(np.linalg.eigvals(anchor))
    for operator in operators:
        observed = np.sort_complex(np.linalg.eigvals(operator))
        assert np.max(np.abs(observed - expected)) < 1e-10
    action_cov = np.eye(2)
    metric = np.eye(5)
    geometry = _joint_geometry(rng.normal(size=(5, 5)), action_cov, (anchor, rng.normal(size=(2, 5))), metric)
    assert all(np.isfinite(value) for value in geometry.values())
