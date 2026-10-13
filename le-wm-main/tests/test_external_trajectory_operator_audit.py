from __future__ import annotations

import numpy as np

from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (
    _fit_transition_family,
    _operator_diagnostics,
    _pca_fit,
    _pca_transform,
    _crossfit,
)
from experiments.empirical_spectral_audit.run_external_trajectory_operator_association import _association
from experiments.empirical_spectral_audit.run_external_physical_observability_audit import (
    _decode,
    _fit_probe,
    _normalized_errors,
)
from experiments.empirical_spectral_audit.run_external_physical_band_probe import (
    _band_coordinates,
    _load_bases,
)
from experiments.empirical_spectral_audit.run_external_physical_dynamics_probe import (
    _physical_displacement,
    _r2,
)
from experiments.empirical_spectral_audit.run_external_physical_action_alignment_probe import (
    _permutations,
)


def test_pca_and_operator_diagnostics_are_finite():
    rng = np.random.default_rng(12)
    states = rng.normal(size=(8, 9, 6))
    fit = _pca_fit(states, dimension=4)
    reduced = _pca_transform(states, fit)
    assert reduced.shape == (8, 9, 4)
    family = _fit_transition_family(reduced, (1, 2, 4, 8), ridge=1e-2)
    metrics = _operator_diagnostics(family["transition"]["row_operator"], (1, 2, 4, 8))
    assert metrics["dimension"] == 4
    assert np.isfinite(metrics["spectral_radius"])
    assert set(metrics["operator_powers"]) == {"1", "2", "4", "8"}


def test_crossfit_keeps_episode_clusters_and_reports_error_amplification():
    rng = np.random.default_rng(4)
    rows = []
    transition = np.diag([0.95, 0.85, 0.75, 0.65, 0.55, 0.45])
    for episode_id in range(6):
        for start in range(2):
            initial = rng.normal(size=6)
            true = np.stack([initial @ np.linalg.matrix_power(transition, step).T for step in range(9)])
            predicted = true + rng.normal(scale=0.01, size=true.shape)
            rows.append({"episode_id": episode_id, "start_observation": start, "true": true, "predicted": predicted})
    report = _crossfit(rows, dimension=4, ridge=1e-2, horizons=(1, 2, 4, 8), bootstrap_repeats=128)
    assert len(report["folds"]) == 6
    assert report["summary"]["error_amplification"]["h8"]["mse"]["clusters"] == 6
    assert np.isfinite(report["summary"]["paired_operator_difference"]["composition_defect/h8"]["mean"])


def test_heldout_association_reports_rank_and_linear_correlations():
    result = _association(np.arange(6, dtype=float), -np.arange(6, dtype=float))
    assert result["n"] == 6
    assert result["pearson"] < -0.99
    assert result["spearman"] < -0.99


def test_physical_probe_decodes_a_linear_state_and_normalizes_errors():
    rng = np.random.default_rng(9)
    source = rng.normal(size=(32, 4))
    weights = rng.normal(size=(4, 2))
    target = source @ weights
    probe = _fit_probe(source, target, ridge=1e-6)
    decoded = _decode(probe, source[:3])
    mse, rmse = _normalized_errors(decoded, target[:3], np.ones(2))
    assert mse < 1e-8
    assert np.max(rmse) < 1e-4


def test_physical_band_probe_loads_orthonormal_bands(tmp_path):
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(6, 6))
    basis, _ = np.linalg.qr(matrix)
    path = tmp_path / "basis.npz"
    np.savez(
        path,
        latent_mean=np.zeros(6),
        covariance_basis=basis,
        dynamics_basis=basis,
        residual_basis=basis,
        residual_covariance_basis=basis,
    )
    mean, bases = _load_bases(path, random_seed=7)
    coords = _band_coordinates(np.ones((2, 6)), mean, bases["covariance_basis"], (0, 3))
    assert coords.shape == (2, 3)
    assert np.allclose(bases["random_basis"].T @ bases["random_basis"], np.eye(6), atol=1e-6)


def test_physical_displacement_wraps_angle_and_r2_is_finite():
    physical = np.zeros((2, 5), dtype=float)
    physical[0, 4] = 2.0 * np.pi - 0.05
    physical[1, 4] = 0.05
    displacement = _physical_displacement(physical, 1)
    assert np.isclose(displacement[4], 0.1)
    target = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(_r2(target, target, target.mean(axis=0)), 1.0)
    assert np.isfinite(_r2(target * 0.5, target, target.mean(axis=0)))


def test_action_alignment_controls_break_alignment_without_fixed_points():
    rows = [
        {"episode_id": 0, "start_observation": 0},
        {"episode_id": 0, "start_observation": 5},
        {"episode_id": 1, "start_observation": 0},
        {"episode_id": 1, "start_observation": 5},
    ]
    permutations = _permutations(rows, seed=3)
    assert np.array_equal(permutations["within_episode_shift"], [1, 0, 3, 2])
    for name in ("within_episode_shift", "global_derangement"):
        assert not np.any(permutations[name] == np.arange(len(rows)))
