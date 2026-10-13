from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit.run_external_rank_target_specificity_audit import (
    _load_all_bases,
    _target,
)


def _orthogonal(dimension: int, seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def test_rank_target_loader_combines_separate_basis_artifacts(tmp_path: Path):
    dimension = 8
    mean = np.arange(dimension, dtype=np.float64)
    covariance = _orthogonal(dimension, 1)
    dynamics = _orthogonal(dimension, 2)
    residual = _orthogonal(dimension, 3)
    residual_covariance = _orthogonal(dimension, 4)
    dynamics_path = tmp_path / "dynamics.npz"
    residual_path = tmp_path / "residual.npz"
    np.savez(
        dynamics_path,
        latent_mean=mean,
        covariance_basis=covariance,
        dynamics_basis=dynamics,
    )
    np.savez(
        residual_path,
        latent_mean=mean,
        residual_basis=residual,
        residual_covariance_basis=residual_covariance,
    )
    loaded_mean, bases = _load_all_bases(dynamics_path, residual_path, random_seed=7)
    assert np.array_equal(loaded_mean, mean)
    assert set(bases) == {
        "covariance_basis",
        "dynamics_basis",
        "residual_basis",
        "random_basis",
        "residual_covariance_basis",
    }
    for basis in bases.values():
        assert basis.shape == (dimension, dimension)
        np.testing.assert_allclose(basis.T @ basis, np.eye(dimension), atol=1e-10)


def test_target_specificity_projection_and_physical_wrap():
    dimension = 8
    identity = np.eye(dimension)
    bases = {
        "dynamics_basis": identity,
        "residual_basis": identity,
    }
    row = {
        "true": np.stack([np.zeros(dimension), np.arange(dimension, dtype=np.float64)]),
        "physical": np.asarray(
            [[0.0, 0.0, 0.0, 0.0, np.pi - 0.1], [1.0, 2.0, 3.0, 4.0, -np.pi + 0.1]]
        ),
    }
    dynamics_target = _target(row, 1, "dynamics_band_increment", bases)
    residual_target = _target(row, 1, "residual_band_increment", bases)
    full_target = _target(row, 1, "full_latent_increment", bases)
    physical_target = _target(row, 1, "physical_displacement", bases)
    assert dynamics_target.shape == (8,)
    assert residual_target.shape == (8,)
    assert full_target.shape == (8,)
    assert physical_target.shape == (5,)
    np.testing.assert_allclose(dynamics_target, np.arange(8, dtype=np.float64))
    np.testing.assert_allclose(residual_target, dynamics_target)
    np.testing.assert_allclose(physical_target[:4], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(physical_target[4], 0.2)
