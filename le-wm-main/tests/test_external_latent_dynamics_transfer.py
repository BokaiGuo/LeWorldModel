import numpy as np

from experiments.empirical_spectral_audit.run_external_latent_dynamics_transfer import (
    _fit_probe,
    _load_transfer_bases,
    _r2,
)


def test_transfer_basis_loader_accepts_dynamics_artifact(tmp_path):
    dimension = 4
    eye = np.eye(dimension)
    path = tmp_path / "basis.npz"
    np.savez(
        path,
        latent_mean=np.zeros(dimension),
        covariance_basis=eye,
        dynamics_basis=eye,
        conditional_dynamics_basis=eye,
    )
    mean, bases = _load_transfer_bases(path, random_seed=3)
    assert mean.shape == (dimension,)
    assert set(bases) == {"covariance_basis", "dynamics_basis", "conditional_dynamics_basis", "random_basis"}
    assert all(np.allclose(value.T @ value, eye, atol=1e-6) for value in bases.values())


def test_ridge_probe_recovers_linear_latent_increment_signal():
    source = np.arange(60, dtype=np.float64).reshape(20, 3)
    target = np.stack([source[:, 0] - source[:, 1], 2.0 * source[:, 2]], axis=1)
    probe = _fit_probe(source, target, ridge=1e-3)
    prediction = (source - probe["source_mean"]) @ probe["weights"] + probe["target_mean"]
    assert _r2(prediction, target, target.mean(axis=0)) > 0.99
