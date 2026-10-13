import json

import numpy as np

from experiments.empirical_spectral_audit.run_uncertainty_subspace_conflict_audit import run_report


def test_uncertainty_subspace_audit_is_cross_fitted_and_finite(tmp_path):
    rng = np.random.default_rng(7)
    n, dim = 120, 8
    latent = rng.normal(size=(n, dim))
    dynamics = latent + 0.1 * rng.normal(size=(n, dim))
    predicted = dynamics + 0.1 * rng.normal(size=(n, dim))
    residual = rng.normal(size=(n, dim))
    residual[:12, 0] += 5.0
    arrays = {
        "latents": latent,
        "actions": rng.normal(size=(n, 3)),
        "dynamics": dynamics,
        "predicted_dynamics": predicted,
        "planning": latent[:, :dim] + 0.1 * rng.normal(size=(n, dim)),
        "nuisance": rng.normal(size=(n, dim)),
        "prediction_residual": residual,
    }
    source = tmp_path / "latents.npz"
    np.savez(source, **arrays)
    output = tmp_path / "report.json"
    report = run_report(source, output, max_samples=100, seeds=[1, 2, 3], test_fraction=0.3, ks=[2, 4], ridge=1e-3, permutation_repeats=20, random_basis_repeats=8)
    assert output.is_file() and output.with_suffix(".md").is_file()
    assert report["protocol"]["row_level_split_without_episode_ids"] is True
    assert len(report["split_reports"]) == 3
    assert report["summary"]["planning"]["2"]["n_splits"] == 3
    assert "nuisance_minus_planning" in report["basis_contrasts"]
    assert report["basis_contrasts"]["nuisance_minus_planning"]["2"]["n_splits"] == 3
    assert all(np.isfinite(value) for value in report["summary"]["residual_covariance"]["4"]["split_values"])
    json.loads(output.read_text())
