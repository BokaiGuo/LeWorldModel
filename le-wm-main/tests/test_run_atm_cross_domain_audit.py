import numpy as np

from experiments.empirical_spectral_audit.run_atm_cross_domain_audit import audit_export


def test_export_audit_reports_inverse_matrix_and_ranking_boundary(tmp_path):
    path = tmp_path / "latent.npz"
    source = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    dynamics = np.asarray([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
    predicted = np.asarray([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
    actions = np.asarray([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    np.savez(path, latents=source, dynamics=dynamics, predicted_dynamics=predicted, actions=actions)
    report = audit_export(path, max_samples=6)
    assert report["schema"] == "atm_cross_domain_diagnostic.v1"
    assert len(report["inverse_risk_matrix"]) == 2
    assert report["ranking_audit"]["status"].startswith("unavailable")
