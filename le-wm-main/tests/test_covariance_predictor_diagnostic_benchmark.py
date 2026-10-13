import numpy as np

from experiments.empirical_spectral_audit.run_covariance_predictor_diagnostic_benchmark import (
    Config,
    candidates,
    covariance_and_spectrum,
    run,
)


def test_candidate_family_preserves_predictor_spectrum():
    config = Config()
    _, spectrum = covariance_and_spectrum(config)
    rows = candidates(config, spectrum)
    reference = np.sort(np.linalg.eigvalsh(rows[0]["operator"]))
    for row in rows:
        assert np.allclose(np.sort(np.linalg.eigvalsh(row["operator"])), reference)


def test_benchmark_has_disjoint_selection_and_transfer_protocol():
    report = run(Config(repeats=8, train_samples=64, validation_samples=96, test_samples=256))
    protocol = report["protocol"]
    assert protocol["all_candidate_predictor_spectra_identical"] is True
    assert protocol["alignment_fit_split"] == "training covariance only"
    assert protocol["transfer_evaluation_split"] == "independent test states"
    assert np.isfinite(report["summary"]["alignment_selection_regret"]["mean"])
    assert np.isfinite(report["summary"]["matched_pair_transfer_mse_A_minus_B"]["mean"])


def test_primary_benchmark_is_not_claimed_as_universal():
    report = run(Config(repeats=4, train_samples=64, validation_samples=96, test_samples=256))
    assert "controlled linear" in report["protocol"]["claim_boundary"]
    assert "planning evidence" in report["protocol"]["claim_boundary"]
