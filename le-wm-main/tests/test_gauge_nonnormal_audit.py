import numpy as np

from experiments.empirical_spectral_audit.run_gauge_nonnormal_audit import run


def test_metric_aware_commutator_survives_general_reparameterization():
    report = run()
    gauge = report["gauge"]
    assert abs(gauge["original_metric_commutator"] - gauge["reparameterized_metric_commutator"]) < 1e-12
    assert abs(gauge["original_euclidean_commutator"] - gauge["reparameterized_euclidean_commutator"]) > 1e-5


def test_covariance_basis_identity_is_exact_to_numerical_precision():
    report = run()
    assert report["arbitrary_predictor_identity"]["absolute_delta"] < 1e-10


def test_spectrum_fixed_schur_family_changes_nonnormality():
    report = run()
    rows = report["spectrum_fixed_nonnormal_family"]
    spectra = {
        tuple(tuple(np.round(pair, 8)) for pair in row["eigenvalues"])
        for row in rows
    }
    assert len(spectra) == 1
    assert rows[-1]["nonnormality"] > rows[0]["nonnormality"]
    assert all(np.isfinite(row["metric_commutator"]) for row in rows)
