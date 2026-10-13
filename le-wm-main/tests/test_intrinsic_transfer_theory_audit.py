import numpy as np

from experiments.theory_spectral_transfer.run_intrinsic_transfer_theory_audit import run


def test_metric_diagnostic_is_invariant_and_covariance_only_alignment_is_degenerate():
    report = run(seed=7)
    assert report["gauge"]["metric_invariance_abs_error"] < 1e-10
    assert abs(report["gauge"]["euclidean_before"] - report["gauge"]["euclidean_after"]) > 1e-4
    assert report["covariance_only_no_go"]["whitened_covariance_endomorphism_error"] < 1e-10
    assert report["covariance_only_no_go"]["covariance_only_commutator"] < 1e-10


def test_fixed_spectrum_nonnormality_changes_finite_horizon_gain():
    report = run(seed=7)
    rows = report["spectrum_fixed_nonnormal_family"]
    assert max(row["eigenvalue_match_max_abs"] for row in rows) < 1e-10
    assert rows[0]["metric_nonnormality"] < 1e-10
    assert rows[-1]["metric_nonnormality"] > rows[0]["metric_nonnormality"] + 1e-3
    assert rows[-1]["finite_horizon_gain_h8"] > rows[0]["finite_horizon_gain_h8"]
    assert all(
        row["cauchy_circle_bound_h8"] + 1e-10 >= row["finite_horizon_gain_h8"]
        for row in rows
    )


def test_zero_commutator_jordan_control_changes_transient_gain():
    report = run(seed=7)
    rows = report["jordan_zero_commutator_control"]["rows"]
    assert max(row["eigenvalue_match_max_abs"] for row in rows) < 1e-10
    assert max(row["covariance_commutator"] for row in rows) < 1e-10
    assert rows[0]["metric_nonnormality"] < 1e-10
    assert rows[-1]["metric_nonnormality"] > rows[0]["metric_nonnormality"] + 1e-3
    assert rows[-1]["finite_horizon_gain_h8"] > 1.0
    assert rows[-1]["finite_horizon_gain_h8"] > rows[0]["finite_horizon_gain_h8"]
    assert rows[-1]["covariance_weighted_energy_h8"] > rows[0]["covariance_weighted_energy_h8"]


def test_nonlinear_jacobian_remainder_bound_is_finite_and_valid():
    report = run(seed=7)
    result = report["nonlinear_jacobian_bound"]
    assert np.isfinite(result["actual_two_step_linearization_remainder"])
    assert np.isfinite(result["certified_two_step_remainder_bound"])
    assert result["bound_slack"] >= -1e-12
