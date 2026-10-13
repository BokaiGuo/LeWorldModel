import numpy as np

from experiments.theory_spectral_transfer.finite_horizon_transfer import (
    companion_realization,
    finite_horizon_response,
    finite_horizon_transfer_risk_bound,
    same_spectrum_different_transfer_example,
    similarity_transform,
    transfer_function,
)


def test_finite_horizon_response_is_similarity_invariant():
    A = np.diag([0.8, 0.5])
    B = np.asarray([[1.0], [0.2]])
    D = np.asarray([[0.4, 1.2]])
    T = np.asarray([[1.0, 0.3], [0.0, 1.0]])
    transformed = similarity_transform(A, B, D, T)
    np.testing.assert_allclose(finite_horizon_response(A, B, D, 4), finite_horizon_response(*transformed, 4))
    np.testing.assert_allclose(transfer_function(A, B, D, 1.7), transfer_function(*transformed, 1.7))


def test_companion_realization_has_expected_dimensions():
    A0 = np.eye(2) * 0.5
    A1 = np.eye(2) * 0.2
    A, B, D = companion_realization([A0, A1], np.ones((2, 1)), np.eye(2))
    assert A.shape == (4, 4)
    assert B.shape == (4, 1)
    assert D.shape == (2, 4)


def test_transfer_bound_is_finite_and_declares_unidentified_terms():
    result = finite_horizon_transfer_risk_bound(
        horizon=4, operator_norm=0.8, state_bound=2.0, action_bound=1.0,
        delta_operator=0.1, delta_input=0.2, delta_output=0.05,
    )
    assert result["status"] == "bound_available"
    assert np.isfinite(result["task_error_bound"])
    assert "delta_operator" in result["unidentified_without_calibration"]


def test_same_spectrum_counterexample_changes_finite_horizon_response():
    result = same_spectrum_different_transfer_example()
    assert result["same_eigenvalues"]
    assert result["horizon_2_left"] != result["horizon_2_right"]
