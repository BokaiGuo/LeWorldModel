import numpy as np
import pytest

from experiments.theory_spectral_transfer.finite_horizon_transfer import (
    finite_horizon_transfer_risk_decomposition,
    spectrum_alignment_commutator_counterexample,
    transfer_bound_tightness_example,
)


def test_decomposed_bound_components_sum_to_task_bound():
    report = finite_horizon_transfer_risk_decomposition(
        A=np.asarray([[0.8, 0.2], [0.0, 0.7]]),
        D=np.eye(2),
        horizon=4,
        closure_residuals=[0.1] * 4,
        off_manifold_residuals=[0.2] * 4,
        action_effect_errors=[0.05] * 4,
        readout_error=0.03,
        readout_state_bound=2.0,
        initial_error=0.1,
    )
    components = report["components"]
    assert np.isfinite(report["task_error_bound"])
    assert np.isclose(report["task_error_bound"], sum(components.values()))
    assert report["estimability"]["off_manifold"].startswith("independent")


def test_scalar_positive_system_is_tight():
    report = transfer_bound_tightness_example()
    assert report["tight"] is True
    assert report["bound"]["components"]["E_transient"] == 0.0


def test_normal_operator_need_not_saturate_task_readout_spectral_gain():
    """A normal A only implies g_j <= ||D|| rho(A)^j, not equality."""

    report = finite_horizon_transfer_risk_decomposition(
        A=np.diag([0.9, 0.1]),
        D=np.asarray([[0.0, 1.0]]),
        horizon=2,
        closure_residuals=[0.1, 0.1],
        off_manifold_residuals=[0.0, 0.0],
        action_effect_errors=[0.0, 0.0],
        readout_error=0.0,
        readout_state_bound=0.0,
    )
    assert report["finite_horizon_gains"][1] == pytest.approx(0.1)
    assert report["spectral_radius_gains"][1] == pytest.approx(0.9)
    assert report["components"]["E_transient"] == 0.0


def test_same_spectrum_alignment_commutator_does_not_identify_transfer():
    report = spectrum_alignment_commutator_counterexample()
    assert report["same_spectrum"] is True
    assert report["same_covariance_predictor_commutator"] is True
    assert report["different_transfer"] is True


def test_nonnegative_residual_lengths_are_required():
    with pytest.raises(ValueError):
        finite_horizon_transfer_risk_decomposition(
            A=np.eye(2), D=np.eye(2), horizon=2,
            closure_residuals=[0.1], off_manifold_residuals=[0.0, 0.0],
            action_effect_errors=[0.0, 0.0], readout_error=0.0,
            readout_state_bound=1.0,
        )
