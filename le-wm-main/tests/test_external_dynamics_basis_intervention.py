import torch

from experiments.empirical_spectral_audit.run_external_dynamics_basis_intervention import (
    _apply_basis_intervention,
)


def test_basis_intervention_zeroes_selected_coefficients_only():
    prediction = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    basis = torch.eye(8)
    result = _apply_basis_intervention(
        prediction,
        torch.zeros(8),
        basis,
        ("covariance_basis", 0, 2),
        {"covariance_basis": basis},
    )
    assert torch.allclose(result[:, :2], torch.zeros(1, 2))
    assert torch.allclose(result[:, 2:], prediction[:, 2:])
