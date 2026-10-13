import torch

from experiments.empirical_spectral_audit.run_external_energy_matched_dynamics_intervention import (
    _apply_energy_matched_intervention,
    _orthonormal_basis,
)


def test_energy_matched_control_has_reference_perturbation_energy():
    prediction = torch.arange(1.0, 17.0).reshape(1, 16)
    mean = torch.zeros(16)
    target_basis = torch.eye(16)
    control_basis = torch.roll(torch.eye(16), shifts=4, dims=1)
    result, diagnostics = _apply_energy_matched_intervention(
        prediction, mean, control_basis, target_basis, matched=True
    )
    reference = torch.square(prediction[:, :8]).sum().item()
    applied = torch.square(result - prediction).sum().item()
    assert abs(applied - reference) < 1e-4
    assert abs(diagnostics["applied_energy"] - diagnostics["reference_energy"]) < 1e-4


def test_random_basis_is_orthonormal_and_seeded():
    first = _orthonormal_basis(8, 42)
    second = _orthonormal_basis(8, 42)
    assert torch.allclose(first, second)
    assert torch.allclose(first.T @ first, torch.eye(8), atol=1e-6)
