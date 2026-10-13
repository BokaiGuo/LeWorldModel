import numpy as np
import torch

from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import NonlinearWorldModel
from experiments.empirical_spectral_audit.run_nonlinear_residual_strength_calibration_audit import (
    STRENGTHS,
    _matrix_diagnostics,
    _residual_precision,
    _strength_matrices,
)


def test_strength_control_matches_spectrum_and_condition_number():
    precision = torch.diag(torch.tensor([0.25, 0.5, 1.0, 2.25]))
    generator = torch.Generator().manual_seed(12)
    basis, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
    randomized = basis @ precision @ basis.T
    matrices = _strength_matrices(precision, randomized)
    for strength in STRENGTHS:
        diagnostics = _matrix_diagnostics(matrices[f"aligned_a{strength:g}"], matrices[f"random_basis_a{strength:g}"])
        assert diagnostics["same_spectrum_max_abs_error"] < 1e-5
        assert abs(diagnostics["aligned_condition_number"] - diagnostics["random_condition_number"]) < 1e-4
        assert abs(diagnostics["aligned_trace"] - 4.0) < 1e-5
        assert abs(diagnostics["random_trace"] - 4.0) < 1e-5


def test_heldout_residual_precision_is_finite_and_psd():
    model = NonlinearWorldModel(4, 16)
    generator = torch.Generator().manual_seed(5)
    state = torch.randn(64, 4, generator=generator)
    action = (torch.rand(64, generator=generator) < 0.5).long()
    target = model(state, action).detach() + 0.1 * torch.randn(64, 4, generator=generator)
    matrix, diagnostics = _residual_precision(model, {"state": state, "action": action, "target": target}, 19)
    assert torch.isfinite(matrix).all()
    assert float(torch.linalg.eigvalsh((matrix + matrix.T) / 2).min()) >= -1e-6
    assert diagnostics["residual_count"] == 64
    assert np.isfinite(diagnostics["matrix_condition_number"])
