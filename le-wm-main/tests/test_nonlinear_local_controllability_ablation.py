import torch

from experiments.empirical_spectral_audit.run_nonlinear_local_controllability_ablation import (
    _local_precision,
)
from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
    NonlinearLocalConfig,
    _fit_model,
)


def test_local_controllability_precision_is_per_state_psd_and_trace_normalized():
    config = NonlinearLocalConfig(epochs=2, train_samples=64, validation_samples=16, test_samples=16)
    model, _train, _test, _epoch, _validation, _scale = _fit_model("none", 23, config)
    states = torch.randn(7, 4)
    matrix, diagnostics = _local_precision(model, states, horizon=4)
    assert matrix.shape == (7, 4, 4)
    assert torch.isfinite(matrix).all()
    assert torch.allclose(
        matrix.diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.full((7,), 4.0),
        atol=1e-5,
    )
    eigenvalues = torch.linalg.eigvalsh((matrix + matrix.transpose(-1, -2)) / 2)
    assert float(eigenvalues.min()) >= -1e-6
    assert diagnostics["ridge_fraction"] == 1e-2
