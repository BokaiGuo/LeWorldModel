import torch

from experiments.empirical_spectral_audit.run_nonlinear_controllability_metric_ablation import (
    _controllability_precision,
    _fit_linear_transition,
)
from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
    NonlinearLocalConfig,
    _fit_model,
)


def test_controllability_precision_is_finite_psd_and_horizon_sensitive():
    config = NonlinearLocalConfig(epochs=2, train_samples=64, validation_samples=16, test_samples=16)
    _model, train, _test, _epoch, _validation, _scale = _fit_model("none", 17, config)
    A, B = _fit_linear_transition(train)
    short, short_info = _controllability_precision(A, B, 2)
    long, long_info = _controllability_precision(A, B, 8)
    for matrix in (short, long):
        assert torch.isfinite(matrix).all()
        assert float(torch.linalg.eigvalsh((matrix + matrix.T) / 2).min()) >= -1e-6
        assert torch.allclose(matrix.trace(), torch.tensor(4.0), atol=1e-5)
    assert short_info["horizon"] == 2
    assert long_info["horizon"] == 8
    assert short_info["gramian_trace"] != long_info["gramian_trace"]
