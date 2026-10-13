import numpy as np

from experiments.empirical_spectral_audit.run_nonlinear_residual_aware_planner_ablation import (
    _residual_metrics,
)
from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import NonlinearWorldModel


def test_residual_metrics_are_psd_and_finite():
    model = NonlinearWorldModel(4, 16)
    state = np.random.default_rng(4).normal(size=(40, 4)).astype("float32")
    import torch

    states = torch.tensor(state)
    actions = torch.arange(40) % 2
    target = model(states, actions).detach() + 0.1 * torch.randn(40, 4)
    matrices, diagnostics = _residual_metrics(model, {"state": states, "action": actions, "target": target}, 9)
    for name, matrix in matrices.items():
        assert torch.isfinite(matrix).all()
        assert float(torch.linalg.eigvalsh((matrix + matrix.T) / 2).min()) >= -1e-6
    assert np.isfinite(diagnostics["residual_trace"])
    assert set(matrices) == {"residual_precision", "residual_covariance", "residual_random_spectral"}
