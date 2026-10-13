import torch

from experiments.empirical_spectral_audit.run_nonlinear_goal_supervised_metric_ablation import (
    TASK_WEIGHTS,
    _fit_task_metric,
)
from experiments.empirical_spectral_audit.run_nonlinear_planner_metric_ablation import (
    _make_task_batch,
)


def test_goal_supervised_metric_is_positive_and_finite():
    tasks = _make_task_batch(32, 16, 4, torch.Generator().manual_seed(29), 0.0)
    matrix, diagnostics = _fit_task_metric(tasks, steps=30)
    assert matrix.shape == (4, 4)
    assert torch.isfinite(matrix).all()
    assert float(torch.linalg.eigvalsh(matrix).min()) > 0.0
    assert torch.allclose(matrix.trace(), torch.tensor(4.0), atol=1e-5)
    assert diagnostics["positive_labels"] > 0
    assert len(diagnostics["normalized_true_task_weights"]) == len(TASK_WEIGHTS)
