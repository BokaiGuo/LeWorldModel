import torch

from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
    NonlinearLocalConfig,
    _fit_model,
)
from experiments.empirical_spectral_audit.run_nonlinear_planner_metric_ablation import (
    _build_metric_matrices,
    _evaluate_metric,
    _make_task_batch,
    _predicted_endpoints,
)


def test_shared_task_batch_contains_a_reachable_target_and_is_deterministic():
    first = _make_task_batch(5, 7, 4, torch.Generator().manual_seed(11), 1.5)
    second = _make_task_batch(5, 7, 4, torch.Generator().manual_seed(11), 1.5)
    for key in first:
        assert torch.equal(first[key], second[key])
    assert torch.allclose(first["target_endpoint"], first["true_endpoints"][:, 0])


def test_metric_matrices_are_psd_and_random_spectral_preserves_trace():
    config = NonlinearLocalConfig(epochs=2, train_samples=48, validation_samples=16, test_samples=16)
    _model, train, _test, _epoch, _validation, _scale = _fit_model("none", 3, config)
    matrices, diagnostics = _build_metric_matrices(train, 70003)
    for matrix in matrices.values():
        assert torch.isfinite(matrix).all()
        assert float(torch.linalg.eigvalsh((matrix + matrix.T) / 2).min()) >= -1e-6
    assert torch.allclose(
        matrices["random_spectral"].trace(), matrices["covariance_precision"].trace(), atol=1e-5
    )
    assert diagnostics["metric_estimation"] == "training_targets_only"


def test_oracle_metric_has_zero_regret_on_shared_reachable_target():
    batch = _make_task_batch(8, 6, 3, torch.Generator().manual_seed(21), 0.0)
    predicted = batch["true_endpoints"].clone()
    result = _evaluate_metric(predicted, batch["true_endpoints"], batch["target_endpoint"], "oracle", None)
    assert result["planning_regret"] == 0.0
    assert result["oracle_choice_agreement"] == 1.0
    assert result["target_candidate_choice_rate"] == 1.0
