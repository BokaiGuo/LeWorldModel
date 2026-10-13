import math

import torch

from experiments.empirical_spectral_audit.run_alignment_training_synthetic import (
    BenchmarkConfig,
    _make_ground_truth,
    _population_geometry,
    _sample_split,
    run_one,
)


def test_synthetic_geometry_is_locally_commuting_but_globally_heterogeneous():
    covariances, operators = _make_ground_truth(4, torch.device("cpu"))
    geometry = _population_geometry(covariances, operators)

    assert max(geometry["local_commutator_by_group"]) < 1e-8
    assert geometry["pooled_commutator"] > 1e-2


def test_synthetic_training_emits_prediction_rollout_and_group_metrics():
    config = BenchmarkConfig(
        train_samples_per_group=32,
        validation_samples_per_group=32,
        test_samples_per_group=64,
        epochs=2,
    )
    result = run_one("conditional", 0.01, 0, config, torch.device("cpu"))

    assert math.isfinite(result["test_one_step_mse"])
    assert set(result["rollout_mse"]) == {"1", "2", "4", "8"}
    assert set(result["action_composition_error"]) == {"0->0", "0->1", "1->0", "1->1"}
    assert all(math.isfinite(value) for value in result["action_composition_error"].values())
    assert set(result["test_one_step_mse_by_group"]) == {"0", "1"}
    assert math.isfinite(result["diagnostics"]["conditional_commutator_mean"])
