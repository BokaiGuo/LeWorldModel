import math

import torch

from experiments.empirical_spectral_audit.run_alignment_functional_synthetic import (
    FunctionalAlignmentConfig,
    _planning_proxy,
    _sample_sequence_split,
    run_one,
)
from experiments.empirical_spectral_audit.run_alignment_training_synthetic import (
    ActionConditionedLinear,
    _make_ground_truth,
)


def test_functional_alignment_emits_random_rollout_and_planning_metrics():
    config = FunctionalAlignmentConfig(
        train_samples_per_group=32,
        validation_samples_per_group=32,
        test_samples_per_group=64,
        epochs=3,
        rollout_trajectories=16,
        planning_tasks=8,
        planning_candidates=8,
    )
    record = run_one("conditional", 0.01, 0, config)

    assert record["device"] == "cpu"
    assert set(record["random_rollout_mse"]["id"]) == {"1", "2", "4", "8"}
    assert set(record["random_rollout_mse"]["ood"]) == {"1", "2", "4", "8"}
    assert set(record["planning"]) == {"id", "ood"}
    assert math.isfinite(record["planning"]["ood"]["planning_regret"])
    assert math.isfinite(record["diagnostics"]["conditional_commutator_mean"])


def test_perfect_action_conditioned_model_has_zero_planning_regret():
    config = FunctionalAlignmentConfig(
        planning_tasks=12,
        planning_candidates=10,
        planning_horizon=4,
    )
    covariances, operators = _make_ground_truth(4, torch.device("cpu"))
    model = ActionConditionedLinear(
        4, torch.Generator().manual_seed(7), torch.device("cpu")
    )
    with torch.no_grad():
        model.operators.copy_(operators)
    result = _planning_proxy(
        model,
        operators,
        config,
        torch.Generator().manual_seed(9),
        group_one_probability=0.5,
        state_shift=0.0,
    )

    assert result["oracle_goal_distance"] >= 0.0
    assert result["selected_goal_distance"] >= 0.0
    assert abs(result["planning_regret"]) < 1e-6
    assert result["oracle_choice_agreement"] == 1.0


def test_sequence_split_has_expected_shape_and_exact_transition():
    covariances, operators = _make_ground_truth(4, torch.device("cpu"))
    split = _sample_sequence_split(
        7,
        5,
        covariances,
        operators,
        torch.Generator().manual_seed(11),
    )
    assert split["states"].shape == (7, 6, 4)
    assert split["groups"].shape == (7, 5)
    expected = split["states"][:, 0]
    for step in range(5):
        expected = torch.bmm(
            operators[split["groups"][:, step]], expected.unsqueeze(-1)
        ).squeeze(-1)
        assert torch.allclose(expected, split["states"][:, step + 1])
