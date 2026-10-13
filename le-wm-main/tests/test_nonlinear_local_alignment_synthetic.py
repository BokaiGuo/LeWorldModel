import math

import torch

from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
    NonlinearLocalConfig,
    _symmetry_matrix,
    run_one,
    true_transition,
)


def test_transition_has_declared_involutive_action_symmetry():
    generator = torch.Generator().manual_seed(3)
    state = torch.randn(16, 4, generator=generator)
    action = torch.randint(0, 2, (16,), generator=generator)
    transformed = true_transition(state, action)
    expected = true_transition(state @ _symmetry_matrix(state.device).T, 1 - action)
    assert torch.allclose(expected, transformed @ _symmetry_matrix(state.device).T, atol=1e-6)


def test_nonlinear_local_smoke_emits_functional_readouts():
    config = NonlinearLocalConfig(
        train_samples=48,
        validation_samples=32,
        test_samples=64,
        epochs=3,
        local_anchors=16,
        rollout_trajectories=12,
        planning_tasks=6,
        planning_candidates=8,
    )
    record = run_one("both", 0, config)
    assert record["device"] == "cpu"
    assert set(record["id_rollout_mse"]) == {"1", "2", "4", "8"}
    assert set(record["ood_rollout_mse"]) == {"state_shift", "action_imbalance"}
    assert math.isfinite(record["planning"]["state_shift"]["planning_regret"])
    assert math.isfinite(record["diagnostics"]["conditional_local_commutator"])


def test_random_control_reports_gradient_scale_and_anti_is_valid():
    config = NonlinearLocalConfig(
        train_samples=48,
        validation_samples=32,
        test_samples=64,
        epochs=2,
        local_anchors=16,
    )
    random_record = run_one("random", 0, config)
    anti_record = run_one("anti", 0, config)
    assert math.isfinite(random_record["random_control_gradient_scale_mean"])
    assert random_record["random_control_gradient_scale_mean"] > 0.0
    assert anti_record["alignment_weight"] == config.alignment_weight
