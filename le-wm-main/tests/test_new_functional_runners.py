import math

import torch

from experiments.empirical_spectral_audit.run_joint_functional_synthetic import run_one as run_joint_one
from experiments.empirical_spectral_audit.run_pixel_nuisance_ood_synthetic import (
    PixelNuisanceConfig,
    _make_one_step_split,
    run_one as run_pixel_nuisance_one,
)
from experiments.empirical_spectral_audit.run_equivariance_functional_synthetic import (
    FunctionalBenchmarkConfig,
)


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def test_pixel_nuisance_split_has_matched_transition_shapes():
    config = PixelNuisanceConfig()
    split = _make_one_step_split(12, config, torch.Generator().manual_seed(3), 0.30)
    assert split["source"].shape == (12, 1, config.image_size, config.image_size)
    assert split["target"].shape == split["source"].shape
    assert split["actions"].shape == (12, 2)
    assert torch.isfinite(split["source"]).all()
    assert torch.isfinite(split["target"]).all()


def test_joint_functional_smoke_emits_finite_functional_readouts():
    config = FunctionalBenchmarkConfig(
        train_epochs=2,
        train_samples=32,
        validation_samples=16,
        test_samples=32,
        rollout_trajectories=8,
        planning_tasks=4,
        planning_candidates=4,
    )
    record = run_joint_one("both", 0, config, predictor_weight=1.0, alignment_weight=0.003)
    assert _finite(record)
    assert "planning" in record and "diagnostics" in record


def test_pixel_nuisance_smoke_emits_tangent_and_ood_readouts():
    config = PixelNuisanceConfig(
        epochs=2,
        train_samples=32,
        validation_samples=16,
        test_samples=32,
        rollout_trajectories=8,
    )
    record = run_pixel_nuisance_one("conditional", 0.01, 0, config)
    assert _finite(record)
    assert "ood_tangent_diagnostics" in record
    assert record["ood_one_step_mse"] >= 0.0
