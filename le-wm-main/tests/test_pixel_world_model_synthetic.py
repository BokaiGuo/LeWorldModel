import math

import torch

from experiments.empirical_spectral_audit.run_pixel_world_model_synthetic import (
    PixelBenchmarkConfig,
    run_one,
)


def test_pixel_world_model_emits_rollout_composition_and_geometry_metrics():
    config = PixelBenchmarkConfig(
        train_samples=16,
        validation_samples=8,
        test_samples=16,
        rollout_trajectories=8,
        epochs=2,
        sigreg_num_proj=8,
    )
    result = run_one("conditional", 0.01, 0, config)

    assert math.isfinite(result["test_one_step_mse"])
    assert set(result["rollout_mse"]) == {"1", "2", "4", "8"}
    assert len(result["action_composition_error"]) == 16
    assert all(math.isfinite(value) for value in result["action_composition_error"].values())
    assert math.isfinite(result["diagnostics"]["global_commutator"])
    assert math.isfinite(result["diagnostics"]["conditional_commutator"])
    assert result["diagnostics"]["stable_rank"] > 0
