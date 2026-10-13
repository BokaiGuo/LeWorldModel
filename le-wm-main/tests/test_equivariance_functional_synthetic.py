import torch

from experiments.empirical_spectral_audit.run_equivariance_functional_synthetic import (
    FunctionalBenchmarkConfig,
    _planning_proxy,
)


class _PerfectAdditiveModel:
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return torch.cat((state, torch.zeros_like(state)), dim=-1)

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat((latent[:, :4] + action, latent[:, 4:]), dim=-1)


def test_candidate_set_planning_proxy_recovers_oracle_for_perfect_predictor():
    config = FunctionalBenchmarkConfig(
        planning_tasks=8,
        planning_candidates=8,
        planning_horizon=3,
        planning_success_threshold=0.25,
    )
    metrics = _planning_proxy(
        _PerfectAdditiveModel(),
        config,
        torch.Generator().manual_seed(123),
        action_scale=0.25,
        state_shift=0.0,
    )

    assert metrics["oracle_goal_distance"] < 1e-6
    assert metrics["selected_goal_distance"] < 1e-6
    assert metrics["planning_regret"] < 1e-6
    assert metrics["goal_success_rate"] == 1.0
    assert metrics["oracle_success_rate"] == 1.0
    assert metrics["oracle_choice_agreement"] == 1.0
