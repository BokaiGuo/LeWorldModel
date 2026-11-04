"""Analytic fixture checks CEM ABI and Q orientation, not neural inference."""
import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric
from experiments.spectral_reachability.reacher_pc_q_suffix_cem import optimize


def test_common_noise_and_q_orientation():
    metric = GoalEncoderMetric(torch.diag(torch.tensor([0., 9.], dtype=torch.float64)))
    seen = []

    def objective(actions):
        seen.append(actions.clone())
        residual = actions[:, 0, :].double() - torch.tensor([0.3, -0.2])
        return metric.apply(residual.T).square().mean(dim=0)

    x, report = optimize(objective, horizon=1, action_dim=2, device="cpu", seed=8)
    seen_again = []

    def repeat(actions):
        seen_again.append(actions.clone())
        return objective(actions)

    y, other = optimize(repeat, horizon=1, action_dim=2, device="cpu", seed=8)
    assert torch.equal(x, y)
    assert torch.equal(seen[0], seen_again[0])
    assert report == other
    assert report["forward_rollout_evaluations"] == 32
    assert metric.apply(torch.tensor([1., 0.], dtype=torch.float64)).square().sum() > (
        metric.apply(torch.tensor([0., 1.], dtype=torch.float64)).square().sum())


def test_nonfinite_objective_and_invalid_elites_fail():
    with pytest.raises(ValueError, match="finite"):
        optimize(lambda x: torch.full((len(x),), float("nan")), horizon=1,
                 action_dim=2, device="cpu", seed=1)
    with pytest.raises(ValueError, match="configuration"):
        optimize(lambda x: x.sum((1, 2)), horizon=1, action_dim=2,
                 device="cpu", seed=1, samples=4, elite=1)
