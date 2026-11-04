import pytest
import torch

from experiments.spectral_reachability.replay_reacher_constrained_goal_q_cuda import score_endpoints


class Identity:
    def apply(self, value):
        return value


def test_goal_q_batched_matches_manual_costs():
    endpoints = torch.arange(21 * 5 * 2, dtype=torch.float64).reshape(21, 5, 2) / 10
    target = torch.tensor([0.25, -0.5], dtype=torch.float64)
    actual = score_endpoints(Identity(), endpoints, target)
    assert torch.allclose(actual, (endpoints - target).square().mean(dim=-1))


def test_rejects_wrong_candidate_count():
    with pytest.raises(ValueError, match="shape"):
        score_endpoints(Identity(), torch.zeros((20, 5, 2)), torch.zeros(2))
