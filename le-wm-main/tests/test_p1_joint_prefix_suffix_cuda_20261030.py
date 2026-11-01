import pytest
import torch

from experiments.spectral_reachability import p1_joint_prefix_suffix_cuda_20261030 as module


def test_joint_updates_all_packets_and_retains_best():
    initial = torch.zeros((3, 5, 2))

    def analytic(action):
        loss = (action - 0.5).square().mean()
        return loss, [action[i].reshape(-1) for i in range(3)]

    result = module.optimize(initial, analytic)
    assert result["best_iteration"] == module.UPDATES
    assert result["best_score"] < result["trajectory"][0]["score"]
    assert all(torch.tensor(result["best_actions"])[i].abs().sum() > 0 for i in range(3))


def test_invalid_action_fails_closed():
    with pytest.raises(ValueError, match="initial action"):
        module.optimize(torch.full((3, 5, 2), 1.1), lambda x: (x.sum(), [x] * 3))
