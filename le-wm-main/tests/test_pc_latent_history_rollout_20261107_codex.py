import pytest
import torch
from torch import nn
from stable_worldmodel.wm.lewm import LeWM

from experiments.pc_latent_history_rollout_20261107_codex import rollout_history


class TemporalPredictor(nn.Module):
    def forward(self, emb, actions):
        return torch.tanh(torch.cumsum(emb + actions, dim=1) * 0.2)


@pytest.mark.parametrize("prefix", [1, 2, 4])
@pytest.mark.parametrize("history_size", [1, 3])
def test_full_chain_and_prefix_gradients(prefix, history_size):
    torch.manual_seed(410)
    model = LeWM(nn.Identity(), TemporalPredictor(), nn.Identity())
    initial = torch.randn(3, 1, 4)
    actions = torch.randn(3, 5, 4, requires_grad=True)
    full = rollout_history(model, initial, actions[:, :0], actions, history_size)
    prefix_traj = rollout_history(model, initial, actions[:, :0], actions[:, :prefix], history_size)
    split = rollout_history(model, prefix_traj, actions[:, :prefix], actions[:, prefix:], history_size)
    full_grad = torch.autograd.grad(full[:, -1].square().sum(), actions, retain_graph=True)[0]
    split_grad = torch.autograd.grad(split[:, -1].square().sum(), actions)[0]
    torch.testing.assert_close(split, full)
    torch.testing.assert_close(split_grad, full_grad)
    assert full_grad[:, :prefix].abs().max() > 0


def test_reject_misaligned_histories():
    model = LeWM(nn.Identity(), TemporalPredictor(), nn.Identity())
    with pytest.raises(ValueError):
        rollout_history(model, torch.zeros(2, 3, 4), torch.zeros(2, 1, 4), torch.zeros(2, 1, 4))
    with pytest.raises(ValueError):
        rollout_history(model, torch.zeros(2, 1, 4), torch.zeros(2, 0, 4), torch.zeros(2, 0, 4))
    with pytest.raises(TypeError):
        rollout_history(nn.Identity(), torch.zeros(2, 1, 4), torch.zeros(2, 0, 4), torch.zeros(2, 1, 4))
