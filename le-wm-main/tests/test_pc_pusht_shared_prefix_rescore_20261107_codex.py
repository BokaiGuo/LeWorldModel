import pytest
import torch

from experiments.pc_pusht_shared_prefix_rescore_20261107_codex import conditional_rollout


class Model:
    def __init__(self):
        self.calls = []

    def action_encoder(self, actions):
        return actions

    def predict(self, states, actions):
        self.calls.append((states.clone(), actions.clone()))
        return states + actions


def test_common_observations_and_actions_stay_clock_aligned():
    model = Model()
    obs = torch.tensor([[0.], [1.], [3.]])
    actions = torch.tensor([[[1.], [2.], [4.], [8.], [16.]],
                            [[1.], [2.], [5.], [10.], [20.]]])
    pred = conditional_rollout(model, obs, actions, 2)
    torch.testing.assert_close(pred[:, :, 0], torch.tensor([[7., 15., 31.], [8., 18., 38.]]))
    assert [x.shape[1] for x, _ in model.calls] == [3, 3, 3]
    torch.testing.assert_close(model.calls[1][0][0, :, 0], torch.tensor([1., 3., 7.]))
    torch.testing.assert_close(model.calls[1][1][0, :, 0], torch.tensor([2., 4., 8.]))


def test_short_context_and_clock_validation():
    model = Model()
    result = conditional_rollout(model, torch.tensor([[3.]]), torch.tensor([[[4.], [8.], [16.]]]), 0)
    torch.testing.assert_close(result[0, :, 0], torch.tensor([7., 15., 31.]))
    assert [x.shape[1] for x, _ in model.calls] == [1, 2, 3]
    with pytest.raises(ValueError):
        conditional_rollout(model, torch.zeros(2, 1), torch.zeros(1, 5, 1), 2)
