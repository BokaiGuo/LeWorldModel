import pytest
import torch

from experiments.pc_tworoom_real_vs_imagined_feedback_20261107_codex import latent_rollout


class MockModel:
    def encode(self, info):
        raise AssertionError("latent injection was not used")

    def rollout(self, info, actions):
        initial = self.encode(info)["emb"][0, 0]
        states = [initial[None].expand(actions.shape[1], -1)]
        for step in range(actions.shape[2]):
            states.append(states[-1] + actions[0, :, step, :1])
        return {"predicted_emb": torch.stack(states, dim=1)[None]}


def test_latent_rollout_uses_actions_in_order():
    initial = torch.tensor([1.0])
    actions = torch.zeros((2, 3, 10))
    actions[:, :, 0] = torch.tensor([2.0, 3.0, 4.0])
    result = latent_rollout(MockModel(), initial, actions)
    assert result[0, :, 0].tolist() == [1.0, 3.0, 6.0, 10.0]


def test_invalid_clock_fails_closed():
    with pytest.raises(ValueError):
        latent_rollout(MockModel(), torch.zeros(1), torch.zeros(2, 6, 10))
