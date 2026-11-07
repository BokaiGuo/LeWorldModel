import pytest
import torch
from torch import nn

from experiments.pc_rgb_shared_encoder_cache_20261107_codex import CachedJEPA


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.eval()

    def encode(self, info):
        return {"emb": info["pixels"].mean(dim=(-1, -2)).float()}

    def action_encoder(self, actions):
        return actions

    def predict(self, emb, act):
        return emb + act

    def criterion(self, info):
        return ((info["predicted_emb"][:, :, -1] - info["goal_emb"][:, -1, None]) ** 2).sum(-1)


def inputs():
    return {"pixels": torch.zeros((1, 1, 1, 3, 2, 2)),
            "goal": torch.ones((1, 1, 1, 3, 2, 2)),
            "action": torch.empty((1, 1, 0, 3))}


def test_cache_invalidates_changed_inputs_and_honors_values():
    model, info = CachedJEPA(FakeModel()), inputs()
    actions = torch.ones((1, 2, 2, 3))
    with torch.inference_mode():
        first = model.get_cost(dict(info), actions)
        model.get_cost(dict(info), actions)
        assert model.encoder_calls == 2
        assert first.shape == (1, 2)
        info["pixels"][0, 0, 0, 0, 0, 0] = 1
        model.get_cost(dict(info), actions)
        assert model.encoder_calls == 4
        info["goal"] = torch.zeros_like(info["goal"])
        model.get_cost(dict(info), actions)
        assert model.encoder_calls == 6


def test_cache_refuses_autograd_mode():
    with pytest.raises(ValueError, match="inference"):
        CachedJEPA(FakeModel()).get_cost(inputs(), torch.ones((1, 2, 2, 3)))
