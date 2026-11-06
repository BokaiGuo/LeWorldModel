import torch

from experiments.pc_explicit_prefix_equal_replan_dev_20261106_codex import model_cost


class FakeModel:
    def get_cost(self, info, actions):
        assert set(info) == {"pixels", "goal", "action"}
        assert info["pixels"].shape[:3] == (1, actions.shape[1], 1)
        assert info["goal"].shape == info["pixels"].shape
        return actions.square().sum((-1, -2))


def test_model_cost_uses_only_legal_observation_and_goal():
    pixels = torch.zeros(3, 8, 8)
    candidates = torch.ones(3, 5, 10)
    assert torch.equal(model_cost(FakeModel(), pixels, pixels, candidates), torch.full((3,), 50.0))
