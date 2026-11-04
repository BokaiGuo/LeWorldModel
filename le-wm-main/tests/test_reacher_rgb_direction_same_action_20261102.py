import pytest
import torch

from experiments.spectral_reachability import reacher_rgb_direction_same_action_20261102 as check


class AnalyticModel:
    def encode(self, values):
        return {"emb": values["pixels"]}

    def action_encoder(self, actions):
        return actions

    def predict(self, embeddings, actions):
        return embeddings + actions[..., :2]


def test_direct_rollout_matches_existing_bounded_rollout():
    model = AnalyticModel()
    pixels = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    history = torch.zeros(1, 3, 10)
    params = torch.linspace(-1, 1, 40).reshape(2, 2, 10)
    lower, upper = torch.full((10,), -1.0), torch.full((10,), 1.0)
    normalized = check.setup.p0._bounded_normalized_actions(params, lower, upper)
    actual = check.direct_terminal(model, pixels, history, normalized)
    expected = check.setup.p0._terminal_batch(model, pixels, pixels, history, params,
                                               action_lower=lower, action_upper=upper)
    assert torch.equal(actual, expected)


def test_direct_rejects_bad_history():
    with pytest.raises(ValueError):
        check.direct_terminal(AnalyticModel(), torch.zeros(1, 3, 2), torch.zeros(1, 2, 10),
                              torch.zeros(2, 2, 10))
