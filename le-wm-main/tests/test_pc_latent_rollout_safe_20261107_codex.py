from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import torch
from torch import nn

from stable_worldmodel.wm.lewm import LeWM
from experiments.pc_latent_rollout_safe_20261107_codex import rollout_single_latent


class Predictor(nn.Module):
    def forward(self, emb, act):
        return torch.tanh(emb + act)


def native(model, initial, actions):
    def encoded(info):
        return {**info, "emb": initial[None, None]}
    with patch.object(model, "encode", side_effect=encoded):
        return model.rollout({"pixels": torch.zeros(1, 1, 1, 1)}, actions[None])["predicted_emb"][0]


@pytest.mark.parametrize("horizon", [1, 2, 5, 8])
def test_native_predictions_and_action_gradients(horizon):
    torch.manual_seed(19)
    model = LeWM(nn.Identity(), Predictor(), nn.Identity())
    initial = torch.randn(4)
    actions = torch.randn(7, horizon, 4, requires_grad=True)
    reference = native(model, initial, actions)
    reference_grad = torch.autograd.grad(reference.square().sum(), actions)[0]
    result = rollout_single_latent(model, initial, actions)
    result_grad = torch.autograd.grad(result.square().sum(), actions)[0]
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(result_grad, reference_grad, rtol=0, atol=0)


def test_no_model_mutation_during_parallel_calls():
    model = LeWM(nn.Identity(), Predictor(), nn.Identity())
    actions = torch.ones(3, 4, 4)
    before = model.encode.__func__
    initials = [torch.full((4,), float(i)) for i in range(8)]
    expected = [rollout_single_latent(model, x, actions) for x in initials]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda x: rollout_single_latent(model, x, actions), initials))
    assert model.encode.__func__ is before
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cached_path_does_not_call_encoder_or_backpropagate_initial():
    model = LeWM(nn.Identity(), Predictor(), nn.Identity())
    initial = torch.ones(4, requires_grad=True)
    actions = torch.ones(2, 3, 4, requires_grad=True)
    with patch.object(model, "encode", side_effect=AssertionError("unexpected encoding")):
        out = rollout_single_latent(model, initial, actions)
    initial_grad, action_grad = torch.autograd.grad(
        out.sum(), (initial, actions), allow_unused=True
    )
    assert initial_grad is None
    assert action_grad is not None


def test_reject_empty_or_bad_dimensions():
    model = LeWM(nn.Identity(), Predictor(), nn.Identity())
    with pytest.raises(ValueError):
        rollout_single_latent(model, torch.zeros(4), torch.empty(2, 0, 4))
    with pytest.raises(ValueError):
        rollout_single_latent(model, torch.zeros(1, 4), torch.zeros(2, 1, 4))
    with pytest.raises(TypeError):
        rollout_single_latent(nn.Identity(), torch.zeros(4), torch.zeros(2, 1, 4))
