"""Affine algebra/protocol fixtures, not CPU neural experiments."""
import copy

import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_state_suffix as state


class Affine:
    def __init__(self):
        self.samples = 0

    def action_encoder(self, value):
        return value

    def predict(self, emb, actions):
        self.samples += emb.shape[0]
        return actions[..., :1]


def inputs():
    return dict(model=Affine(), goal_latent=torch.tensor([.25]), emb_context=torch.zeros(1, 3, 1),
        action_context=torch.zeros(1, 3, 1), reference_params=torch.full((1, 2, 1), .2),
        action_lower=torch.tensor([-1.]), action_upper=torch.tensor([1.]), seed=71)


def test_cached_state_exactly_matches_historical_six_candidate_optimizer(monkeypatch):
    kw = inputs()
    old_model = Affine()
    monkeypatch.setattr(state.p0, "_latent_context", lambda *a: kw["emb_context"])
    monkeypatch.setattr(state.p0, "_goal_latent", lambda *a: kw["goal_latent"])

    def cost_fn(model, pixels, goal, history, params, **bounds):
        return state.state_costs(model, kw["goal_latent"], kw["emb_context"], history, params, **bounds)

    monkeypatch.setattr(state.p0, "_cost_fn", cost_fn)
    old, receipt = state.historical.compile_candidates(old_model, None, None, kw["action_context"],
        incumbent=kw["reference_params"], action_lower=kw["action_lower"], action_upper=kw["action_upper"], seed=kw["seed"])
    actual, new_receipt = state.compile_state_candidates(**kw)
    assert all(torch.equal(old[name], actual[name]) for name in old)
    assert receipt["model_costs"] == new_receipt["model_costs"]
    assert receipt["receipts"] == new_receipt["receipts"]
    assert receipt["model_predict_sample_evaluations"] == new_receipt["model_predict_sample_evaluations"] == kw["model"].samples
    assert old_model.samples == kw["model"].samples


def test_exact_carried_boundary_and_accounting():
    kw = inputs()
    carried = torch.ones_like(kw["reference_params"])
    saved = carried.clone()
    variants, receipt = state.compile_state_candidates(**kw, planned_commands=carried)
    assert torch.equal(carried, saved) and torch.equal(variants["carry"], saved)
    assert abs(float(variants["carry_box_32"][0, -1, 0]) - .25) < 1e-3
    assert receipt["model_predict_sample_evaluations"] == kw["model"].samples
    assert receipt["physical_labels_used"] is False
    assert receipt["model_costs"][receipt["selected"]] <= receipt["model_costs"]["carry"]


@pytest.mark.parametrize("value", [torch.full((1, 2, 1), 1.1), torch.full((1, 2, 1), float("nan")),
    torch.zeros(1, 2, 1, dtype=torch.float64), torch.zeros(1, 1, 1), [[[0.]]]])
def test_invalid_carry_rejected_before_model_call(value):
    kw = inputs()
    with pytest.raises(ValueError):
        state.compile_state_candidates(**kw, planned_commands=value)
    assert kw["model"].samples == 0


def test_incomplete_carried_bank_rejected():
    costs = dict.fromkeys(state.historical.NAMES, 1.)
    costs["carry"] = .1
    with pytest.raises(ValueError):
        state.choose_state_candidate(costs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1])
def test_invalid_model_cost(value):
    costs = dict.fromkeys(state.historical.NAMES, 1.)
    costs["incumbent"] = value
    with pytest.raises(ValueError):
        state.choose_state_candidate(costs)
