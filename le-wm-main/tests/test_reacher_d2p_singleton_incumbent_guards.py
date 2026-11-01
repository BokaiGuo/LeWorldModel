"""Adversarial CPU algebra only; these fixtures are not neural experiments."""
import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_masked_programs import flat_cem_control
from experiments.spectral_reachability.reacher_d2p_masked_suffix_search import search


def test_flat_cem_cannot_replace_gn_with_worse_singleton_deployment_score():
    def inconsistent(x):
        return x if len(x)==1 else torch.zeros_like(x)
    initial,target = torch.ones(2),torch.zeros(2)
    action,prediction,receipt = flat_cem_control(inconsistent,initial,initial,target,seed=13)
    assert float(prediction.square().mean()) <= 1.
    assert receipt["singleton_incumbent_guard"] is True


def test_flat_cem_falls_back_to_exact_original_on_adversarial_batch_scoring():
    def inconsistent(x):
        return x+10 if len(x)==1 else torch.zeros_like(x)
    initial,target = torch.ones(2),torch.zeros(2)
    action,prediction,receipt = flat_cem_control(inconsistent,initial,initial,target,seed=13)
    torch.testing.assert_close(action,initial,rtol=0,atol=0)
    torch.testing.assert_close(prediction,initial,rtol=0,atol=0)
    assert receipt["singleton_fallback"] is True


@pytest.mark.parametrize("bad",["nan","shape","overflow"])
def test_invalid_initial_suffix_prediction_fails_closed(bad):
    calls = 0
    def callback(x):
        nonlocal calls
        calls += 1
        if calls==1:
            if bad=="nan": return torch.full((1,2),float("nan"))
            if bad=="shape": return torch.zeros(2)
            return torch.full((1,2),1e30)
        return x
    with pytest.raises(ValueError): search(callback,torch.zeros(2),torch.ones(2),seed=13)
    assert calls==1


@pytest.mark.parametrize("bad",["nan","shape","overflow"])
def test_invalid_final_suffix_prediction_fails_closed(bad):
    calls = 0
    def callback(x):
        nonlocal calls
        calls += 1
        if calls==5:
            if bad=="nan": return torch.full((1,2),float("nan"))
            if bad=="shape": return torch.zeros(2)
            return torch.full((1,2),1e30)
        return x
    with pytest.raises(ValueError): search(callback,torch.zeros(2),torch.ones(2),seed=13)


@pytest.mark.parametrize("bad",["initial","batch","singleton"])
def test_flat_control_bad_predictions_rejected(bad):
    def callback(x):
        if bad=="batch" and len(x)==128: return torch.zeros(1,2)
        if bad=="singleton" and len(x)==1: return torch.full((1,2),float("nan"))
        return x
    initial = torch.zeros(2)
    prediction = torch.full((2,),float("nan")) if bad=="initial" else initial
    with pytest.raises(ValueError): flat_cem_control(callback,initial,prediction,torch.ones(2),seed=13)
