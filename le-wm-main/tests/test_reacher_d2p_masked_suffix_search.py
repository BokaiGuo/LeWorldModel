"""Synthetic numerical/logic tests, not learned-model inference evidence."""
import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_masked_suffix_search import search


def test_incumbent_preserved_when_already_exact():
    value,receipt = search(lambda x:x+0,torch.tensor([.25,-.5]),torch.tensor([.25,-.5]),seed=3)
    torch.testing.assert_close(value,torch.tensor([.25,-.5]),rtol=0,atol=0)
    assert receipt["best_cost"] == receipt["initial_cost"] == 0
    assert receipt["model_terminal_sample_evaluations"] == 386
    assert receipt["physical_labels_used"] is False


def test_search_is_deterministic_and_never_worse_than_incumbent():
    first,a = search(lambda x:x+0,torch.zeros(4),torch.full((4,),.75),seed=8)
    second,b = search(lambda x:x+0,torch.zeros(4),torch.full((4,),.75),seed=8)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert a == b and a["best_cost"] < a["initial_cost"]
    assert all(t["best_cost"] <= a["initial_cost"] for t in a["trace"])
    assert first.abs().max() <= 1


@pytest.mark.parametrize("value",[torch.tensor([1.1]),torch.tensor([float("nan")])])
def test_bad_incumbent_rejected(value):
    with pytest.raises(ValueError): search(lambda x:x,value,torch.zeros(1),seed=3)


def test_incomplete_predictions_rejected():
    with pytest.raises(ValueError): search(lambda x:x[:1],torch.zeros(2),torch.zeros(2),seed=3)


def test_singleton_contract_rechecks_batched_false_improvement():
    def inconsistent_kernel(x):
        # Synthetic adversarial batch-vs-singleton discrepancy.
        return torch.zeros_like(x) if len(x)>1 else x.square()+.01
    initial = torch.zeros(2)
    value,receipt = search(inconsistent_kernel,initial,torch.zeros(2),seed=8)
    torch.testing.assert_close(value,initial,rtol=0,atol=0)
    assert receipt["best_cost"] == receipt["initial_cost"]
