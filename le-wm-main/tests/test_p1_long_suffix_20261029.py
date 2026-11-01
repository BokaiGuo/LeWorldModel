import torch
import pytest
from experiments.spectral_reachability.p1_long_suffix_host_20261029 import advance, summarize


def test_history_and_cross_packet_gradient():
    z = torch.tensor([[[1.], [2.], [3.]]])
    history = torch.tensor([[[4.], [5.]]])
    first = torch.tensor([6.], requires_grad=True)
    second = torch.tensor([7.], requires_grad=True)
    def predictor(states, actions):
        return states + actions.sum(dim=1, keepdim=True)
    ends = advance(z, history, [first, second], predictor)
    assert ends[0].item() == 18
    assert ends[1].item() == 36
    grads = torch.autograd.grad(ends[1], (first, second))
    assert grads[0].item() == 2
    assert grads[1].item() == 1


def test_reject_missing_candidates():
    with pytest.raises(ValueError, match='denominator'):
        summarize({'rows': []}, [])
