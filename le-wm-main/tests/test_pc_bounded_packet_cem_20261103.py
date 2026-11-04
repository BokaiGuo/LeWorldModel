import pytest
import torch

from experiments.spectral_reachability.pc_ap_spectral_cem_20261103 import spectral_cem
from experiments.spectral_reachability.pc_bounded_packet_cem_20261103 import packet_cem


def test_one_packet_matches_existing_ap_kernel_exactly():
    prior = torch.zeros((1, 1, 10))
    lower, upper = -torch.ones(10), torch.ones(10)
    calls_a, calls_b = [], []
    def scorer(calls):
        def evaluate(x):
            calls.append(x.clone())
            return (x - .31).square().sum(dim=(-1, -2))
        return evaluate
    a = spectral_cem(scorer(calls_a), prior, lower, upper, torch.Generator().manual_seed(741))
    b = packet_cem(scorer(calls_b), prior, lower, upper, torch.Generator().manual_seed(741))
    assert a[1:] == b[1:]
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    for x, y in zip(calls_a, calls_b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_two_packet_bounds_and_fifty_samples():
    counts = []
    prior = torch.zeros((1, 1, 20))
    lower, upper = -torch.full((20,), .2), torch.full((20,), .3)
    def evaluate(x):
        assert bool((x >= lower).all() and (x <= upper).all())
        counts.append(x.shape[1])
        return (x - .1).square().sum(dim=(-1, -2))
    chosen, _, _, _ = packet_cem(evaluate, prior, lower, upper, torch.Generator().manual_seed(12))
    assert counts == [16, 16, 16, 2]
    assert chosen.shape == (1, 1, 20)
    assert bool((chosen >= lower).all() and (chosen <= upper).all())


def test_rejects_invalid_domain():
    with pytest.raises(ValueError, match="domain"):
        packet_cem(lambda x: x, torch.zeros((1, 1, 20)), -torch.ones(19),
                   torch.ones(19), torch.Generator())
