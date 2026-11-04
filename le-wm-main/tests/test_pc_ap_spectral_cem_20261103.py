import pytest
import torch

from experiments.spectral_reachability.pc_ap_official_cem_kernel import official_cem_kernel
from experiments.spectral_reachability.pc_ap_spectral_cem_20261103 import spectral_cem


def test_no_injection_matches_official_and_injection_enters_first_batch():
    prior = torch.zeros((1, 1, 10))
    lower, upper = -torch.ones(10), torch.ones(10)
    batches_a, batches_b = [], []

    def score(store):
        def apply(x):
            store.append(x.clone())
            return (x - .15).square().sum(dim=(-1, -2))
        return apply

    official = official_cem_kernel(score(batches_a), prior, lower, upper,
                                   torch.Generator().manual_seed(81),
                                   population=16, iterations=3, elites=4)
    output = spectral_cem(score(batches_b), prior, lower, upper,
                          torch.Generator().manual_seed(81))
    assert len(batches_a) == len(batches_b) == 4
    for a, b in zip(batches_a, batches_b, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(official.normalized_block, output[0], rtol=0, atol=0)
    assert (official.initial_cost, official.refined_cost, official.refined_selected) == output[1:]
    injected = torch.full((4, 10), .4)
    batches = []
    spectral_cem(score(batches), prior, lower, upper, torch.Generator().manual_seed(81), injected)
    torch.testing.assert_close(batches[0][0, 1:5, 0], injected)
    torch.testing.assert_close(batches[0][0, 0, 0], prior[0, 0])
    with pytest.raises(ValueError):
        spectral_cem(score([]), prior, lower, upper, torch.Generator(), injected * 5)


def test_projected_elites_fit_the_packets_actually_scored():
    prior = torch.zeros((1, 1, 10))
    lower, upper = -torch.full((10,), .1), torch.full((10,), .1)
    batches = []

    def evaluate(x):
        batches.append(x.clone())
        # A dimension-dependent objective avoids ties between clipped packets.
        return -(x * torch.arange(1, 11)).sum(dim=(-1, -2))

    spectral_cem(evaluate, prior, lower, upper, torch.Generator().manual_seed(3),
                 population=16, iterations=1, elites=4,
                 projected_elite_update=True)
    first, final = batches
    indices = torch.topk(evaluate(first), 4, dim=1, largest=False).indices
    expected = first[torch.arange(1)[:, None], indices].mean(dim=1)
    torch.testing.assert_close(final[:, 1], expected, rtol=0, atol=1e-7)
    assert ((first == lower) | (first == upper)).any()


def test_projected_update_is_explicit_and_validated():
    prior = torch.zeros((1, 1, 10))
    with pytest.raises(ValueError, match="must be boolean"):
        spectral_cem(lambda x: x.sum(dim=(-1, -2)), prior, -torch.ones(10),
                     torch.ones(10), torch.Generator(), projected_elite_update="yes")
