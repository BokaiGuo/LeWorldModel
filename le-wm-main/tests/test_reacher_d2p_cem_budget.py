"""Microbatch algebra only, not a CPU neural experiment."""
import pytest
import torch

from experiments.spectral_reachability.audit_reacher_d2p_cem_budget import microbatched_cost


def test_fixed_microbatch_preserves_every_candidate_in_order():
    params = torch.arange(530., dtype=torch.float64).reshape(265, 1, 2)
    calls = []
    def cost(p):
        calls.append(len(p))
        return p.square().sum((1, 2))
    value = microbatched_cost(cost, params)
    assert torch.equal(value, params.square().sum((1, 2)))
    assert calls == [128, 128, 9]


@pytest.mark.parametrize("bad", ["shape", "nan", "negative"])
def test_invalid_costs_fail_closed(bad):
    def cost(p):
        if bad == "shape":
            return torch.zeros(len(p), 2)
        return torch.full((len(p),), float("nan") if bad == "nan" else -1.)
    with pytest.raises(ValueError):
        microbatched_cost(cost, torch.zeros(2, 1, 1))
