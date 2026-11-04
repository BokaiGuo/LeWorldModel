import torch
import pytest

from experiments.spectral_reachability.reacher_pc_cem_incumbent_audit import cem_incumbent


def test_retains_early_best_without_extra_evaluations():
    calls = []

    def costs(candidates):
        calls.append(candidates.clone())
        return torch.arange(len(candidates), dtype=torch.float32) + (100 if len(calls) > 1 else 0)

    best, final, receipt = cem_incumbent(costs, horizon=3, action_dim=2, device="cpu",
                                         samples=4, steps=3, elite=2, seed=4)
    assert len(calls) == 3
    assert torch.equal(best, calls[0][:1])
    assert torch.equal(final, calls[-1][:1])
    assert receipt["best_iteration"] == 0
    assert receipt["best_cost"] == 0
    assert receipt["predict_samples_horizon_weighted"] == 36


def test_rejects_nonfinite_costs():
    with pytest.raises(ValueError):
        cem_incumbent(lambda x: torch.full((len(x),), float("nan")), horizon=1,
                      action_dim=1, device="cpu", samples=4, steps=1, elite=2, seed=0)
