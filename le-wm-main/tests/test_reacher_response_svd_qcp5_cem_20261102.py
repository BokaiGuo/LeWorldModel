import pytest
import torch

from experiments.spectral_reachability import reacher_response_svd_qcp5_cem_20261102 as qcem
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as original


def test_common_cem_sampling_matches_original(monkeypatch):
    target = torch.tensor([0.3, -0.4])
    calls = []

    def cost(candidates):
        calls.append(candidates.detach().clone())
        return (candidates - target).square().mean(dim=(1, 2))

    monkeypatch.setattr(original, "_cost_fn", lambda *args, **kw: cost(args[4]))
    history = torch.zeros((1, 3, 2))
    old, old_receipt = original._cem(None, None, None, history, future_horizon=2,
                                     action_lower=None, action_upper=None,
                                     samples=16, steps=2, elite=4, seed=61)
    old_calls = calls[:]
    calls.clear()
    new, receipt = qcem.cem_search(cost, dimension=2, device="cpu", future_horizon=2,
                                   samples=16, steps=2, elite=4, seed=61)
    assert torch.equal(new, old)
    assert all(torch.equal(a, b) for a, b in zip(old_calls, calls, strict=True))
    assert receipt["trace"] == old_receipt["trace"]
    assert receipt["forward_rollout_evaluations"] == 32


def test_invalid_scores_fail_closed():
    with pytest.raises(ValueError, match="invalid Q CEM scores"):
        qcem.cem_search(lambda x: torch.full((len(x),), float("nan")),
                        dimension=2, device="cpu", future_horizon=2,
                        samples=16, steps=2, elite=4, seed=1)
