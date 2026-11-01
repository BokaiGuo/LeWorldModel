import pytest
import torch

from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.reacher_d2p_cem_incumbent import search_with_incumbent


def run(cost, steps=3):
    return search_with_incumbent(cost, future_horizon=2, action_dim=3, device=torch.device("cpu"),
        samples=8, steps=steps, elite=2, seed=8)


def test_best_ever_sample_cannot_be_discarded_by_a_worse_last_generation():
    calls = []

    def cost(candidates):
        calls.append(candidates.clone())
        values = torch.arange(8, dtype=torch.float32) + 10 * (len(calls) - 1)
        return values

    best, legacy, receipt = run(cost)
    assert receipt["best_ever_goal_cost"] == 0
    assert receipt["legacy_last_iteration_goal_cost"] == 20
    assert receipt["selected_iteration"] == 0
    assert len(calls) == 3
    assert torch.equal(best, calls[0][:1])
    assert torch.equal(legacy, calls[-1][:1])


def test_legacy_output_and_full_sampling_trajectory_are_preserved(monkeypatch):
    old_banks, new_banks = [], []

    def record(storage):
        def cost(candidates):
            storage.append(candidates.clone())
            return (candidates - .4).square().sum(dim=(1, 2))
        return cost

    old_cost, new_cost = record(old_banks), record(new_banks)
    monkeypatch.setattr(p0, "_cost_fn", lambda _m, _p, _g, _h, params, **_kw: old_cost(params))
    history = torch.zeros((1, 3, 3))
    old, old_receipt = p0._cem(None, None, None, history, future_horizon=2,
        action_lower=torch.zeros(3), action_upper=torch.ones(3), samples=8, steps=3, elite=2, seed=8)
    best, legacy, receipt = run(new_cost)
    assert torch.equal(old, legacy)
    assert old_receipt["trace"] == receipt["trace"]
    assert all(torch.equal(left, right) for left, right in zip(old_banks, new_banks, strict=True))
    assert receipt["best_ever_goal_cost"] <= receipt["legacy_last_iteration_goal_cost"]
    assert receipt["selection_uses_physical_labels"] is False


@pytest.mark.parametrize("bad", [torch.full((8,), float("nan")), torch.zeros(7)])
def test_invalid_model_scores_fail_closed(bad):
    with pytest.raises(ValueError):
        run(lambda _c: bad)


def test_invalid_elite_count_is_rejected():
    with pytest.raises(ValueError):
        search_with_incumbent(lambda _c: torch.zeros(8), future_horizon=2, action_dim=3,
            device=torch.device("cpu"), samples=8, steps=3, elite=1, seed=8)
