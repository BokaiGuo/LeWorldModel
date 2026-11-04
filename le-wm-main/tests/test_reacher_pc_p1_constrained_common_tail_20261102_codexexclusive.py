import pytest
import torch

from experiments.spectral_reachability import reacher_pc_p1_constrained_common_tail_20261102_codexexclusive as search


def feasible(x):
    return x[0], torch.stack([-x[1], -2*x[1], -3*x[1], -4*x[1], -5*x[1]])


def test_fixed_budget_discovers_analytic_feasible_without_stopping_early():
    calls = []
    def evaluate(x):
        calls.append(x.detach().clone())
        return feasible(x)
    bank, receipt = search.generate(torch.zeros(10, dtype=torch.float64), evaluate)
    assert bank.shape == (21, 10)
    assert len(calls) == receipt["logical_costs"]["evaluation_calls"] == 21
    assert receipt["logical_costs"]["prefix_gradient_vjp_calls"] == 16
    assert receipt["logical_costs"]["suffix_predict_samples"] == 210
    assert len(receipt["optimization_trace"]) == 20
    assert receipt["selected_proposal_index"] is not None
    assert receipt["proposal_rows"][receipt["selected_proposal_index"]]["eligible"]
    assert torch.linalg.vector_norm(bank, dim=1).max() <= search.RADIUS + 1e-12
    bank2, receipt2 = search.generate(torch.zeros(10, dtype=torch.float64), feasible)
    assert torch.equal(bank, bank2)
    assert receipt == receipt2


def test_strict_rule_matches_previous_selector_on_same_proposals():
    origin = torch.zeros(10, dtype=torch.float64)
    old_bank, old = search.prior.generate(origin, feasible)
    base_now, base_final = feasible(origin)
    for i, old_row in enumerate(old["proposal_rows"]):
        now, final = feasible(old_bank[i])
        reconstructed = search.row(i, old_bank[i], origin, now, final, base_now, base_final, "test")
        assert reconstructed["eligible"] == old_row["eligible"]
        assert reconstructed["per_tail_delta"] == old_row["per_tail_delta"]
        assert reconstructed["active_tail_neutral_adjusted_delta"] == old_row["active_tail_neutral_adjusted_delta"]


def test_infeasible_returns_none_without_any_fallback():
    calls = []
    def no_future_effect(x):
        calls.append(1)
        return x[0], torch.stack([0*x[1]] * 5)
    bank, receipt = search.generate(torch.zeros(10, dtype=torch.float64), no_future_effect)
    assert len(calls) == 21 and bank.shape[0] == 21
    assert receipt["selected_proposal_index"] is None
    assert not any(row["eligible"] for row in receipt["proposal_rows"])


def test_single_worsened_tail_or_adjusted_tail_never_admitted():
    z = torch.tensor(0., dtype=torch.float64)
    base = torch.zeros(5, dtype=torch.float64)
    origin = torch.zeros(10, dtype=torch.float64)
    point = origin.clone()
    point[0] = .1
    one_worse = torch.tensor([-.1, -.1, -.1, -.1, .01], dtype=torch.float64)
    adjusted_worse = torch.tensor([-.2, -.1, -.2, -.2, -.2], dtype=torch.float64)
    assert not search.row(1, point, origin, point[0], one_worse, z, base, "x")["eligible"]
    assert not search.row(1, point, origin, point[0], adjusted_worse, z, base, "x")["eligible"]


def test_nonfinite_and_cpu_neural_fallback_rejected():
    def nan_later(x):
        return x[0], torch.stack([x[1]*float("nan")] * 5)
    with pytest.raises(ValueError, match="invalid"):
        search.generate(torch.zeros(10), nan_later)
    model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="host CUDA"):
        search.compile_reacher(model, torch.zeros(1), torch.zeros(1), torch.zeros(1, 3, 10),
                               None, lower=torch.zeros(10), upper=torch.ones(10))
