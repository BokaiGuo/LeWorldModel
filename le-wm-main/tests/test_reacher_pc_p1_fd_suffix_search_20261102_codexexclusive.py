import pytest
import torch

from experiments.spectral_reachability import reacher_pc_p1_fd_suffix_search_20261102_codexexclusive as search


def case(x):
    return x[0], torch.stack([-x[1], -2*x[1], -3*x[1], -4*x[1], -5*x[1]])


def test_fixed_budget_repeatable_and_separate_contracts():
    calls = []
    def evaluate(x):
        calls.append(x.clone())
        return case(x)
    origin = torch.zeros(10, dtype=torch.float64)
    bank, receipt = search.generate(origin, evaluate)
    again, second = search.generate(origin, case)
    assert torch.equal(bank, again) and receipt == second
    assert bank.shape == (9, 10) and len(calls) == 29
    assert len(receipt["fd_probes"]) == 10
    assert receipt["logical_costs"]["suffix_token_predict_samples"] == 290
    assert receipt["development_selected_index"] is not None
    assert receipt["strict_selected_index"] is not None
    for axis in range(10):
        expected = torch.zeros_like(origin)
        expected[axis] = search.FD_STEP
        assert torch.equal(calls[1 + 2 * axis], expected)
        assert torch.equal(calls[2 + 2 * axis], -expected)
    rows = receipt["proposal_rows"]
    eligible = [r for r in rows[1:] if r["eligible"]]
    assert receipt["strict_selected_index"] == min(
        eligible, key=lambda r: (sum(r["per_tail_delta"]), r["proposal_index"])
    )["proposal_index"]


def test_development_selection_cannot_imply_strict_mechanism():
    def conflicting(x):
        return x[0], torch.stack([-x[1], -2*x[1], x[1], x[1], x[1]])
    _, receipt = search.generate(torch.zeros(10, dtype=torch.float64), conflicting)
    assert receipt["development_selected_index"] is not None
    assert receipt["strict_selected_index"] is None


def test_no_candidate_and_zero_tangents_are_not_fallbacks():
    def no_tail(x):
        return x[0], torch.stack([x[0]*0]*5)
    _, receipt = search.generate(torch.zeros(10, dtype=torch.float64), no_tail)
    assert receipt["development_selected_index"] is None
    assert receipt["strict_selected_index"] is None
    assert not any(receipt["direction_valid"])


def test_zero_immediate_gradient_and_nonfinite_fail_closed():
    with pytest.raises(ValueError, match="zero immediate"):
        search.generate(torch.zeros(10), lambda x: (x[0]*0, torch.zeros(5)))
    with pytest.raises(ValueError, match="invalid"):
        search.generate(torch.zeros(10), lambda x: (x[0], torch.full((5,), float("nan"))))


def test_cpu_neural_adapter_rejected():
    model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="host CUDA"):
        search.compile_reacher(model, torch.zeros(1), torch.zeros(1), torch.zeros(1, 3, 10),
                               None, lower=torch.zeros(10), upper=torch.ones(10))
