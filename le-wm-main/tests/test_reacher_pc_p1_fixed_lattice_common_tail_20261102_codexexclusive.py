import pytest
import torch

from experiments.spectral_reachability import reacher_pc_p1_fixed_lattice_common_tail_20261102_codexexclusive as search


def feasible(x):
    return x[0], torch.stack([-x[1], -2*x[1], -3*x[1], -4*x[1], -5*x[1]])


def test_fixed_bank_is_reproducible_and_budgeted():
    calls = []

    def evaluate(x):
        calls.append(x.detach().clone())
        return feasible(x)

    origin = torch.zeros(10, dtype=torch.float64)
    bank, receipt = search.generate(origin, evaluate)
    bank2, receipt2 = search.generate(origin, feasible)
    assert bank.shape == (41, 10)
    assert len(calls) == receipt["logical_costs"]["evaluation_calls"] == 41
    assert receipt["logical_costs"]["prefix_gradient_vjp_calls"] == 0
    assert receipt["logical_costs"]["suffix_predict_samples"] == 410
    assert torch.equal(bank, bank2)
    assert receipt == receipt2
    assert torch.allclose(torch.linalg.vector_norm(bank[1:], dim=1),
                          torch.full((40,), search.RADIUS, dtype=bank.dtype))


def test_strict_selector_can_choose_model_feasible_row_without_fallback():
    origin = torch.zeros(10, dtype=torch.float64)
    _, receipt = search.generate(origin, feasible)
    assert receipt["selected_proposal_index"] is not None
    selected = receipt["proposal_rows"][receipt["selected_proposal_index"]]
    assert selected["eligible"]


def test_no_physical_or_privileged_argument_and_no_fallback():
    def no_effect(x):
        return x[0], torch.zeros(5, dtype=x.dtype)

    _, receipt = search.generate(torch.zeros(10), no_effect)
    assert receipt["selected_proposal_index"] is None
    assert not any(row["eligible"] for row in receipt["proposal_rows"])
    with pytest.raises(TypeError):
        search.generate(torch.zeros(10), lambda x, hidden: feasible(x))


def test_single_absolute_or_adjusted_tail_violation_rejects():
    z = torch.tensor(0.0, dtype=torch.float64)
    base = torch.zeros(5, dtype=torch.float64)
    point = torch.zeros(10, dtype=torch.float64)
    point[0] = search.RADIUS
    absolute_worse = torch.tensor([-0.1, -0.1, -0.1, -0.1, 0.01], dtype=torch.float64)
    adjusted_worse = torch.tensor([-0.2, -0.1, -0.2, -0.2, -0.2], dtype=torch.float64)
    assert not search._row(1, point, point, point[0], absolute_worse, z, base, "x")["eligible"]
    assert not search._row(1, point, point, point[0], adjusted_worse, z, base, "x")["eligible"]


def test_nonfinite_costs_are_rejected():
    def nan_eval(x):
        return x[0], torch.full((5,), float("nan"), dtype=x.dtype)

    with pytest.raises(ValueError, match="invalid"):
        search.generate(torch.zeros(10), nan_eval)
