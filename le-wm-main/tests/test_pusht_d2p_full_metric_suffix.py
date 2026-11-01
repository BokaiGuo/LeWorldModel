import numpy as np
import pytest
import torch

from experiments.spectral_reachability.pusht_d2p_full_metric_suffix import refine


def test_refinement_keeps_zero_and_initial_incumbents_and_finite_box():
    terminal = lambda x: torch.stack((torch.tanh(x[0, 0]), 2*torch.tanh(x[0, 1])))
    params, receipt = refine(terminal, torch.tensor([.6, -.7]), torch.tensor([[9., -9.]]))
    assert receipt["final_objective"] <= receipt["zero_objective"]
    assert receipt["final_objective"] <= receipt["initial_objective"]
    assert all(r["after"] <= r["before"] for r in receipt["iterations"])
    assert (torch.tanh(params).abs() <= 1).all()
    np.testing.assert_allclose(terminal(params).detach(), receipt["predicted_terminal_vector"])


def test_zero_iterations_never_discards_better_initial_solution():
    terminal = lambda x: x.reshape(-1)
    start = torch.tensor([[.2, -.1]])
    params, receipt = refine(terminal, start.reshape(-1), start, iterations=0)
    assert torch.equal(params, start)
    assert receipt["final_objective"] < receipt["zero_objective"]


@pytest.mark.parametrize("bad", [torch.tensor([[float("nan"), 0.]]), torch.zeros(3)])
def test_invalid_parameter_contract_rejected(bad):
    with pytest.raises(ValueError):
        refine(lambda x: x.reshape(-1), torch.zeros(2), bad)


def test_nonfinite_terminal_rejected():
    with pytest.raises(ValueError):
        refine(lambda x: x.reshape(-1)*float("nan"), torch.zeros(2), torch.zeros(1, 2))
