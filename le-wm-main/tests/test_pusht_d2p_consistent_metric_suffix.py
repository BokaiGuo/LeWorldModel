import pytest
import torch

from experiments.spectral_reachability.pusht_d2p_consistent_metric_suffix import refine_in_metric


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host CUDA numerical fixture required; no CPU fallback")
def test_same_metric_controls_jacobian_objective_and_returned_vector():
    device = torch.device("cuda")
    goal = torch.tensor([100., 1.], device=device)
    initial = torch.zeros((1, 2), device=device)
    projection = torch.diag(torch.tensor([0., 1.], device=device))
    action, receipt = refine_in_metric(lambda value: value.reshape(2), goal, initial, projection)
    assert abs(float(action[0, 0])) < 1e-6
    assert float(action[0, 1]) == pytest.approx(1/1.01, abs=1e-5)
    assert receipt["final_objective"] < receipt["zero_objective"]
    assert receipt["predicted_terminal_vector"][0] == 0.
    assert all(step["after"] <= step["before"] for step in receipt["iterations"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host CUDA numerical fixture required; no CPU fallback")
def test_identity_metric_matches_original_GN():
    from experiments.spectral_reachability.pusht_d2p_full_metric_suffix import refine
    goal = torch.tensor([.5, -.2], device="cuda")
    initial = torch.zeros((1, 2), device="cuda")
    terminal = lambda value: value.reshape(2).tanh()
    actual, receipt = refine_in_metric(terminal, goal, initial, torch.eye(2, device="cuda"))
    expected, original = refine(terminal, goal, initial, damping=.01, iterations=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert receipt["final_objective"] == original["final_objective"]


def test_CPU_inputs_are_rejected_before_any_callable_evaluation():
    def forbidden(_):
        raise AssertionError("must reject before model evaluation")
    with pytest.raises(ValueError, match="CUDA"):
        refine_in_metric(forbidden, torch.zeros(2), torch.zeros((1, 2)), torch.eye(2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host CUDA required; no CPU fallback")
def test_saturated_action_keeps_exact_parameter_energy_receipt():
    from experiments.spectral_reachability.pusht_d2p_full_metric_suffix import refine
    start = torch.tensor([[10., -10.]], device="cuda")
    goal = torch.tensor([10., -10.], device="cuda")
    params, receipt = refine(lambda value: value.reshape(2), goal, start, iterations=0)
    assert torch.equal(params, start)
    assert torch.tanh(params).abs().eq(1).all()
    assert receipt["final_parameters"] == [[10., -10.]]
    assert receipt["final_parameter_energy"] == 200.
    assert receipt["final_terminal_sse"] == 0.
    assert receipt["final_objective"] == pytest.approx(2.)
    assert receipt["regularization_center"] == "absolute_zero_not_current_iterate"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host CUDA required; no CPU fallback")
def test_nonzero_nominal_absolute_ridge_matches_analytic_solution():
    from experiments.spectral_reachability.pusht_d2p_full_metric_suffix import refine
    initial = torch.tensor([[.5, 0.]], device="cuda")
    goal = torch.tensor([1., 0.], device="cuda")
    params, receipt = refine(lambda value: value.reshape(2), goal, initial, damping=1., iterations=1)
    torch.testing.assert_close(params, initial, rtol=0, atol=0)
    assert receipt["final_terminal_sse"] == pytest.approx(.25)
    assert receipt["final_parameter_energy"] == pytest.approx(.25)
    assert receipt["final_objective"] == pytest.approx(.5)
