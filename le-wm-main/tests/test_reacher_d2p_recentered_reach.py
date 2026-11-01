"""Analytic numeric fixtures, never neural CPU or efficacy evidence."""
import copy
import numpy as np
import pytest
import torch
from scipy.optimize import lsq_linear

from experiments.spectral_reachability import reacher_d2p_recentered_reach as r


def affine():
    generator = torch.Generator().manual_seed(104)
    matrix = torch.randn(192, 3, generator=generator, dtype=torch.float64)
    offset = torch.randn(192, generator=generator, dtype=torch.float64)
    target = offset+matrix@torch.tensor([1.4, -.6, -.8], dtype=torch.float64)
    return matrix, offset, target


@pytest.mark.parametrize("anchor", [[0., 0., 0.], [.9, -.4, .7], [-1., 1., -.8]])
def test_nonzero_anchor_retains_absolute_box_and_regularizer(anchor):
    j, offset, goal = affine()
    a = torch.tensor(anchor, dtype=torch.float64)
    value = offset+j@a
    command, residual, cost, cert = r.affine_problem(j, value, goal, a)
    np.testing.assert_allclose(residual.numpy(), (goal-offset).numpy(), atol=1e-13)
    fit = lsq_linear(np.vstack((j.numpy(), np.sqrt(.01)*np.eye(3))),
        np.r_[goal.numpy()-offset.numpy(), np.zeros(3)], bounds=(-1, 1), tol=1e-13)
    np.testing.assert_allclose(command.numpy(), fit.x, atol=1e-8)
    assert cost == pytest.approx(r.objective(offset+j@command, goal, command), abs=1e-12)
    assert cert["active_set_terminated"] is True


def test_affine_refinement_and_exact_incumbent_monotonicity():
    j, offset, goal = affine()
    command, value, receipt = r.refine(lambda x: offset+j@x, torch.zeros(3, dtype=torch.float64), goal)
    assert torch.equal(value, offset+j@command)
    previous = receipt["initial_objective"]
    for step in receipt["trace"]:
        assert step["objective_before"] == previous
        assert step["objective_after"] <= previous
        previous = step["objective_after"]
    assert previous == receipt["final_objective"]
    assert receipt["global_nonlinear_lower_bound_claimed"] is False


def test_local_qp_is_not_a_nonlinear_impossibility_certificate():
    def curve(x):
        return torch.cat((x.square(), torch.zeros(191, dtype=x.dtype)))
    zero, target = torch.zeros(1, dtype=torch.float64), torch.zeros(192, dtype=torch.float64)
    target[0] = 1
    j = torch.autograd.functional.jacobian(curve, zero)
    _, _, local_cost, _ = r.affine_problem(j, curve(zero), target, zero)
    incumbent = torch.ones(1, dtype=torch.float64)
    assert local_cost == pytest.approx(1.)
    assert r.objective(curve(incumbent), target, incumbent) == pytest.approx(.01)
    _, _, receipt = r.refine(curve, incumbent, target)
    assert receipt["final_objective"] <= .01


@pytest.mark.parametrize("bad", ["shape", "nan", "box", "target", "response"])
def test_invalid_local_inputs_fail_closed(bad):
    j, offset, goal = affine()
    anchor = torch.zeros(3, dtype=torch.float64)
    if bad == "shape": anchor = anchor[:2]
    elif bad == "nan": offset[0] = float("nan")
    elif bad == "box": anchor[0] = 1.1
    elif bad == "target": goal = goal[:191]
    else: j[0, 0] = float("inf")
    with pytest.raises(ValueError): r.affine_problem(j, offset, goal, anchor)


def bank():
    return [{"index": i, "prefix_cost": 1., "feasible": 1.} for i in range(17)]


def test_model_only_selector_ties_and_finite_contract():
    b = bank()
    assert r.select(b, "feasible")["native_index"] == 0
    b[4]["feasible"] = 0.
    assert r.select(b, "feasible")["native_index"] == 4
    assert r.select(b, "feasible")["P1_admission"] is False


@pytest.mark.parametrize("bad", ["missing", "reordered", "nan", "negative"])
def test_incomplete_or_invalid_bank_rejected(bad):
    b = copy.deepcopy(bank())
    if bad == "missing": b.pop()
    elif bad == "reordered": b.reverse()
    elif bad == "nan": b[0]["feasible"] = float("nan")
    else: b[0]["prefix_cost"] = -1.
    with pytest.raises(ValueError): r.select(b, "feasible")
