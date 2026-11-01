"""Nonlinear-validated bounded suffix refinement, not a global reachability proof.

Keep the absolute raw-action box and lambda*||u||^2 objective when moving the
Taylor anchor. A box-QP certificate applies only to that affine model; actual
frozen-model singleton evaluation decides whether an update is accepted.
"""
from __future__ import annotations

import math
import torch

from experiments.spectral_reachability.bounded_response_qp import solve_box_quadratic

DAMPING = .01
SCALES = (1., .5, .25, .125, .0625, .03125)
CONTRACT = {
    "schema": "reacher_d2p_recentered_reach.v1", "iterations": 2, "damping": DAMPING,
    "action_box": [-1., 1.], "regularizer": "absolute raw suffix action, not anchor displacement",
    "initial": "exact original cem_then_gn4 suffix", "scales": list(SCALES),
    "selection": "prefix MSE + nonlinear-validated feasible objective / 192; beta=1; ties original index",
    "certificate_scope": "local affine box QP only; no global nonlinear lower bound",
    "P1_admission": False, "physical_labels_used": False,
}


def objective(prediction, target, command):
    if (prediction.shape != (192,) or target.shape != (192,) or command.ndim != 1 or len(command) < 1
            or any(not torch.isfinite(v).all() for v in (prediction, target, command))
            or prediction.device != target.device or prediction.device != command.device
            or bool((command.abs() > 1).any())):
        raise ValueError("finite full192 prediction/goal and same-device raw box command required")
    value = (prediction.double()-target.double()).square().sum()+DAMPING*command.double().square().sum()
    if not torch.isfinite(value):
        raise ValueError("nonfinite regularized objective")
    return float(value.detach())


def affine_problem(response, value, target, anchor):
    """F(u) ~= F(a)+J(u-a), hence r_affine = g-F(a)+Ja.

    The unknown is the *absolute* u in [-1,1], not a delta in that box. No
    lambda*a term belongs on the right side for absolute-action regularization.
    """
    objective(value, target, anchor)
    if (response.shape != (192, len(anchor)) or response.device != anchor.device
            or not torch.isfinite(response).all()):
        raise ValueError("finite aligned full response required")
    j, a = response.double(), anchor.detach().double()
    residual = target.double()-value.double()+j@a
    gram = j.T@j+DAMPING*torch.eye(len(a), device=a.device, dtype=torch.float64)
    endpoint, certificate = solve_box_quadratic(gram, j.T@residual)
    gap = certificate["objective_duality_gap_upper_bound"]
    if (certificate["active_set_terminated"] is not True or certificate["box_feasibility_violation"] != 0
            or not math.isfinite(gap) or gap < 0 or gap > 1e-8*max(1., float(residual.square().sum()))):
        raise RuntimeError("uncertified local affine box QP")
    local_value = float((residual-j@endpoint).square().sum()+DAMPING*endpoint.square().sum())
    return endpoint, residual, local_value, certificate


def refine(terminal, initial, target):
    current = initial.detach().clone()
    with torch.no_grad(): value = terminal(current)
    initial_vector = value.detach().clone()
    first = objective(value, target, current)
    trace = []
    for iteration in range(CONTRACT["iterations"]):
        anchor, anchor_value = current.detach().clone(), value.detach().clone()
        response = torch.autograd.functional.jacobian(terminal, anchor.requires_grad_(True), vectorize=True).detach()
        endpoint, residual, local_value, cert = affine_problem(response, anchor_value, target, anchor)
        before, best = objective(anchor_value, target, anchor), objective(anchor_value, target, anchor)
        chosen, chosen_value, accepted = anchor.detach().clone(), anchor_value, 0.
        trials = []
        with torch.no_grad():
            for scale in SCALES:
                command = ((1-scale)*anchor.detach()+scale*endpoint.to(anchor.dtype)).clamp(-1, 1)
                prediction = terminal(command)
                cost = objective(prediction, target, command)
                trials.append({"scale": scale, "command": command.tolist(), "vector": prediction.tolist(), "objective": cost})
                if cost < best:
                    chosen, chosen_value, best, accepted = command.clone(), prediction.clone(), cost, scale
        trace.append({"iteration": iteration+1, "anchor_command": anchor.detach().tolist(),
            "anchor_vector": anchor_value.tolist(), "response": response.tolist(), "affine_residual": residual.tolist(),
            "qp_action": endpoint.tolist(), "local_affine_objective": local_value, "qp_certificate": cert,
            "objective_before": before, "objective_after": best, "accepted_scale": accepted, "trials": trials})
        current, value = chosen.detach(), chosen_value.detach()
    with torch.no_grad(): final = terminal(current)
    if not torch.equal(final, value) or objective(final, target, current) > first:
        raise RuntimeError("nonlinear singleton incumbent lost or nondeterministic forecast")
    return current, final, {"initial_vector": initial_vector.tolist(), "initial_objective": first,
        "final_objective": objective(final, target, current), "trace": trace,
        "global_nonlinear_lower_bound_claimed": False, "physical_labels_used": False}


def select(bank, cost_key):
    if len(bank) != 17 or [b["index"] for b in bank] != list(range(17)):
        raise ValueError("all17 original ordered prefixes required")
    scores = []
    for b in bank:
        stage, reach = float(b["prefix_cost"]), float(b[cost_key])
        if any(not math.isfinite(v) or v < 0 for v in (stage, reach)):
            raise ValueError("finite nonnegative stage and validated objective required")
        scores.append(stage+reach/192.)
    return {"native_index": min(range(17), key=lambda i: (scores[i], i)), "scores": scores,
        "reach_objective": cost_key, "P1_admission": False, "physical_labels_used": False}
