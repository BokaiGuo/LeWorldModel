"""Small SPD box quadratic, active-set solve and primal/dual error certificate.

All linear algebra stays on the supplied device (CUDA in live experiments).
Unlike a fixed gradient-step count, the receipt exposes an objective gap
bound. No simulator observations or physical labels are inputs.
"""
from __future__ import annotations

import math

import torch


def solve_box_quadratic(gram, projected, *, max_iterations=500):
    """Minimize u' gram u - 2 projected' u subject to -1 <= u <= 1."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or projected.shape != gram.shape[:1]:
        raise ValueError("square SPD Gram and matching projection required")
    if gram.device != projected.device or gram.dtype != projected.dtype or gram.dtype != torch.float64:
        raise ValueError("matching float64 Gram/projection on the same device required")
    if not torch.isfinite(gram).all() or not torch.isfinite(projected).all() or max_iterations < 1:
        raise ValueError("finite inputs and positive iteration budget required")
    if not torch.allclose(gram, gram.T, rtol=1e-12, atol=1e-12):
        raise ValueError("symmetric Gram required")
    # Fail closed on non-SPD objectives, rather than silently pseudoinverting.
    torch.linalg.cholesky(gram)
    action = torch.zeros_like(projected)
    active = torch.zeros_like(projected, dtype=torch.bool)
    terminated = False
    for iteration in range(1, max_iterations + 1):
        free = ~active
        trial = action.clone()
        if bool(free.any()):
            rhs = projected[free] - gram[free][:, active] @ action[active]
            trial[free] = torch.linalg.solve(gram[free][:, free], rhs)
        direction = trial - action
        if bool((trial[free].abs() > 1 + 1e-12).any()):
            limits = torch.full_like(action, torch.inf)
            positive, negative = free & (direction > 0), free & (direction < 0)
            limits[positive] = (1 - action[positive]) / direction[positive]
            limits[negative] = (-1 - action[negative]) / direction[negative]
            alpha = limits.min().clamp(0, 1)
            action = (action + alpha * direction).clamp(-1, 1)
            hits = free & (action.abs() >= 1 - 1e-11)
            if not bool(hits.any()):
                raise RuntimeError("box active-set step failed to hit a constraint")
            action[hits] = action[hits].sign()
            active |= hits
            continue
        action = trial.clamp(-1, 1)
        gradient = gram @ action - projected
        violation = torch.where(active, gradient * action, torch.full_like(action, -torch.inf))
        largest, index = violation.max(dim=0)
        tolerance = 1e-11 * max(1., float(projected.abs().max()))
        if float(largest) > tolerance:
            active[index] = False
            continue
        terminated = True
        break
    gradient = gram @ action - projected
    # Convex tangent-plane lower bound over the entire box:
    # f(v) >= f(u) + 2 grad(u)'(v-u), min_v grad'v = -||grad||_1.
    dual_gap = (2 * (gradient.abs().sum() + gradient @ action)).clamp_min(0)
    objective = action @ gram @ action - 2 * projected @ action
    return action, {"iterations": iteration, "active_set_terminated": terminated,
        "active_constraints": int(active.sum()), "quadratic_objective_without_constant": float(objective),
        "objective_duality_gap_upper_bound": float(dual_gap),
        "box_feasibility_violation": float((action.abs() - 1).clamp_min(0).max()),
        "device": str(action.device)}


def certified_response_gain(response, residual, *, damping=.01, steps=1000):
    """Drop-in gain metric, with a globally valid box-QP objective gap bound.

    steps is accepted for caller compatibility but is NOT a convergence claim;
    the active-set cap is always 500 and the actual certificate is checked.
    """
    if response.ndim != 2 or residual.shape != response.shape[:1] or min(response.shape) < 1:
        raise ValueError("aligned response and task residual required")
    if response.device != residual.device or response.dtype != residual.dtype or response.dtype != torch.float64:
        raise ValueError("matching float64 response/residual device required")
    if not math.isfinite(damping) or damping <= 0 or not torch.isfinite(response).all() or not torch.isfinite(residual).all():
        raise ValueError("finite response/residual and positive damping required")
    gram = response.T @ response + damping * torch.eye(response.shape[1], device=response.device, dtype=response.dtype)
    projected = response.T @ residual
    action, certificate = solve_box_quadratic(gram, projected)
    baseline = residual.square().sum()
    objective = (residual - response @ action).square().sum() + damping * action.square().sum()
    gain = float(baseline - objective)
    gap = certificate["objective_duality_gap_upper_bound"]
    tolerance = 1e-8 * max(1., float(baseline))
    certified = certificate["active_set_terminated"] and gap <= tolerance and certificate["box_feasibility_violation"] == 0
    if not certified:
        raise RuntimeError("bounded response QP lacks requested objective certificate: " + str(certificate))
    return {"bounded_quadratic_gain": gain, "bounded_quadratic_gain_upper_bound": gain + gap,
        "baseline_squared_residual": float(baseline), "bounded_quadratic_objective": float(objective),
        "response_singular_values": torch.linalg.svdvals(response).tolist(), "qp_certificate": certificate,
        "objective_certificate_passed": certified, "physical_labels_used": False}


def qualify_certified_decision(receipt, geometry):
    """Only admit gains whose lower bound beats both control upper bounds."""
    from experiments.spectral_reachability import reacher_d2p_geometry_dispatch as compiler
    greedy, winners = compiler.endpoint_winners(receipt)
    # Preserve the old complete-coverage and finite-value validation.
    decision = compiler.qualify_decision(receipt, geometry)
    if any(not cell.get("objective_certificate_passed") or not math.isfinite(cell["bounded_quadratic_gain_upper_bound"])
           or cell["bounded_quadratic_gain_upper_bound"] < cell["bounded_quadratic_gain"] for cell in geometry.values()):
        raise ValueError("certified finite gain intervals required for all candidates")
    threshold = max(geometry[index]["bounded_quadratic_gain_upper_bound"] for index in (0, greedy))
    qualified = [index for index in winners if geometry[index]["bounded_quadratic_gain"] > threshold]
    chosen = min(qualified, key=lambda index: (receipt["proposal_rows"][index]["refined_goal_cost"], index)) if qualified else greedy
    return {**decision, "policy": "certified_common_task_response_reconfiguration_v1",
        "route": "certified_task_response_reconfiguration" if qualified else "same_bank_goal_progress",
        "selected_program_index": chosen, "reconfiguration_witness_found": bool(qualified),
        "geometry_qualified_indices": qualified, "objective_comparison_certified": True,
        "common_task_gain_lower_above_both_control_uppers": geometry[chosen]["bounded_quadratic_gain"] - threshold,
        "refined_greedy_gain": receipt["proposal_rows"][greedy]["refined_goal_cost"] - receipt["proposal_rows"][chosen]["refined_goal_cost"]}
