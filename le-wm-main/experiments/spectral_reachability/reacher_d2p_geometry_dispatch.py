"""Model-only common-residual qualification for reconfiguration witnesses.

A lower optimized terminal cost alone does not establish changed action
response. Keep the full-bank fine8 endpoint test, but additionally require
bounded local action response to improve on BOTH controls for the SAME task
residual, fixed by the imagined greedy-prefix/raw-zero-suffix endpoint.
This is a separately named diagnostic policy, not a retroactive P1 pass.
"""
from __future__ import annotations

import math

import torch

from experiments.spectral_reachability import reacher_d2p_refined_dispatch as base
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0

POLICY = "full_bank_fine8_common_task_response_qualified_reconfiguration_v1"


def bounded_response_gain(response, residual, *, damping=.01, steps=1000):
    """Raw-torque local linear surrogate with a fixed box and fixed iterations."""
    if response.ndim != 2 or residual.shape != response.shape[:1] or min(response.shape) < 1:
        raise ValueError("output-by-action response and aligned task residual required")
    if response.device != residual.device or response.dtype != residual.dtype:
        raise ValueError("matching response/residual device and dtype required")
    if not torch.isfinite(response).all() or not torch.isfinite(residual).all() or not math.isfinite(damping) or damping <= 0 or steps < 1:
        raise ValueError("finite responses and positive fixed optimizer settings required")
    gram = response.T @ response
    projected = response.T @ residual
    lipschitz = torch.linalg.eigvalsh(gram)[-1].clamp_min(0) + damping
    regularized = gram + torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * damping
    action = torch.zeros_like(projected)
    for _ in range(steps):
        action = (action - (regularized @ action - projected) / lipschitz).clamp(-1, 1)
    baseline = residual.square().sum()
    objective = (residual - response @ action).square().sum() + damping * action.square().sum()
    gradient = regularized @ action - projected
    return {"bounded_quadratic_gain": float(baseline - objective), "baseline_squared_residual": float(baseline),
        "bounded_quadratic_objective": float(objective), "response_singular_values": torch.linalg.svdvals(response).tolist(),
        "projected_gradient_residual": float((action - (action - gradient / lipschitz).clamp(-1, 1)).norm()),
        "physical_labels_used": False}


def endpoint_winners(receipt):
    rows = receipt["proposal_rows"]
    # Revalidate full required coverage; a truncated bank cannot open a witness.
    decision = base.select_refined_bank(rows, non_greedy_margin=receipt["non_greedy_margin"])
    greedy = decision["same_bank_greedy_index"]
    eligible = [row["proposal_index"] for row in rows[1:]
        if row["immediate_goal_cost"] >= rows[0]["immediate_goal_cost"] + receipt["non_greedy_margin"]]
    return greedy, [index for index in eligible if rows[index]["refined_goal_cost"] < min(rows[0]["refined_goal_cost"], rows[greedy]["refined_goal_cost"])]


def qualify_decision(receipt, geometry):
    greedy, candidates = endpoint_winners(receipt)
    required = set(candidates) | {0, greedy}
    if set(geometry) != required or any(not math.isfinite(geometry[index]["bounded_quadratic_gain"]) for index in required):
        raise ValueError("every endpoint winner and both controls require finite common-residual geometry")
    threshold = max(geometry[0]["bounded_quadratic_gain"], geometry[greedy]["bounded_quadratic_gain"])
    qualified = [index for index in candidates if geometry[index]["bounded_quadratic_gain"] > threshold]
    chosen = min(qualified, key=lambda index: (receipt["proposal_rows"][index]["refined_goal_cost"], index)) if qualified else greedy
    return {**receipt["dispatch"], "policy": POLICY, "route": "common_task_response_reconfiguration" if qualified else "same_bank_goal_progress",
        "selected_program_index": chosen, "reconfiguration_witness_found": bool(qualified), "physical_labels_used": False,
        "endpoint_winner_indices": candidates, "geometry_qualified_indices": qualified, "geometry_required_indices": sorted(required),
        "common_task_gain_above_both_controls": geometry[chosen]["bounded_quadratic_gain"] - threshold,
        "refined_greedy_gain": receipt["proposal_rows"][greedy]["refined_goal_cost"] - receipt["proposal_rows"][chosen]["refined_goal_cost"]}


def response_bank_metrics(model, pixels, goal, history, *, proposal_bank, action_lower,
                          action_upper, required, greedy, damping=.01, gain_solver=None):
    """Shared model response path for numerical audits and future compilers."""
    if greedy not in required or 0 not in required or len(required) != len(set(required)):
        raise ValueError("unique response candidates including both controls required")
    gain_solver = bounded_response_gain if gain_solver is None else gain_solver
    responses, endpoints, geometry = {}, {}, {}
    goal_coordinate = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach().to(torch.float64)
    extra_samples, extra_vjp = 0, 0
    for index in required:
        emb, actions = p1._state_after_prefix(model, pixels, history, proposal_bank[index:index + 1],
            action_lower=action_lower, action_upper=action_upper)
        calls = 0

        def terminal(flat):
            nonlocal calls
            calls += 1
            value = p1._terminal_from_state(model, emb, actions, flat.reshape(1, 2, history.shape[-1]),
                action_lower=action_lower, action_upper=action_upper)[0]
            return p0._planning_coordinate(model, value, role="predicted")

        origin = torch.zeros(2 * history.shape[-1], device=history.device, dtype=history.dtype, requires_grad=True)
        endpoint = terminal(origin)
        response = torch.autograd.functional.jacobian(terminal, origin, vectorize=True)
        endpoints[index], responses[index] = endpoint.detach().to(torch.float64), response.detach().to(torch.float64)
        extra_samples += 1 + calls * 2
        extra_vjp += int(response.shape[0])
    residual = goal_coordinate - endpoints[greedy]
    for index in required:
        geometry[index] = gain_solver(responses[index], residual, damping=damping, steps=1000)
        geometry[index]["zero_suffix_endpoint"] = endpoints[index].tolist()
    return geometry, residual, extra_samples, extra_vjp


def compile_refined_dispatch(model, pixels, goal, history, *, proposal_bank, action_lower,
                             action_upper, damping=.01, non_greedy_margin=1e-6, progress=None):
    _, receipt = base.compile_refined_dispatch(model, pixels, goal, history, proposal_bank=proposal_bank,
        action_lower=action_lower, action_upper=action_upper, damping=damping, non_greedy_margin=non_greedy_margin, progress=progress)
    greedy, winners = endpoint_winners(receipt)
    required = sorted(set(winners) | {0, greedy})
    geometry, residual, extra_samples, extra_vjp = response_bank_metrics(model, pixels, goal, history,
        proposal_bank=proposal_bank, action_lower=action_lower, action_upper=action_upper,
        required=required, greedy=greedy, damping=damping)
    decision = qualify_decision(receipt, geometry)
    receipt.update({"program_type": POLICY, "endpoint_only_dispatch": receipt["dispatch"], "dispatch": decision,
        "common_task_response": {"anchor": "imagined_greedy_prefix_raw_zero_suffix", "residual": residual.tolist(),
            "damping": damping, "projected_steps": 1000, "candidate_metrics": geometry, "physical_labels_used": False},
        "model_predict_sample_evaluations": receipt["model_predict_sample_evaluations"] + extra_samples,
        "backward_vjp_rows": receipt["backward_vjp_rows"] + extra_vjp,
        "scope": "Model-only common-task response witness qualification; finite local surrogate, NOT physical controllability or P1 admission."})
    chosen = decision["selected_program_index"]
    return proposal_bank[chosen:chosen + 1], receipt
