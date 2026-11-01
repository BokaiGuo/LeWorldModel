"""Direct bounded-command GN, separate from the historical tanh optimizer.

Commands are exactly tanh(legacy_params), but their derivative does not carry
the vanishing tanh factor. Each GN endpoint is solved in the actual command
box with an SPD-QP certificate; nonlinear model descent still accepts steps.
This is a numerical action-chart diagnostic, not a new program mechanism.
"""
from __future__ import annotations

import math

import torch

from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability.bounded_response_qp import solve_box_quadratic


def terminal_commands(model, emb_context, action_context, commands, *, action_lower, action_upper):
    """Same frozen recurrence/affine normalization, without tanh or atanh."""
    if (commands.ndim != 3 or commands.shape[0] != emb_context.shape[0] or commands.shape[1] < 1
            or commands.shape[-1] != action_context.shape[-1] or emb_context.shape[:2] != action_context.shape[:2]
            or emb_context.shape[1] < 2 or action_lower.ndim != 1 or action_upper.shape != action_lower.shape
            or action_lower.numel() != commands.shape[-1] or not torch.all(action_lower < action_upper)):
        raise ValueError("aligned action/latent contexts, commands and physical bounds required")
    midpoint, radius = (action_lower + action_upper) / 2., (action_upper - action_lower) / 2.
    future = midpoint + radius * commands
    history_size = int(emb_context.shape[1])
    actions = torch.cat([action_context, future[:, :1]], dim=1)
    embeddings = emb_context
    for step in range(commands.shape[1]):
        terminal = model.predict(embeddings[:, -history_size:], model.action_encoder(actions[:, -history_size:]))[:, -1]
        if step + 1 < commands.shape[1]:
            embeddings = torch.cat([embeddings, terminal[:, None]], dim=1)
            actions = torch.cat([actions, future[:, step + 1:step + 2]], dim=1)
    return terminal


def compile_box_snapshots(model, goal_latent, emb_context, action_context, *, initial_commands,
                           action_lower, action_upper, damping=.01, checkpoints=(8, 32)):
    if (not checkpoints or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
            or any(not isinstance(step, int) or isinstance(step, bool) or step < 1 for step in checkpoints)
            or not math.isfinite(damping) or damping <= 0):
        raise ValueError("positive damping and ordered unique positive iteration checkpoints required")
    if (not isinstance(initial_commands, torch.Tensor) or initial_commands.ndim != 3
            or initial_commands.shape[0] != 1 or initial_commands.shape[1] < 1
            or initial_commands.shape[-1] != action_context.shape[-1]
            or initial_commands.device != emb_context.device or initial_commands.dtype != emb_context.dtype
            or not torch.isfinite(initial_commands).all() or bool((initial_commands.abs() > 1).any())):
        raise ValueError("finite bounded initial commands with exact dtype/device required")
    horizon, action_dim = initial_commands.shape[1:]
    flat = initial_commands.detach().reshape(-1).clone()
    goal = p0._planning_coordinate(model, goal_latent, role="goal")
    evaluations, vjp_rows, trace, snapshots, stopped = 0, 0, [], {}, False

    def terminal(value):
        nonlocal evaluations
        evaluations += 1
        latent = terminal_commands(model, emb_context, action_context, value.reshape(1, horizon, action_dim),
            action_lower=action_lower, action_upper=action_upper)[0]
        return p0._planning_coordinate(model, latent, role="predicted")

    for iteration in range(1, max(checkpoints) + 1):
        if not stopped:
            origin = flat.detach().requires_grad_(True)
            predicted = terminal(origin)
            jacobian = torch.autograd.functional.jacobian(terminal, origin, vectorize=True)
            if jacobian.ndim != 2 or not torch.isfinite(jacobian).all() or float(jacobian.norm()) <= 1e-10:
                raise RuntimeError("nonfinite or zero direct-command response")
            response = jacobian.detach().to(torch.float64)
            residual = (goal - predicted).detach().to(torch.float64)
            current = flat.to(torch.float64)
            gram = response.T @ response + damping * torch.eye(len(flat), device=flat.device, dtype=torch.float64)
            # v is an absolute feasible command endpoint; delta = v-current.
            # This accounts for asymmetric available motion at a box boundary.
            projected = response.T @ residual + gram @ current
            endpoint, certificate = solve_box_quadratic(gram, projected)
            tolerance = 1e-8 * max(1., float(residual.square().sum()), float(current @ gram @ current))
            if (not certificate["active_set_terminated"] or certificate["box_feasibility_violation"] != 0
                    or certificate["objective_duality_gap_upper_bound"] > tolerance):
                raise RuntimeError("direct-command local QP lacks objective certificate")
            direction = endpoint.to(flat.dtype) - flat
            delta, accepted, receipt = rescue.accept_with_fine_rescue(lambda offset: terminal(flat + offset),
                direction, predicted, goal, step_rule="backtracking")
            flat = (flat + delta).detach()
            before = float((predicted - goal).square().mean().detach())
            after = float((accepted - goal).square().mean().detach())
            if not torch.isfinite(flat).all() or bool((flat.abs() > 1 + 1e-6).any()) or after > before:
                raise RuntimeError("nonlinear descent or command box contract failed")
            vjp_rows += int(jacobian.shape[0])
            trace.append({"iteration": iteration, "cost_before": before, "cost_after": after,
                "boundary_fraction": float((flat.abs() >= .99).float().mean()), "qp_certificate": certificate,
                "direct_response_norm": float(jacobian.norm()), **receipt})
            stopped = receipt["accepted_step_scale"] == 0.
        if iteration in checkpoints:
            snapshots[iteration] = (flat.reshape(1, horizon, action_dim).clone(), {
                "program_type": "direct_box_command_certified_gn_suffix_v1", "damping": damping,
                "iterations_requested": iteration, "iterations_executed": len(trace), "stopped_no_model_descent": stopped,
                "trace": list(trace), "terminal_forward_evaluations": evaluations,
                "model_predict_sample_evaluations": evaluations * horizon, "backward_vjp_rows": vjp_rows,
                "cost_nonincreasing": all(item["cost_after"] <= item["cost_before"] for item in trace),
                "physical_labels_used": False, "scope": "Bounded action-chart diagnostic, not P1 admission or physical guarantee."})
    return snapshots
