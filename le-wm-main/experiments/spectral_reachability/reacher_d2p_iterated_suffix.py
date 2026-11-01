"""Separately named iterative SVD-DLS suffix; historical P1 is unchanged.

The first iteration is exactly the original singleton bounded-SVD solver.
Further iterations relinearize at accepted parameters, not at zero. Neither
physical simulator state nor outcome labels are inputs.
"""
from __future__ import annotations

import math

import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def compile_snapshots(model, goal_latent, emb_context, action_context, *, future_horizon,
                      action_lower, action_upper, damping, checkpoints=(1, 2, 4, 8), accept_update=None,
                      initial_params=None):
    if (not checkpoints or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
            or any(not isinstance(step, int) or step < 1 for step in checkpoints)):
        raise ValueError("ordered unique positive iteration checkpoints required")
    if future_horizon < 1 or not math.isfinite(damping) or damping <= 0:
        raise ValueError("positive horizon and finite positive damping required")
    if emb_context.shape[0] != 1 or emb_context.shape[:2] != action_context.shape[:2]:
        raise ValueError("exactly one aligned latent/action context required")
    action_dim = int(action_context.shape[-1])
    flat = torch.zeros(future_horizon * action_dim, device=emb_context.device, dtype=emb_context.dtype)
    if initial_params is not None:
        if (not isinstance(initial_params, torch.Tensor)
                or initial_params.shape != (1, future_horizon, action_dim)
                or initial_params.device != emb_context.device or initial_params.dtype != emb_context.dtype
                or not torch.isfinite(initial_params).all()):
            raise ValueError("finite initial parameters with exact horizon, dtype and device required")
        # Preserve the planned action witness, not its old autograd graph.
        # The default zero-start path and its historical receipts are unchanged.
        flat = initial_params.detach().reshape(-1).clone()
    goal_coordinate = p0._planning_coordinate(model, goal_latent, role="goal")
    trace, snapshots, evaluations, rows = [], {}, 0, 0
    stopped = False
    accept = p1._accept_suffix_update if accept_update is None else accept_update

    def terminal(value):
        nonlocal evaluations
        evaluations += 1
        latent = p1._terminal_from_state(model, emb_context, action_context,
            value.reshape(1, future_horizon, action_dim), action_lower=action_lower, action_upper=action_upper)[0]
        return p0._planning_coordinate(model, latent, role="predicted")

    for iteration in range(1, max(checkpoints) + 1):
        if not stopped:
            origin = flat.detach().requires_grad_(True)
            predicted = terminal(origin)
            jacobian = torch.autograd.functional.jacobian(terminal, origin, vectorize=True)
            if jacobian.ndim != 2 or not torch.isfinite(jacobian).all() or float(jacobian.norm()) <= 1e-10:
                raise RuntimeError("non-finite or zero iterated suffix Jacobian")
            left, singular, right_h = torch.linalg.svd(jacobian, full_matrices=False)
            residual = goal_coordinate - predicted
            update = right_h.T @ (singular / (singular.square() + float(damping)) * (left.T @ residual))
            delta, accepted, receipt = accept(
                lambda offset: terminal(flat + offset), update, predicted, goal_coordinate, step_rule="backtracking")
            flat = (flat + delta).detach()
            cost_before = float((predicted - goal_coordinate).square().mean().detach())
            cost_after = float((accepted - goal_coordinate).square().mean().detach())
            rows += int(jacobian.shape[0])
            trace.append({"iteration": iteration, "cost_before": cost_before, "cost_after": cost_after,
                "jacobian_output_dimension": int(jacobian.shape[0]), "unbounded_update_l2": float(update.norm().detach()),
                "parameter_l2_after": float(flat.norm()), "tanh_saturation_fraction": float((flat.tanh().abs() >= .99).float().mean()),
                **receipt})
            stopped = receipt["accepted_step_scale"] == 0.
        if iteration in checkpoints:
            snapshots[iteration] = (flat.reshape(1, future_horizon, action_dim).clone(), {
                "program_type": "iterated_singleton_svd_dls_suffix_v1" if accept_update is None else "iterated_svd_dls_explicit_acceptance_variant",
                "damping": float(damping),
                "iterations_requested": iteration, "iterations_executed": len(trace), "stopped_no_model_descent": stopped,
                "trace": list(trace), "terminal_forward_evaluations": evaluations,
                "model_predict_sample_evaluations": evaluations * future_horizon, "backward_vjp_rows": rows,
                "physical_labels_used": False, "cost_nonincreasing": all(item["cost_after"] <= item["cost_before"] for item in trace)})
            if initial_params is not None:
                snapshots[iteration][1]["initialization"] = "explicit_planned_suffix"
    return snapshots
