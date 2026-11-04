"""Model-action Jacobian candidates for development, not a PC gate."""
from __future__ import annotations

import torch

from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def spectral_candidates(terminal, origin, goal, *, radius: float, modes: int, damping: float):
    """Compile symmetric action-coordinate singular directions and a DLS step.

    ``terminal`` is a differentiable, task-chart prediction from flattened
    bounded-action parameters. Nonlinear rollout selection is deliberately
    separate; this routine does not claim physical reconfiguration.
    """
    if origin.ndim != 1 or goal.ndim != 1 or radius <= 0 or modes < 1 or damping <= 0:
        raise ValueError("invalid spectral candidate contract")
    origin = origin.detach().requires_grad_(True)
    predicted = terminal(origin)
    if predicted.shape != goal.shape or not torch.isfinite(predicted).all():
        raise ValueError("terminal and goal coordinates must be finite and aligned")
    jacobian = torch.autograd.functional.jacobian(terminal, origin, vectorize=True)
    if jacobian.ndim != 2 or not torch.isfinite(jacobian).all() or not bool(jacobian.norm() > 1e-10):
        raise ValueError("invalid or degenerate action response")
    left, singular, right = torch.linalg.svd(jacobian, full_matrices=False)
    count = min(modes, len(singular))
    # Include strongest and weakest modes; remove overlaps deterministically.
    indices = list(dict.fromkeys(list(range(count)) + list(range(len(singular) - count, len(singular)))))
    direction = right[indices[0]].detach()
    epsilon = 1e-3
    with torch.no_grad():
        fd = (terminal(origin + epsilon * direction) - terminal(origin - epsilon * direction)) / (2 * epsilon)
    linear = jacobian @ direction
    fd_relative_error = float((fd - linear).norm() / linear.norm().clamp_min(1e-12))
    residual = goal - predicted
    dls = right.T @ ((singular / (singular.square() + damping)) * (left.T @ residual))
    dls = dls * min(1.0, radius / max(float(dls.norm().detach()), 1e-12))
    candidates = [origin.detach(), (origin + dls).detach()]
    for index in indices:
        candidates.extend([(origin + radius * right[index]).detach(),
                           (origin - radius * right[index]).detach()])
    bank = torch.stack(candidates)
    return bank, {
        "singular_values": singular.detach().cpu().tolist(),
        "mode_indices": indices,
        "jacobian_shape": list(jacobian.shape),
        "backward_vjp_rows": int(jacobian.shape[0]),
        "jacobian_primal_rollouts": 1,
        "finite_difference_primal_rollouts": 2,
        "finite_difference_epsilon": epsilon,
        "finite_difference_direction_relative_error": fd_relative_error,
        "radius_parameter_l2": float(radius),
        "dls_norm": float(dls.norm()),
        "candidate_count": len(candidates),
        "note": "local parameter-space directions; nonlinear and physical utility unproven",
    }


def compile_reacher(model, pixels, goal, history, *, lower, upper, radius, modes, damping):
    if history.shape[0] != 1:
        raise ValueError("only one observed root supported")
    dim = history.shape[-1]
    goal_coordinate = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()
    def terminal(flat):
        prediction = p0._terminal_batch(
            model, pixels, goal, history, flat.reshape(1, 1, dim),
            action_lower=lower, action_upper=upper,
        )[0]
        return p0._planning_coordinate(model, prediction, role="predicted")
    origin = torch.zeros(dim, dtype=history.dtype, device=history.device)
    bank, receipt = spectral_candidates(terminal, origin, goal_coordinate, radius=radius, modes=modes, damping=damping)
    return bank[:, None, :], receipt
