"""Development-only task-weighted response SVD proposals.

The callback returns the immediate scalar cost, fixed-tail terminal costs,
and task-weighted terminal coordinates.  This module never sees physical
outcomes and does not change the strict P1 selection contract.
"""
from __future__ import annotations

import math
import torch

from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as common
from experiments.spectral_reachability import reacher_pc_p1_constrained_common_tail_20261102_codexexclusive as strict

RADII = (0.05, 0.10, 0.20)
MAX_MODES = 2
NORMAL_MIX = 0.25
EPS = 1e-10
FD_STEP = 1e-3


def generate(origin, evaluate):
    """Generate bounded proposals from the prefix-to-tail response Jacobian.

    ``evaluate(x)`` returns (immediate, per-tail costs, [tail, feature]
    task-weighted coordinates. Tail zero is the predeclared control.
    Coordinates are differenced against tail zero before differentiation,
    removing state changes common to all suffixes. No physical data are used.
    """
    if (origin.ndim != 1 or not origin.is_floating_point()
            or not torch.isfinite(origin).all()):
        raise ValueError("finite vector origin required")
    origin = origin.detach()

    def checked(x):
        immediate, costs, coordinates = evaluate(x)
        strict._costs(immediate, costs)
        if (coordinates.ndim != 2 or coordinates.shape[0] != costs.numel()
                or coordinates.shape[1] < 1 or not torch.isfinite(coordinates).all()):
            raise ValueError("invalid terminal coordinates")
        return immediate, costs, coordinates

    def response(x):
        _, _, coordinates = checked(x)
        return (coordinates[1:] - coordinates[:1]).reshape(-1)

    point = origin.clone().requires_grad_(True)
    immediate0, costs0, coordinates0 = checked(point)
    gradient = torch.autograd.grad(immediate0, point)[0].detach()
    norm = float(gradient.norm())
    if not math.isfinite(norm) or norm <= EPS:
        raise ValueError("immediate cost has no usable normal")
    normal = gradient / norm
    # Input-space JVPs avoid batching hundreds of latent-output reverse passes.
    # Each call is still accounted separately; internal autograd work is not.
    columns = []
    for index in range(origin.numel()):
        tangent = torch.zeros_like(origin)
        tangent[index] = 1
        _, column = torch.autograd.functional.jvp(response, origin, tangent)
        columns.append(column)
    jacobian = torch.stack(columns, dim=1)
    if jacobian.ndim != 2 or not torch.isfinite(jacobian).all():
        raise ValueError("invalid response Jacobian")
    normal_j = normal.to(dtype=jacobian.dtype)
    projected = jacobian - (jacobian @ normal_j).unsqueeze(1) * normal_j.unsqueeze(0)
    _, singular, right = torch.linalg.svd(projected, full_matrices=False)
    valid = [i for i in range(min(MAX_MODES, len(singular))) if float(singular[i]) > EPS]
    if not valid:
        raise ValueError("no nonzero tangent response modes")

    direction = right[valid[0]].to(dtype=origin.dtype).detach()
    with torch.no_grad():
        finite = (response(origin + FD_STEP * direction) -
                  response(origin - FD_STEP * direction)) / (2 * FD_STEP)
        linear = jacobian @ direction.to(dtype=jacobian.dtype)
        fd_error = float((finite - linear).norm() /
                         torch.maximum(finite.norm(), linear.norm()).clamp_min(EPS))
        if not math.isfinite(fd_error) or fd_error > common.FD_TOLERANCE:
            raise ValueError("response Jacobian finite-difference check failed")
        base_i, base_f = immediate0.detach(), costs0.detach()
        bank = [origin.clone()]
        rows = [strict.row(0, origin, origin, base_i, base_f, base_i, base_f, "neutral")]
        for mode in valid:
            for sign in (1, -1):
                vector = sign * right[mode].to(dtype=origin.dtype) + NORMAL_MIX * normal
                vector = vector / vector.norm()
                for radius in RADII:
                    candidate = origin + radius * vector
                    current, finals, _ = checked(candidate)
                    bank.append(candidate.detach().clone())
                    rows.append(strict.row(len(rows), candidate, origin, current, finals,
                                           base_i, base_f,
                                           f"mode_{mode}_sign_{sign:+d}_radius_{radius:g}"))
    eligible = [row for row in rows[1:] if row["eligible"]]
    selected = min(eligible, key=lambda row: (sum(row["per_tail_delta"]),
                                             row["proposal_index"])) if eligible else None
    return torch.stack(bank), {
        "proposal_rows": rows,
        "strict_selected_index": None if selected is None else selected["proposal_index"],
        "response_jacobian_shape": list(jacobian.shape),
        "singular_values": singular.detach().cpu().tolist(),
        "mode_indices": valid,
        "finite_difference_relative_error": fd_error,
        "finite_difference_step": FD_STEP,
        "baseline_coordinate_shape": list(coordinates0.shape),
        "logical_evaluations": 1 + origin.numel() + 2 + (len(bank) - 1),
        "jvp_calls": origin.numel(),
        "jacobian_output_rows": int(jacobian.shape[0]),
        "scope": "model-only development; no physical evidence or P1 admission",
    }


def compile_reacher(model, pixels, goal, history, metric, *, lower, upper):
    """Bind to the existing frozen CUDA Reacher model and goal-only metric."""
    if (pixels.device.type != "cuda" or goal.device.type != "cuda"
            or history.device.type != "cuda" or model.training
            or any(p.requires_grad for p in model.parameters())):
        raise ValueError("frozen evaluation model and host CUDA inputs required")
    if history.ndim != 3 or history.shape[0] != 1 or history.shape[-1] != 10:
        raise ValueError("expected one Reacher action history of 10D packets")
    from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0

    origin = history.new_zeros(10)
    tails = common.common_tails(origin)
    target = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()

    def evaluate(point):
        embeddings, actions = p1._state_after_prefix(
            model, pixels, history, point.reshape(1, 1, 10),
            action_lower=lower, action_upper=upper)
        immediate = p0._planning_coordinate(model, embeddings[:, -1], role="predicted")
        terminal = p1._terminal_from_state(
            model, embeddings.expand(len(tails), -1, -1),
            actions.expand(len(tails), -1, -1), tails,
            action_lower=lower, action_upper=upper)
        final = p0._planning_coordinate(model, terminal, role="predicted")
        weighted = metric.apply((final - target).double().T).T
        return common.q_costs(metric, immediate, target)[0], weighted.square().mean(dim=1), weighted

    # JVP uses reverse-over-reverse in torch.autograd.functional. CUDA's
    # efficient SDPA backward does not implement the required derivative.
    # Scope the math backend to this compiler; do not mutate global settings.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        bank, receipt = generate(origin, evaluate)
    receipt["common_suffix_parameters"] = tails.detach().cpu().tolist()
    receipt["attention_backend"] = "math_for_reverse_over_reverse_jvp"
    receipt["cost_boundary"] = ("logical calls and Jacobian output rows only; "
                                "encoder, actual autograd work, memory and wall time not certified")
    return bank[:, None, :], receipt
