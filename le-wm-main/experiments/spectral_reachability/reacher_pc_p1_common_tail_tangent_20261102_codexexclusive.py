"""Model-only, fixed common-tail tangent proposals; not a P1 admission gate.

The pure callback kernel is suitable for analytic CPU tests. The Reacher adapter
requires a frozen CUDA model, a pre-authenticated goal metric and observed RGB
context; it never accepts physical outcomes or executes an environment.
"""
from __future__ import annotations

import math
import torch

RADII = (0.05, 0.10, 0.20)
NORMAL_MIX = 0.25
MARGIN = 1e-6
FD_STEP = 1e-3
FD_TOLERANCE = 0.20
EPS = 1e-10


def _check_costs(immediate, future, expected=None):
    if (immediate.ndim != 0 or future.ndim != 1 or future.numel() < 2
            or (expected is not None and future.numel() != expected)
            or not torch.isfinite(immediate) or not torch.isfinite(future).all()):
        raise ValueError("invalid immediate/common-tail costs")


def common_tails(reference):
    """Five predetermined parameter-space tails, each two 5-step packets."""
    if reference.ndim != 1 or reference.numel() != 10 or not torch.isfinite(reference).all():
        raise ValueError("expected finite 10-dimensional action-packet reference")
    basis = reference.new_tensor([[0., 0.], [.3, 0.], [-.3, 0.], [0., .3], [0., -.3]])
    return basis.repeat(1, 5).unsqueeze(1).repeat(1, 2, 1)


def q_costs(metric, predicted, target):
    """Score [batch, latent] with the existing [latent, batch] metric ABI."""
    if predicted.ndim != 2 or target.ndim != 1 or predicted.shape[1] != target.shape[0]:
        raise ValueError("Q score shape mismatch")
    return metric.apply((predicted-target).double().T).square().mean(dim=0)


def generate(origin, evaluate):
    """Fixed zero plus 6 signed tangent proposals, with exact common-tail scores.

    ``evaluate(x)`` must return one scalar immediate cost and a vector of
    per-tail final costs, in the same fixed tail order for every x. The scores
    and gradients are model predictions, not physical outcomes.
    """
    if origin.ndim != 1 or not torch.isfinite(origin).all() or not origin.is_floating_point():
        raise ValueError("invalid parameter-space origin")
    x = origin.detach().clone().requires_grad_(True)
    immediate, future = evaluate(x)
    _check_costs(immediate, future)
    count = future.numel()
    g_now = torch.autograd.grad(immediate, x, retain_graph=True)[0]
    g_tail = torch.autograd.grad((future[1:] - future[0]).mean(), x)[0]
    if not torch.isfinite(g_now).all() or not torch.isfinite(g_tail).all():
        raise ValueError("nonfinite model gradient")
    now_norm = float(g_now.norm())
    if now_norm <= EPS:
        raise ValueError("zero immediate gradient: tangent normal unavailable")
    normal = g_now / g_now.norm()
    tangent = -g_tail + (g_tail @ normal) * normal
    tangent_norm = float(tangent.norm())
    if tangent_norm <= EPS:
        raise ValueError("no first-order common-tail improvement tangent")
    tangent = (tangent / tangent.norm()).detach()
    normal = normal.detach()
    with torch.no_grad():
        plus_i, plus_f = evaluate(origin + FD_STEP * tangent)
        minus_i, minus_f = evaluate(origin - FD_STEP * tangent)
        _check_costs(plus_i, plus_f, count)
        _check_costs(minus_i, minus_f, count)
        finite = float(((plus_f[1:] - plus_f[0]).mean() -
                        (minus_f[1:] - minus_f[0]).mean()) / (2 * FD_STEP))
        analytic = float(g_tail @ tangent)
        fd_error = abs(finite - analytic) / max(abs(finite), abs(analytic), EPS)
        immediate_fd = float((plus_i - minus_i) / (2 * FD_STEP))
        if not math.isfinite(fd_error) or fd_error > FD_TOLERANCE:
            raise ValueError("common-tail tangent finite-difference check failed")
        base_now = float(immediate.detach())
        base_future = future.detach()
        proposals, rows = [origin.detach().clone()], []

        def row(index, vector, cost, tails, label):
            delta = tails - base_future
            relative_to_neutral_tail = delta[1:] - delta[0]
            return {"proposal_index": index, "label": label,
                    "parameter_delta_l2": float((vector-origin).norm()),
                    "immediate_cost": float(cost), "final_costs": tails.cpu().tolist(),
                    "per_tail_delta": delta.cpu().tolist(),
                    "active_tail_neutral_adjusted_delta": relative_to_neutral_tail.cpu().tolist(),
                    "eligible": bool(index > 0 and float(cost) >= base_now + MARGIN
                                     and float(delta.mean()) <= -MARGIN
                                     and bool((delta <= 0).all())
                                     and float(relative_to_neutral_tail.mean()) <= -MARGIN
                                     and bool((relative_to_neutral_tail <= 0).all()))}
        rows.append(row(0, proposals[0], immediate.detach(), base_future, "neutral"))
        for radius in RADII:
            for sign in (1, -1):
                direction = sign * tangent + NORMAL_MIX * normal
                direction = direction / direction.norm()
                candidate = (origin + radius * direction).detach()
                current, tails = evaluate(candidate)
                _check_costs(current, tails, count)
                proposals.append(candidate)
                rows.append(row(len(rows), candidate, current, tails,
                                f"radius_{radius:g}_tangent_{sign:+d}"))
    eligible = [r for r in rows[1:] if r["eligible"]]
    selected = min(eligible, key=lambda r: (sum(r["per_tail_delta"]), r["proposal_index"])) if eligible else None
    return torch.stack(proposals), {
        "proposal_rows": rows, "selected_proposal_index": None if selected is None else selected["proposal_index"],
        "selection": "non-immediate cost >= neutral + 1e-6; mean same-tail delta <= -1e-6; "
                     "no common tail worsens; mean active-tail neutral-adjusted delta <= -1e-6 "
                     "and none worsens; ties by fixed proposal index; no fallback",
        "gradient_check": {"analytic_neutral_adjusted_tail_directional": analytic,
                           "fd_neutral_adjusted_tail_directional": finite,
                           "relative_error": fd_error, "immediate_fd": immediate_fd,
                           "immediate_tangent_dot": float(g_now @ tangent), "step": FD_STEP},
        "logical_costs": {"evaluation_calls": 9, "prefix_predict_samples": 9,
                          "suffix_predict_samples": 9 * int(count) * 2,
                          "prefix_gradient_vjp_calls": 2},
        "scope": "model-only development proposal; common fixed suffixes, not physical suffix success",
    }


def compile_reacher(model, pixels, goal, history, metric, *, lower, upper):
    """Attach to existing Reacher packet dynamics and goal-only Q, on host CUDA."""
    if (pixels.device.type != "cuda" or goal.device.type != "cuda" or history.device.type != "cuda"
            or any(p.requires_grad for p in model.parameters()) or model.training):
        raise ValueError("frozen evaluation model and host CUDA inputs required")
    if history.ndim != 3 or history.shape[0] != 1 or history.shape[-1] != 10:
        raise ValueError("expected one Reacher action history of 10D packets")
    from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0

    origin = history.new_zeros(10)
    tails = common_tails(origin)
    target = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()

    def evaluate(point):
        embeddings, actions = p1._state_after_prefix(model, pixels, history, point.reshape(1, 1, 10),
                                                      action_lower=lower, action_upper=upper)
        current = p0._planning_coordinate(model, embeddings[:, -1], role="predicted")
        terminal = p1._terminal_from_state(model, embeddings.expand(len(tails), -1, -1),
                                           actions.expand(len(tails), -1, -1), tails,
                                           action_lower=lower, action_upper=upper)
        final = p0._planning_coordinate(model, terminal, role="predicted")
        return q_costs(metric, current, target)[0], q_costs(metric, final, target)

    bank, receipt = generate(origin, evaluate)
    receipt.update(common_suffix_parameters=tails.detach().cpu().tolist(),
                   adapter_note="caller must authenticate checkpoint, metric, RGB/history and action chart; "
                                "encoder/goal-Q/Jacobian, VJP and wall costs separately required")
    return bank[:, None, :], receipt
