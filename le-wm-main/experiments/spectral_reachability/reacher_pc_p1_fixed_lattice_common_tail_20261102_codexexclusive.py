"""Fixed model-only Reacher candidate lattice with the unchanged strict selector.

The candidate bank is fixed before evaluation.  It contains coordinate and
balanced sign directions, not gradients, physical outcomes, or hidden state.
"""
from __future__ import annotations

import torch

MARGIN = 1e-6
DIMENSION = 10
RADIUS = 0.10


def _check_costs(immediate, final):
    if (immediate.ndim != 0 or final.ndim != 1 or final.numel() != 5
            or not torch.isfinite(immediate) or not torch.isfinite(final).all()):
        raise ValueError("invalid fixed-lattice common-tail costs")


def _balanced_directions(dtype=torch.float64):
    # A deterministic 10x10 sign lattice; no data or model quantity enters it.
    rows = []
    for i in range(DIMENSION):
        rows.append([1.0 if ((i & j).bit_count() % 2 == 0) else -1.0
                     for j in range(DIMENSION)])
    return torch.tensor(rows, dtype=dtype) / DIMENSION ** 0.5


def fixed_directions(dtype=torch.float64):
    """Return 40 ordered unit directions: coordinates, then balanced signs."""
    eye = torch.eye(DIMENSION, dtype=dtype)
    balanced = _balanced_directions(dtype)
    return torch.cat((eye, -eye, balanced, -balanced), dim=0)


def _row(index, point, origin, immediate, final, base_immediate, base_final, label):
    _check_costs(immediate, final)
    delta = final - base_final
    adjusted = delta[1:] - delta[0]
    eligible = bool(
        index > 0
        and float(immediate) >= float(base_immediate) + MARGIN
        and float(delta.mean()) <= -MARGIN
        and bool((delta <= 0).all())
        and float(adjusted.mean()) <= -MARGIN
        and bool((adjusted <= 0).all())
    )
    return {
        "proposal_index": index,
        "label": label,
        "parameter_delta_l2": float((point.detach() - origin.detach()).norm()),
        "immediate_cost": float(immediate),
        "final_costs": final.detach().cpu().tolist(),
        "per_tail_delta": delta.detach().cpu().tolist(),
        "active_tail_neutral_adjusted_delta": adjusted.detach().cpu().tolist(),
        "eligible": eligible,
    }


def generate(origin, evaluate):
    """Evaluate a fixed 41-row bank and apply only the prior strict selector.

    ``evaluate`` receives only a parameter-space point and returns model
    predictions.  It cannot pass physical outcomes or privileged state to the
    selector.  Every point is retained; no fallback is selected.
    """
    if (origin.ndim != 1 or origin.numel() != DIMENSION
            or not origin.is_floating_point() or not torch.isfinite(origin).all()):
        raise ValueError("expected finite 10D parameter origin")
    directions = fixed_directions(origin.dtype).to(origin.device)
    points = [origin.detach().clone()]
    with torch.no_grad():
        base_immediate, base_final = evaluate(points[0])
        _check_costs(base_immediate, base_final)
        base_immediate, base_final = base_immediate.detach(), base_final.detach()
        rows = [_row(0, points[0], origin, base_immediate, base_final,
                     base_immediate, base_final, "neutral")]
        for index, direction in enumerate(directions, start=1):
            point = (origin + RADIUS * direction).detach()
            immediate, final = evaluate(point)
            _check_costs(immediate, final)
            points.append(point)
            rows.append(_row(index, point, origin, immediate, final,
                             base_immediate, base_final, f"lattice_{index:02d}"))
    eligible = [item for item in rows[1:] if item["eligible"]]
    winner = min(eligible, key=lambda item: (sum(item["per_tail_delta"]),
                                             item["proposal_index"])) if eligible else None
    return torch.stack(points), {
        "proposal_rows": rows,
        "selected_proposal_index": None if winner is None else winner["proposal_index"],
        "selection": "unchanged strict common-tail selector; no fallback",
        "fixed_bank": {
            "directions": 40,
            "coordinate_directions": 20,
            "balanced_sign_directions": 20,
            "radius": RADIUS,
            "all_rows_retained": True,
        },
        "logical_costs": {
            "evaluation_calls": 41,
            "prefix_predict_samples": 41,
            "suffix_predict_samples": 410,
            "prefix_gradient_vjp_calls": 0,
        },
        "scope": "fixed model-only development; no physical or privileged-state input",
    }


def compile_reacher(model, pixels, goal, history, metric, *, lower, upper):
    """Compile against the same frozen-model Q and common tails as prior runs."""
    if (pixels.device.type != "cuda" or goal.device.type != "cuda"
            or history.device.type != "cuda" or model.training
            or any(p.requires_grad for p in model.parameters())):
        raise ValueError("frozen evaluation model and host CUDA inputs required")
    if history.ndim != 3 or history.shape[0] != 1 or history.shape[-1] != 10:
        raise ValueError("expected one Reacher action history of 10D packets")
    from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
    from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as prior

    origin = history.new_zeros(10)
    tails = prior.common_tails(origin)
    target = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()

    def evaluate(point):
        embeddings, actions = p1._state_after_prefix(
            model, pixels, history, point.reshape(1, 1, 10), action_lower=lower, action_upper=upper)
        immediate = p0._planning_coordinate(model, embeddings[:, -1], role="predicted")
        final = p1._terminal_from_state(
            model, embeddings.expand(5, -1, -1), actions.expand(5, -1, -1),
            tails, action_lower=lower, action_upper=upper)
        final = p0._planning_coordinate(model, final, role="predicted")
        return prior.q_costs(metric, immediate, target)[0], prior.q_costs(metric, final, target)

    bank, receipt = generate(origin, evaluate)
    receipt["common_suffix_parameters"] = tails.detach().cpu().tolist()
    return bank[:, None, :], receipt
