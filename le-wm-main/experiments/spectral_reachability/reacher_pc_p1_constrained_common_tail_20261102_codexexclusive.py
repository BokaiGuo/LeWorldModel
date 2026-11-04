"""Fixed-budget model-only common-tail constraint search; no physical inputs."""
from __future__ import annotations

import torch

from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as prior

# Frozen development protocol; never infer these values from an outcome.
START = 0.06
RADIUS = 0.20
STEP = 0.05
UPDATES = 4
STARTS = ((0, 1), (0, -1), (1, 1), (1, -1))
MARGIN = prior.MARGIN
MIN_NORM = 1e-12


def _costs(immediate, final, expected=5):
    prior._check_costs(immediate, final, expected)


def constraint_vector(immediate, final, base_immediate, base_final):
    """Positive entries violate the original selector's exact conjunction."""
    _costs(immediate, final)
    _costs(base_immediate, base_final)
    delta = final - base_final
    adjusted = delta[1:] - delta[0]
    return torch.cat(((base_immediate + MARGIN - immediate).reshape(1),
                      (delta.mean() + MARGIN).reshape(1), delta,
                      (adjusted.mean() + MARGIN).reshape(1), adjusted))


def row(index, point, origin, immediate, final, base_immediate, base_final, label):
    _costs(immediate, final)
    delta = final - base_final
    adjusted = delta[1:] - delta[0]
    # Byte-for-byte semantics of the prior selection comparisons, including
    # inclusive endpoints and tie key; this is not a new gate.
    eligible = bool(index > 0 and float(immediate) >= float(base_immediate) + MARGIN
                    and float(delta.mean()) <= -MARGIN and bool((delta <= 0).all())
                    and float(adjusted.mean()) <= -MARGIN and bool((adjusted <= 0).all()))
    return {"proposal_index": index, "label": label,
            "parameter_delta_l2": float((point.detach()-origin.detach()).norm()),
            "immediate_cost": float(immediate), "final_costs": final.detach().cpu().tolist(),
            "per_tail_delta": delta.detach().cpu().tolist(),
            "active_tail_neutral_adjusted_delta": adjusted.detach().cpu().tolist(),
            "eligible": eligible}


def generate(origin, evaluate):
    """Evaluate fixed four starts and four updates; retain every visited point.

    Evaluate must use the same five ordered model-only suffixes at every call.
    Its immediate and final costs must have autograd paths to the input point.
    The returned bank includes neutral and all 20 visited non-neutral points.
    """
    if (origin.ndim != 1 or origin.numel() != 10 or not origin.is_floating_point()
            or not torch.isfinite(origin).all()):
        raise ValueError("expected finite 10D origin")
    with torch.no_grad():
        base_i, base_f = evaluate(origin.detach())
        _costs(base_i, base_f)
        base_i, base_f = base_i.detach(), base_f.detach()
        proposals = [origin.detach().clone()]
        rows = [row(0, proposals[0], origin, base_i, base_f, base_i, base_f, "neutral")]
    traces = []
    for axis, sign in STARTS:
        point = origin.detach().clone()
        point[axis] += sign * START
        for iteration in range(UPDATES + 1):
            point = point.detach().requires_grad_(iteration < UPDATES)
            immediate, final = evaluate(point)
            _costs(immediate, final)
            constraint = constraint_vector(immediate, final, base_i, base_f)
            violation = torch.relu(constraint)
            loss = violation.square().sum()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite constraint loss")
            proposals.append(point.detach().clone())
            rows.append(row(len(rows), point, origin, immediate.detach(), final.detach(), base_i, base_f,
                            f"axis{axis}_sign{sign:+d}_iteration{iteration}"))
            record = {"proposal_index": len(rows)-1, "loss": float(loss.detach()),
                      "max_positive_violation": float(violation.max().detach())}
            if iteration < UPDATES:
                if not loss.requires_grad:
                    raise ValueError("constraint loss disconnected from prefix")
                gradient = torch.autograd.grad(loss, point)[0]
                if not torch.isfinite(gradient).all():
                    raise ValueError("nonfinite constraint gradient")
                norm = float(gradient.norm())
                record["gradient_norm"] = norm
                with torch.no_grad():
                    update = gradient / norm if norm > MIN_NORM else torch.zeros_like(gradient)
                    shifted = point - STEP * update
                    distance = shifted - origin
                    magnitude = float(distance.norm())
                    point = origin + distance * min(1., RADIUS / max(magnitude, MIN_NORM))
            traces.append(record)
    eligible = [item for item in rows[1:] if item["eligible"]]
    winner = min(eligible, key=lambda item: (sum(item["per_tail_delta"]), item["proposal_index"])) if eligible else None
    return torch.stack(proposals), {
        "proposal_rows": rows,
        "selected_proposal_index": None if winner is None else winner["proposal_index"],
        "optimization_trace": traces,
        "fixed_search": {"starts": [list(x) for x in STARTS], "start_norm": START,
                         "updates_each": UPDATES, "step_norm": STEP, "trust_radius": RADIUS,
                         "objective": "sum of squared positive original-selector constraint violations",
                         "all_iterates_retained": True},
        "selection": "unchanged common-tail tangent selector; no fallback",
        "logical_costs": {"evaluation_calls": 21, "prefix_predict_samples": 21,
                          "suffix_predict_samples": 210, "prefix_gradient_vjp_calls": 16},
        "scope": "fixed-budget model-only development; no physical outcome selection",
    }


def compile_reacher(model, pixels, goal, history, metric, *, lower, upper):
    """Use existing observed-root JEPA adapter and goal-Q on host CUDA only."""
    if (pixels.device.type != "cuda" or goal.device.type != "cuda" or history.device.type != "cuda"
            or model.training or any(p.requires_grad for p in model.parameters())):
        raise ValueError("frozen evaluation model and host CUDA inputs required")
    if history.ndim != 3 or history.shape[0] != 1 or history.shape[-1] != 10:
        raise ValueError("expected one Reacher action history of 10D packets")
    from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0

    origin = history.new_zeros(10)
    tails = prior.common_tails(origin)
    target = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()

    def evaluate(point):
        embeddings, actions = p1._state_after_prefix(model, pixels, history, point.reshape(1, 1, 10),
                                                      action_lower=lower, action_upper=upper)
        immediate = p0._planning_coordinate(model, embeddings[:, -1], role="predicted")
        final = p1._terminal_from_state(model, embeddings.expand(5, -1, -1),
                                        actions.expand(5, -1, -1), tails,
                                        action_lower=lower, action_upper=upper)
        final = p0._planning_coordinate(model, final, role="predicted")
        return prior.q_costs(metric, immediate, target)[0], prior.q_costs(metric, final, target)

    bank, receipt = generate(origin, evaluate)
    receipt["common_suffix_parameters"] = tails.detach().cpu().tolist()
    receipt["adapter_note"] = ("caller authenticates source, goal-Q, checkpoint and action chart; "
                               "encoder, inherited Q and wall costs require separate accounting")
    return bank[:, None, :], receipt
