"""H4/20-step candidate primitives and explicit prospective cost schedule.

Pure tensor/array utilities only: no runner installation, model, or physical
execution.  Charges in ``node_cost`` are predictions, not measurements.
"""
import math

import numpy as np
import torch

ARMS = ("structured", "native", "random", "immediate_progress")
AXES = ("encode_calls", "encode_samples", "action_encode_calls", "predict_calls",
        "predict_samples", "backward_calls", "vjp_rows", "jacobian_calls",
        "selection_evaluations", "solve_calls", "svd_calls", "replay_calls",
        "physical_steps")


def dimensions(boundary):
    if type(boundary) is not int or boundary not in (0, 5, 10, 15):
        raise ValueError("boundary must be 0, 5, 10, or 15")
    return 4 - boundary // 5, boundary != 0


def response_bank(jac, projection, goal, nominal, horizon):
    """H4 analogue of zero/DLS/+/- top-three Q-response directions."""
    if type(horizon) is not int or horizon not in (1, 2, 3, 4):
        raise ValueError("invalid horizon")
    jac, projection = np.asarray(jac, np.float64), np.asarray(projection, np.float64)
    goal, nominal = np.asarray(goal, np.float64), np.asarray(nominal, np.float64)
    if (jac.shape != (192, 40) or projection.shape != (192, 192)
            or goal.shape != (192,) or nominal.shape != (192,)
            or not all(np.isfinite(x).all() for x in (jac, projection, goal, nominal))
            or np.any(jac[:, 10 * horizon:] != 0)):
        raise ValueError("invalid or nonzero inactive Jacobian")
    matrix, residual = projection @ jac[:, :10 * horizon], projection @ (goal - nominal)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    directions = np.zeros((8, 40), dtype=np.float64)
    if singular[0] != 0:
        damping = .01 * singular[0] ** 2
        dls = right.T @ ((singular / (singular ** 2 + damping)) * (left.T @ residual))
        directions[1, :10 * horizon] = dls / max(1., np.max(np.abs(dls)))
        for i in range(3):
            vector = right[i].copy()
            if vector[np.argmax(abs(vector))] < 0:
                vector *= -1
            vector *= .5 / max(np.max(abs(vector)), 1e-30)
            directions[2 + 2*i, :10 * horizon] = vector
            directions[3 + 2*i, :10 * horizon] = -vector
    return directions.astype(np.float32).reshape(8, 4, 5, 2), singular


def normalized_update(raw, gradient, horizon, boundary, peak=.05):
    """Mask inactive packets and protect packet 0 only at boundary 0."""
    if (type(horizon) is not int or dimensions(boundary)[0] != horizon
            or not isinstance(raw, torch.Tensor) or not isinstance(gradient, torch.Tensor)
            or raw.shape != gradient.shape or raw.ndim != 4 or raw.shape[1:] != (4, 5, 2)
            or raw.device != gradient.device or raw.dtype != gradient.dtype
            or not raw.is_floating_point() or not math.isfinite(peak) or not 0 < peak <= 1
            or not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(gradient).all())
            or not bool((raw.abs() <= 1).all())):
        raise ValueError("invalid normalized update")
    if bool(torch.count_nonzero(gradient[:, horizon:])):
        raise ValueError("inactive gradient must be zero")
    grad = gradient.detach().clone()
    grad[:, horizon:] = 0
    if boundary == 0:
        grad[:, 0] = 0
    # Per candidate normalization is equivalent to the existing one-candidate optimizer.
    norm = grad.abs().amax(dim=(1, 2, 3), keepdim=True)
    step = peak * grad / norm.clamp_min(torch.finfo(grad.dtype).tiny)
    result = (raw - step).clamp(-1, 1).detach()
    if boundary == 0 and not torch.equal(result[:, 0], raw[:, 0]):
        raise RuntimeError("protected prefix changed")
    return result


def cross_bank(bank):
    if (not isinstance(bank, torch.Tensor) or bank.shape != (8, 4, 5, 2)
            or not bank.is_floating_point() or not bool(torch.isfinite(bank).all())
            or not bool((bank.abs() <= 1).all())):
        raise ValueError("expected bounded H4 eight-plan bank")
    return torch.cat((bank[:, None, :1].expand(8, 8, 1, 5, 2),
                      bank[None, :, 1:].expand(8, 8, 3, 5, 2)), dim=2).reshape(64, 4, 5, 2).clone()


def exchange(bank, horizon, score):
    """Select one of eight tails per prefix, using all 64 same-horizon scores."""
    if type(horizon) is not int or horizon not in (2, 3, 4) or not callable(score):
        raise ValueError("exchange requires H2-H4 and scorer")
    candidates = cross_bank(bank)
    scores = [float(score(plan[:horizon])) for plan in candidates]
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("nonfinite exchange score")
    tails = [min(range(8), key=lambda j: (scores[8*i+j], j != i, j)) for i in range(8)]
    selected = torch.stack([candidates[8*i+j] for i, j in enumerate(tails)])
    return selected, scores, tails


def feedback_select(bank, previous, issued, boundary, score):
    """Rescore eight fresh plans and the exact old tail at the current horizon.

    The caller must verify RGB, actual-action provenance, episode and boundary
    identity using the physical driver before invoking this pure function.
    """
    horizon, carry = dimensions(boundary)
    if not carry or not callable(score):
        raise ValueError("feedback boundary and scorer required")
    cross_bank(bank)  # Validate the complete fixed-width fresh bank.
    if (not isinstance(previous, torch.Tensor) or previous.shape != (horizon + 1, 5, 2)
            or not isinstance(issued, torch.Tensor) or issued.shape != (5, 2)
            or previous.device != bank.device or previous.dtype != bank.dtype
            or issued.device != bank.device or issued.dtype != bank.dtype
            or not bool(torch.isfinite(previous).all()) or not bool((previous.abs() <= 1).all())
            or not torch.equal(previous[0], issued)):
        raise ValueError("previous plan or actual packet mismatch")
    candidates = torch.cat((bank[:, :horizon], previous[None, 1:]), dim=0)
    scores = [float(score(plan)) for plan in candidates]
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("nonfinite feedback score")
    winner = min(range(9), key=lambda i: (scores[i], i != 8, i))
    return candidates[winner].clone(), scores, winner


def node_cost(arm, boundary, steps=6):
    """Prospective planner-only operation counts; no physical or visual cost."""
    horizon, carry = dimensions(boundary)
    if arm not in ARMS or type(steps) is not int or steps < 1:
        raise ValueError("invalid arm or steps")
    result = dict.fromkeys(AXES, 0)
    calls = 8 * (steps + 1) * horizon
    result.update(action_encode_calls=calls, predict_calls=calls, predict_samples=calls,
                  backward_calls=8 * steps, vjp_rows=8 * steps,
                  replay_calls=8, selection_evaluations=8)
    if arm == "structured":
        for key in ("action_encode_calls", "predict_calls", "predict_samples"):
            result[key] += horizon
        result["backward_calls"] += 192
        result["vjp_rows"] += 192
        result["jacobian_calls"] = result["svd_calls"] = 1
    if arm == "immediate_progress":
        for key in ("action_encode_calls", "predict_calls", "predict_samples",
                    "replay_calls", "selection_evaluations"):
            result[key] += 16
    if horizon > 1:
        for key in ("action_encode_calls", "predict_calls", "predict_samples"):
            result[key] += 64 * horizon
        result["replay_calls"] += 64
        result["selection_evaluations"] += 64
    if carry:
        for key in ("action_encode_calls", "predict_calls", "predict_samples"):
            result[key] += 9 * horizon
        result["replay_calls"] += 9
        result["selection_evaluations"] += 9
    return result
