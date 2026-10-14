"""Candidate-wise task contrasts and paired decision-transfer certificates.

The certificate is intentionally conditional: it applies only to a declared
candidate bank, exchangeable calibration roots, and a passed action-support
gate.  It certifies a pairwise ordering from a calibrated contrast-error
envelope; it is not a global or per-state conditional guarantee.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _quantile_higher(values: np.ndarray, probability: float) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        raise ValueError("at least one calibration value is required")
    index = min(values.size - 1, max(0, int(np.ceil(float(probability) * values.size)) - 1))
    return float(values[index])


def pairwise_task_contrast_audit(
    real_cost: np.ndarray,
    predicted_cost: np.ndarray,
    group_id: np.ndarray,
    *,
    minimum_decision_margin: float = 0.0,
) -> dict[str, Any]:
    """Compare all within-root task-cost contrasts.

    Lower cost is better.  The primary error is
    ``|(real_i-real_j)-(pred_i-pred_j)|``; pairs with a small real margin are
    excluded from inversion statistics but remain visible in pair counts.
    """

    real = np.asarray(real_cost, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_cost, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_id).reshape(-1)
    if not (len(real) == len(predicted) == len(groups)):
        raise ValueError("real_cost, predicted_cost, and group_id must be aligned")
    if not np.isfinite(real).all() or not np.isfinite(predicted).all():
        raise ValueError("costs must be finite")
    margin = float(minimum_decision_margin)
    if margin < 0.0 or not np.isfinite(margin):
        raise ValueError("minimum_decision_margin must be finite and non-negative")

    errors: list[float] = []
    comparable_errors: list[float] = []
    inversions = 0
    comparable = 0
    pair_count = 0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                pair_count += 1
                i, j = indices[left], indices[right]
                real_delta = float(real[i] - real[j])
                predicted_delta = float(predicted[i] - predicted[j])
                error = abs(real_delta - predicted_delta)
                errors.append(error)
                if abs(real_delta) > margin:
                    comparable += 1
                    comparable_errors.append(error)
                    inversions += int(np.sign(real_delta) != np.sign(predicted_delta))
    return {
        "schema": "paired_task_contrast_audit.v1",
        "group_count": int(len(np.unique(groups))),
        "pair_count": int(pair_count),
        "comparable_pair_count": int(comparable),
        "contrast_error_mean": float(np.mean(errors)) if errors else None,
        "contrast_error_max": float(np.max(errors)) if errors else None,
        "comparable_contrast_error_mean": float(np.mean(comparable_errors)) if comparable_errors else None,
        "inversion_count": int(inversions),
        "inversion_rate": None if comparable == 0 else float(inversions / comparable),
        "minimum_decision_margin": margin,
        "claim_boundary": "candidate-wise task contrast; no causal or closed-loop claim without independent execution",
    }


def calibrate_contrast_bound(
    real_cost: np.ndarray,
    predicted_cost: np.ndarray,
    group_id: np.ndarray,
    calibration_groups: Iterable[Any],
    *,
    alpha: float = 0.1,
    minimum_decision_margin: float = 0.0,
) -> dict[str, Any]:
    """Calibrate a root-level maximum pairwise contrast bound.

    Each calibration root contributes its maximum pairwise contrast error.
    The high quantile is deliberately conservative for planner-selected
    candidate sets.  Calibration roots are not reused as outcome roots.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    real = np.asarray(real_cost, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_cost, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_id).reshape(-1)
    calibration_set = set(calibration_groups)
    if not calibration_set:
        raise ValueError("calibration_groups must not be empty")
    root_errors: list[float] = []
    used_groups: list[Any] = []
    for group in np.unique(groups):
        if group not in calibration_set:
            continue
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            continue
        report = pairwise_task_contrast_audit(
            real[indices], predicted[indices], np.repeat(group, len(indices)),
            minimum_decision_margin=minimum_decision_margin,
        )
        if report["contrast_error_max"] is not None:
            root_errors.append(float(report["contrast_error_max"]))
            used_groups.append(group)
    if not root_errors:
        raise ValueError("no calibration root contains at least two candidates")
    bound = _quantile_higher(np.asarray(root_errors), 1.0 - float(alpha))
    return {
        "schema": "paired_task_contrast_calibration.v1",
        "alpha": float(alpha),
        "bound": bound,
        "calibration_root_count": len(root_errors),
        "calibration_groups_used": [str(group) for group in used_groups],
        "root_max_contrast_errors": root_errors,
        "simultaneous_over_candidate_bank": True,
        "claim_boundary": "marginal root-level calibration under exchangeable declared roots; not conditional coverage under planner shift",
    }


def certify_pair(
    predicted_cost_i: float,
    predicted_cost_j: float,
    contrast_bound: float | None,
    *,
    support_passed: bool,
    candidate_ids: tuple[Any, Any] = (0, 1),
) -> dict[str, Any]:
    """Return a conditional permit/probe/abstain decision for a pair."""

    if not np.isfinite(float(predicted_cost_i)) or not np.isfinite(float(predicted_cost_j)):
        raise ValueError("predicted costs must be finite")
    if contrast_bound is not None and (not np.isfinite(float(contrast_bound)) or float(contrast_bound) < 0.0):
        raise ValueError("contrast_bound must be finite and non-negative")
    predicted_margin = float(predicted_cost_j - predicted_cost_i)
    if not support_passed:
        status = "abstain"
        reason = "action_or_state_support_gate_failed"
    elif contrast_bound is None:
        status = "probe"
        reason = "contrast_bound_missing"
    elif predicted_margin > float(contrast_bound):
        status = "permit"
        reason = "predicted_margin_exceeds_calibrated_contrast_bound"
    else:
        status = "probe"
        reason = "contrast_margin_not_separated_by_bound"
    return {
        "schema": "paired_decision_transfer_certificate.v1",
        "status": status,
        "reason": reason,
        "preferred_candidate": candidate_ids[0] if predicted_margin > 0.0 else candidate_ids[1],
        "candidate_ids": list(candidate_ids),
        "predicted_cost_margin_j_minus_i": predicted_margin,
        "contrast_bound": None if contrast_bound is None else float(contrast_bound),
        "support_passed": bool(support_passed),
        "claim_boundary": "conditional pairwise ranking release; no universal or arbitrary-state guarantee",
    }
