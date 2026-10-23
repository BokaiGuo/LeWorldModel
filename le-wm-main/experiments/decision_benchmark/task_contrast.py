"""Candidate-wise task contrasts and paired decision-transfer certificates.

The certificate is intentionally conditional: it applies only to a declared
candidate bank, exchangeable calibration roots, and a passed action-support
gate.  It certifies a pairwise ordering from a calibrated contrast-error
envelope; it is not a global or per-state conditional guarantee.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Iterable

import numpy as np


def _split_conformal_upper_quantile(
    values: np.ndarray,
    *,
    alpha: float,
) -> tuple[float | None, int]:
    """Return the finite-sample split-conformal upper quantile.

    For ``n`` exchangeable calibration roots the valid order statistic is
    ``ceil((n + 1) * (1 - alpha))``.  If that rank is larger than ``n``, no
    finite distribution-free upper bound is available from this calibration
    set; callers must fail closed instead of silently substituting the sample
    maximum.
    """
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        raise ValueError("at least one calibration value is required")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = int(np.ceil((values.size + 1) * (1.0 - float(alpha))))
    if rank > values.size:
        return None, rank
    return float(values[rank - 1]), rank


def finite_sample_marginal_coverage(
    calibration_root_count: int,
    *,
    alpha: float,
) -> dict[str, Any]:
    """Describe the exact split-conformal coverage implied by one rank.

    The returned quantity is a *marginal* coverage lower bound over a fresh,
    exchangeable root.  It is not an empirical confidence interval and it is
    not a permit-conditional guarantee.  Returning this metadata prevents a
    finite order statistic from being mistaken for a conditional certificate.
    """

    count = int(calibration_root_count)
    if count < 1:
        raise ValueError("calibration_root_count must be positive")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = int(np.ceil((count + 1) * (1.0 - float(alpha))))
    finite = rank <= count
    return {
        "alpha": float(alpha),
        "target_marginal_coverage": float(1.0 - float(alpha)),
        "finite_sample_rank": rank,
        "finite_bound_available": finite,
        "finite_sample_marginal_coverage_lower_bound": (
            None if not finite else float(rank / (count + 1))
        ),
        "calibration_root_count": count,
    }


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z_value: float = 1.959963984540054,
) -> tuple[float, float] | None:
    """Return a deterministic 95% Wilson interval for an outcome proportion."""

    if total < 1:
        return None
    if successes < 0 or successes > total:
        raise ValueError("successes must lie between zero and total")
    proportion = float(successes / total)
    z_squared = float(z_value * z_value)
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = z_value * sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * total)) / total
    ) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def calibrate_root_score_bound(
    scores: np.ndarray,
    *,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Calibrate a finite-sample upper bound for one scalar risk per root."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("scores must contain finite non-negative root risks")
    bound, rank = _split_conformal_upper_quantile(values, alpha=alpha)
    coverage = finite_sample_marginal_coverage(values.size, alpha=alpha)
    return {
        "schema": "root_score_calibration.v1",
        "alpha": float(alpha),
        "bound": bound,
        "status": "ok" if bound is not None else "insufficient_calibration",
        "finite_sample_rank": int(rank),
        "finite_sample_marginal_coverage_lower_bound": coverage[
            "finite_sample_marginal_coverage_lower_bound"
        ],
        "target_marginal_coverage": coverage["target_marginal_coverage"],
        "finite_bound_available": coverage["finite_bound_available"],
        "calibration_root_count": int(values.size),
        "root_scores": values.tolist(),
        "claim_boundary": "marginal scalar root-risk bound under exchangeability; not state-conditional coverage",
    }


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
    observed_groups = set(groups.tolist())
    missing_groups = sorted(str(group) for group in calibration_set - observed_groups)
    singleton_groups: list[str] = []
    root_errors: list[float] = []
    used_groups: list[Any] = []
    # Iterate the declared panel, not only observed groups.  Silently dropping
    # a requested root makes a partial calibration set look like the promised
    # finite-sample panel and can falsely turn an insufficient panel into a
    # certificate.
    for group in sorted(calibration_set, key=str):
        if group not in observed_groups:
            continue
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            singleton_groups.append(str(group))
            continue
        report = pairwise_task_contrast_audit(
            real[indices], predicted[indices], np.repeat(group, len(indices)),
            minimum_decision_margin=minimum_decision_margin,
        )
        if report["contrast_error_max"] is not None:
            root_errors.append(float(report["contrast_error_max"]))
            used_groups.append(group)
    if missing_groups or singleton_groups:
        return {
            "schema": "paired_task_contrast_calibration.v2",
            "alpha": float(alpha), "bound": None, "status": "incomplete_calibration_panel",
            "finite_sample_rank": None, "finite_sample_marginal_coverage_lower_bound": None,
            "target_marginal_coverage": float(1.0 - float(alpha)), "finite_bound_available": False,
            "calibration_root_count": len(root_errors),
            "calibration_groups_requested": [str(group) for group in sorted(calibration_set, key=str)],
            "calibration_groups_used": [str(group) for group in used_groups],
            "missing_calibration_groups": missing_groups, "singleton_calibration_groups": singleton_groups,
            "root_max_contrast_errors": root_errors, "simultaneous_over_candidate_bank": True,
            "claim_boundary": "incomplete declared calibration panel; no finite candidate-bank certificate is available",
        }
    if not root_errors:
        raise ValueError("no calibration root contains at least two candidates")
    bound, conformal_rank = _split_conformal_upper_quantile(
        np.asarray(root_errors), alpha=float(alpha),
    )
    coverage = finite_sample_marginal_coverage(len(root_errors), alpha=float(alpha))
    return {
        "schema": "paired_task_contrast_calibration.v2",
        "alpha": float(alpha),
        "bound": bound,
        "status": "ok" if bound is not None else "insufficient_calibration",
        "finite_sample_rank": int(conformal_rank),
        "finite_sample_marginal_coverage_lower_bound": coverage[
            "finite_sample_marginal_coverage_lower_bound"
        ],
        "target_marginal_coverage": coverage["target_marginal_coverage"],
        "finite_bound_available": coverage["finite_bound_available"],
        "calibration_root_count": len(root_errors),
        "calibration_groups_requested": [str(group) for group in sorted(calibration_set, key=str)],
        "calibration_groups_used": [str(group) for group in used_groups],
        "missing_calibration_groups": [], "singleton_calibration_groups": [],
        "root_max_contrast_errors": root_errors,
        "simultaneous_over_candidate_bank": True,
        "claim_boundary": "marginal root-level calibration under exchangeable declared roots and one declared fixed candidate bank; insufficient calibration fails closed; not conditional coverage under planner shift",
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


def certify_candidate_bank(
    predicted_cost: np.ndarray,
    contrast_bound: float | None,
    *,
    support_passed: bool,
    candidate_ids: Iterable[Any] | None = None,
    candidate_bank_mode: str = "fixed",
) -> dict[str, Any]:
    """Jointly certify the predicted winner against an entire fixed bank.

    The same root-level calibration event is shared by every comparison.  A
    bank is released only when the predicted winner beats *all* competitors by
    more than the calibrated bound.  This is a fixed-bank marginal rule; it
    does not provide conditional coverage for candidates generated adaptively
    after observing model scores (for example, CEM proposals).
    """

    costs = np.asarray(predicted_cost, dtype=np.float64).reshape(-1)
    if costs.size < 2 or not np.isfinite(costs).all():
        raise ValueError("predicted_cost must contain at least two finite values")
    if contrast_bound is not None and (
        not np.isfinite(float(contrast_bound)) or float(contrast_bound) < 0.0
    ):
        raise ValueError("contrast_bound must be finite and non-negative")
    ids = list(range(int(costs.size))) if candidate_ids is None else list(candidate_ids)
    if len(ids) != costs.size:
        raise ValueError("candidate_ids must align with predicted_cost")
    if candidate_bank_mode not in {"fixed", "adaptive"}:
        raise ValueError("candidate_bank_mode must be either 'fixed' or 'adaptive'")
    winner_index = int(np.argmin(costs))
    competitor_indices = [index for index in range(costs.size) if index != winner_index]
    margins = costs[competitor_indices] - costs[winner_index]
    minimum_margin = float(np.min(margins))
    if candidate_bank_mode == "adaptive":
        status = "abstain"
        reason = "adaptive_candidate_generation_not_certified"
    elif not support_passed:
        status = "abstain"
        reason = "action_or_state_support_gate_failed"
    elif contrast_bound is None:
        status = "probe"
        reason = "contrast_bound_missing_or_insufficient_calibration"
    elif bool(np.all(margins > float(contrast_bound))):
        status = "permit"
        reason = "predicted_winner_separated_from_all_competitors"
    else:
        status = "probe"
        reason = "at_least_one_competitor_not_separated_by_bound"
    # On the simultaneous fixed-bank coverage event, for every competitor j:
    #   J(winner) - J(j) <= q - (Jhat(j) - Jhat(winner)).
    # Taking the maximum positive part yields a joint regret bound for the
    # declared bank.  This mathematical statement is unconditional over the
    # declared root distribution; a failed support gate still blocks *release*
    # but does not erase the post-hoc fixed-bank inequality.
    joint_regret_bound = None
    if candidate_bank_mode == "fixed" and contrast_bound is not None:
        joint_regret_bound = float(max(0.0, float(contrast_bound) - minimum_margin))
    if candidate_bank_mode != "fixed":
        regret_status = "unavailable_adaptive_candidate_generation"
    elif contrast_bound is None:
        regret_status = "unavailable_insufficient_calibration"
    elif not support_passed:
        regret_status = "valid_on_simultaneous_fixed_bank_coverage_event_but_release_support_gate_failed"
    else:
        regret_status = "valid_on_simultaneous_fixed_bank_coverage_event"
    return {
        "schema": "candidate_bank_decision_transfer_certificate.v1",
        "status": status,
        "reason": reason,
        "preferred_candidate": ids[winner_index],
        "candidate_ids": ids,
        "competitor_count": len(competitor_indices),
        "minimum_predicted_margin": minimum_margin,
        "predicted_competitor_margins": [float(value) for value in margins],
        "contrast_bound": None if contrast_bound is None else float(contrast_bound),
        "support_passed": bool(support_passed),
        "candidate_bank_mode": candidate_bank_mode,
        "joint_regret_upper_bound": joint_regret_bound,
        "joint_regret_bound_status": regret_status,
        "joint_regret_bound_derivation": (
            "max_j (q - (Jhat_j - Jhat_winner))_+ under the simultaneous "
            "root-level fixed-bank contrast-coverage event"
        ),
        "claim_boundary": "fixed declared candidate-bank marginal release and regret bound; no permit-conditional, state-conditional, or adaptive-candidate guarantee",
    }


def certify_candidate_bank_regret_budget(
    predicted_cost: np.ndarray,
    contrast_bound: float | None,
    *,
    regret_budget: float,
    support_passed: bool,
    candidate_ids: Iterable[Any] | None = None,
    candidate_bank_mode: str = "fixed",
) -> dict[str, Any]:
    """Certify a fixed-bank choice against a declared *regret* budget.

    A unique predicted winner is unnecessarily strong when the physical cost
    landscape has several near-tied actions.  On the same simultaneous
    fixed-bank contrast event used by :func:`certify_candidate_bank`, the
    selected predicted winner has regret at most ``max(0, q - margin)``.
    This rule releases only if that already-derived bound is no larger than a
    predeclared task-cost budget ``epsilon``.  It does not weaken conformal
    calibration, use outcome labels, or turn marginal coverage into a
    state-conditional/permit-conditional guarantee.
    """

    epsilon = float(regret_budget)
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("regret_budget must be finite and non-negative")
    base = certify_candidate_bank(
        predicted_cost,
        contrast_bound,
        support_passed=support_passed,
        candidate_ids=candidate_ids,
        candidate_bank_mode=candidate_bank_mode,
    )
    bound = base["joint_regret_upper_bound"]
    if candidate_bank_mode == "adaptive":
        status, reason = "abstain", "adaptive_candidate_generation_not_certified"
    elif not support_passed:
        status, reason = "abstain", "action_or_state_support_gate_failed"
    elif bound is None:
        status, reason = "probe", "regret_bound_missing_or_insufficient_calibration"
    elif float(bound) <= epsilon:
        status, reason = "permit", "predicted_winner_within_declared_regret_budget"
    else:
        status, reason = "probe", "joint_regret_bound_exceeds_declared_budget"
    return {
        **base,
        "schema": "candidate_bank_regret_budget_certificate.v1",
        "status": status,
        "reason": reason,
        "regret_budget": epsilon,
        "unique_winner_certificate_status": base["status"],
        "unique_winner_certificate_reason": base["reason"],
        "claim_boundary": (
            "fixed declared candidate-bank marginal epsilon-regret release; "
            "not permit-conditional, state-conditional, or adaptive-candidate coverage"
        ),
    }


def evaluate_contrast_bound(
    real_cost: np.ndarray,
    predicted_cost: np.ndarray,
    group_id: np.ndarray,
    contrast_bound: float | None,
) -> dict[str, Any]:
    """Evaluate a frozen root-level contrast bound on independent roots.

    Coverage is the fraction of outcome roots whose *maximum* candidate-pair
    contrast error is no larger than the calibration bound.  When calibration
    did not yield a finite bound, empirical coverage is deliberately left
    undefined rather than treating abstention as coverage.
    """

    real = np.asarray(real_cost, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_cost, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_id).reshape(-1)
    if not (len(real) == len(predicted) == len(groups)):
        raise ValueError("real_cost, predicted_cost, and group_id must be aligned")
    if not np.isfinite(real).all() or not np.isfinite(predicted).all():
        raise ValueError("costs must be finite")
    if contrast_bound is not None and (
        not np.isfinite(float(contrast_bound)) or float(contrast_bound) < 0.0
    ):
        raise ValueError("contrast_bound must be finite and non-negative")

    roots: list[dict[str, Any]] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            continue
        report = pairwise_task_contrast_audit(
            real[indices], predicted[indices], np.repeat(group, len(indices)),
        )
        root_error = float(report["contrast_error_max"])
        root_real = real[indices]
        root_predicted = predicted[indices]
        predicted_winner = int(np.argmin(root_predicted))
        realized_oracle = int(np.argmin(root_real))
        realized_regret = float(root_real[predicted_winner] - root_real[realized_oracle])
        competitor_margins = np.delete(root_predicted - root_predicted[predicted_winner], predicted_winner)
        minimum_margin = float(np.min(competitor_margins))
        joint_regret_bound = (
            None if contrast_bound is None
            else float(max(0.0, float(contrast_bound) - minimum_margin))
        )
        roots.append({
            "group_id": str(group),
            "max_contrast_error": root_error,
            "covered": None if contrast_bound is None else bool(root_error <= float(contrast_bound)),
            "realized_regret_of_predicted_winner": realized_regret,
            "joint_regret_upper_bound_on_coverage_event": joint_regret_bound,
            "regret_bound_holds_when_covered": (
                None if contrast_bound is None or root_error > float(contrast_bound)
                else bool(realized_regret <= float(joint_regret_bound) + 1.0e-12)
            ),
        })
    if not roots:
        raise ValueError("no outcome root contains at least two candidates")
    covered_count = None if contrast_bound is None else int(sum(bool(row["covered"]) for row in roots))
    root_count = len(roots)
    interval = None if covered_count is None else _wilson_interval(covered_count, root_count)
    regret_bound_checks = [
        row["regret_bound_holds_when_covered"] for row in roots
        if row["regret_bound_holds_when_covered"] is not None
    ]
    return {
        "schema": "paired_task_contrast_coverage.v1",
        "status": "ok" if contrast_bound is not None else "not_evaluable_insufficient_calibration",
        "contrast_bound": None if contrast_bound is None else float(contrast_bound),
        "outcome_root_count": root_count,
        "covered_root_count": covered_count,
        "empirical_root_coverage": None if covered_count is None else float(covered_count / root_count),
        "empirical_root_miscoverage": None if covered_count is None else float(1.0 - covered_count / root_count),
        "empirical_root_coverage_wilson_95_interval": (
            None if interval is None else {"lower": interval[0], "upper": interval[1]}
        ),
        "covered_root_regret_bound_check_count": int(len(regret_bound_checks)),
        "covered_root_regret_bound_all_hold": (
            None if not regret_bound_checks else bool(all(regret_bound_checks))
        ),
        "roots": roots,
        "claim_boundary": "empirical independent-root fixed-bank marginal coverage and coverage-event regret checks; neither is state-conditional, permit-conditional, nor adaptive-CEM coverage",
    }
