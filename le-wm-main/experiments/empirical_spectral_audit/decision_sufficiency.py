"""Fail-closed evidence gates for predictive/action/decision sufficiency.

This is a model-agnostic readout, not a new predictor or online selector.
Prediction improvement alone cannot authorize a planning claim.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _lower(value: Any, baseline: Any, tolerance: float) -> bool:
    return _finite(value) and _finite(baseline) and float(value) < float(baseline) - tolerance


def _higher(value: Any, baseline: Any, tolerance: float) -> bool:
    return _finite(value) and _finite(baseline) and float(value) > float(baseline) + tolerance


def assess_record(record: Mapping[str, Any], *, tolerance: float = 0.0) -> dict[str, Any]:
    """Assess one paired method/baseline result using the three-layer schema.

    Sections may be nested under ``predictive``, ``action``, and ``decision``
    or supplied as one flat mapping for easy adaptation of existing artifacts.
    """

    predictive = record.get("predictive", record)
    action = record.get("action", record)
    decision = record.get("decision", record)

    def get(section: Any, name: str) -> Any:
        return section.get(name) if isinstance(section, Mapping) else None

    predictive_gain = _lower(get(predictive, "method_error"), get(predictive, "baseline_error"), tolerance)
    action_gain = _lower(
        get(action, "method_ranking_regret"), get(action, "baseline_ranking_regret"), tolerance
    )
    decision_gain = _lower(
        get(decision, "method_planning_risk"), get(decision, "baseline_planning_risk"), tolerance
    )
    mode_gain = _higher(get(action, "method_mode_accuracy"), get(action, "baseline_mode_accuracy"), tolerance)
    pose_gain = _higher(get(decision, "method_pose_success"), get(decision, "baseline_pose_success"), tolerance)

    missing = []
    for section_name, section, names in (
        ("predictive", predictive, ("method_error", "baseline_error")),
        ("action", action, ("method_ranking_regret", "baseline_ranking_regret")),
        ("decision", decision, ("method_planning_risk", "baseline_planning_risk")),
    ):
        for name in names:
            if not _finite(get(section, name)):
                missing.append(f"{section_name}.{name}")

    flags: list[str] = []
    if predictive_gain and not action_gain:
        flags.append("prediction_improvement_without_action_ranking_improvement")
    if mode_gain and not pose_gain:
        flags.append("mode_accuracy_improvement_without_pose_closure")
    if not action_gain:
        flags.append("action_identifiability_not_improved")
    if not decision_gain:
        flags.append("independent_planning_consequence_not_improved")

    if missing:
        status = "incomplete_fail_closed"
    elif flags:
        status = "boundary_or_negative"
    elif predictive_gain and action_gain and decision_gain:
        status = "three_layer_positive"
    else:
        status = "non_positive_or_inconclusive"
    return {
        "schema": "decision_sufficiency_record.v1",
        "status": status,
        "layers": {
            "predictive_sufficiency": "improved" if predictive_gain else "not_improved",
            "action_identifiability": "improved" if action_gain else "not_improved",
            "decision_sufficiency": "improved" if decision_gain else "not_improved",
        },
        "diagnostic_boundary_flags": flags,
        "missing_fields": missing,
        "positive_claim_permitted": status == "three_layer_positive",
    }


def assess_panel(records: list[Mapping[str, Any]], *, tolerance: float = 0.0) -> dict[str, Any]:
    """Aggregate without averaging a failed record away."""

    reports = [assess_record(record, tolerance=tolerance) for record in records]
    positive = sum(bool(report["positive_claim_permitted"]) for report in reports)
    return {
        "schema": "decision_sufficiency_panel.v1",
        "status": "three_layer_panel_positive" if reports and positive == len(reports) else "gated_panel",
        "record_count": len(reports),
        "complete_record_count": sum(not report["missing_fields"] for report in reports),
        "three_layer_positive_count": positive,
        "positive_claim_permitted": bool(reports) and positive == len(reports),
        "records": reports,
        "claim_boundary": "Prediction, action ranking, and an independent planning consequence are all required.",
    }
