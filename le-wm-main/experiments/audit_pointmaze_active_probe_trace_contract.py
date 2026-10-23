"""Audit whether an active-probe aggregate agrees with its raw RGB traces.

An aggregate report can describe a task-safe active probe while its per-root
trace still records a previous full-plane implementation.  This audit treats
the raw trace as authoritative: it reconstructs the paired arm report and
reuses the current strict trace validator.  A mismatch invalidates the active
probe's mechanism claim; it is not silently reinterpreted as a scientific
negative result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_pointmaze_htsc_active_probe_strict import (
    ACTIVE_METHOD,
    DEPLOYABLE_METHODS,
    expected_probe_design_for_method,
    validate_active_probe_report,
)


SCHEMA = "pointmaze.active_probe_trace_contract_audit.v1"
AFFINE_ACTIVE_METHODS = frozenset({
    "effect_adapter_rgb_online_affine_verified_local",
    "effect_adapter_rgb_online_affine_candidate_anchored_verified_local",
})
TASK_SAFE_PROBE_DESIGNS = frozenset({
    "task_aligned_positive_half_plane_with_predicted_progress_gate",
    "candidate_anchored_task_safe_affine_d_optimal",
    "candidate_anchored_task_safe_prior_residual_rank2",
})
CANDIDATE_ANCHORED_AFFINE_METHOD = "effect_adapter_rgb_online_affine_candidate_anchored_verified_local"


def _expected_probe_design() -> str:
    """Return the immutable geometry promised by the selected active arm."""

    return expected_probe_design_for_method(ACTIVE_METHOD)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _raw_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = report.get("artifacts", {}).get("held_out_arm_reports")
    if not isinstance(artifacts, dict) or set(artifacts) != set(DEPLOYABLE_METHODS):
        raise ValueError("aggregate lacks exactly one raw held-out report for every declared arm")
    rows: list[dict[str, Any]] = []
    for method in DEPLOYABLE_METHODS:
        payload = _load(Path(str(artifacts[method])))
        method_rows = payload.get("rows")
        if not isinstance(method_rows, list) or not method_rows:
            raise ValueError(f"{method}: raw held-out report lacks rows")
        if any(str(row.get("method")) != method for row in method_rows if isinstance(row, dict)):
            raise ValueError(f"{method}: raw report contains a mismatched method label")
        rows.extend(method_rows)
    return rows


def _trace_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw active-arm traces without trusting the aggregate label.

    This intentionally does not depend on the aggregate's declared budget, so
    an older report that omitted its budget cannot hide a separate geometry or
    pre-action safety violation.
    """

    active_rows = [row for row in rows if str(row.get("method")) == ACTIVE_METHOD]
    events = [
        event
        for row in active_rows
        for event in row.get("active_probe_trace", [])
        if isinstance(event, dict) and "probe_index" in event and "error" not in event
    ]
    runtime_safety_rejections = [
        event
        for row in active_rows
        for event in row.get("active_probe_trace", [])
        if isinstance(event, dict) and bool(event.get("pre_execution_safety_rejection", False))
    ]
    designs: dict[str, int] = {}
    for event in events:
        design = str(event.get("probe_design", "<missing>"))
        designs[design] = designs.get(design, 0) + 1
    missing_progress = sum("predicted_goal_progress" not in event for event in events)
    nonpositive_progress = sum(
        float(event.get("predicted_goal_progress", 0.0)) <= 0.0 for event in events
    )
    nonpositive_projection = sum(
        float(event.get("action_goal_projection", 0.0)) <= 0.0 for event in events
    )
    return {
        "active_root_count": len(active_rows),
        "raw_probe_event_count": len(events),
        "probe_design_counts": dict(sorted(designs.items())),
        "missing_predicted_goal_progress_count": int(missing_progress),
        "nonpositive_predicted_goal_progress_count": int(nonpositive_progress),
        "nonpositive_action_goal_projection_count": int(nonpositive_projection),
        "runtime_safety_rejection_count": int(len(runtime_safety_rejections)),
        "invalid_runtime_safety_rejection_count": int(sum(
            float(event.get("action_goal_projection", float("inf"))) > 0.0
            and float(event.get("predicted_goal_progress", float("inf"))) > 0.0
            for event in runtime_safety_rejections
        )),
    }


def _raw_trace_violations(evidence: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    event_count = int(evidence["raw_probe_event_count"])
    designs = evidence["probe_design_counts"]
    if event_count and not set(designs) <= TASK_SAFE_PROBE_DESIGNS:
        violations.append(
            "raw active-arm trace does not use the declared task-safe probe geometry"
        )
    if event_count and set(designs) != {_expected_probe_design()}:
        violations.append("raw active-arm trace does not use its method-specific frozen probe geometry")
    if int(evidence["missing_predicted_goal_progress_count"]):
        violations.append("raw active-arm trace lacks predicted_goal_progress safety evidence")
    if int(evidence["nonpositive_predicted_goal_progress_count"]):
        violations.append("raw active-arm trace contains a nonpositive predicted-progress probe")
    if int(evidence["nonpositive_action_goal_projection_count"]):
        violations.append("raw active-arm trace contains a non-goal-aligned probe action")
    if int(evidence.get("invalid_runtime_safety_rejection_count", 0)):
        violations.append("runtime task-safety rejection did not record an unsafe pulse")
    return violations


def _affine_loo_contract_violations(
    rows: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> list[str]:
    """Bind V4's finite local-chart gate to calibration and raw decisions."""

    if ACTIVE_METHOD not in AFFINE_ACTIVE_METHODS:
        return []
    selected = frozen.get("local_chart_loo_tolerance")
    try:
        threshold = float(selected)
    except (TypeError, ValueError):
        return ["affine active arm lacks a finite calibration-frozen LOO threshold"]
    if not np.isfinite(threshold) or threshold <= 0.0:
        return ["affine active arm lacks a finite calibration-frozen LOO threshold"]
    violations: list[str] = []
    active_rows = [row for row in rows if str(row.get("method")) == ACTIVE_METHOD]
    for row in active_rows:
        contract = row.get("policy_contract", {})
        observed = contract.get("local_chart_loo_tolerance")
        try:
            observed_value = float(observed)
        except (TypeError, ValueError):
            observed_value = float("nan")
        if not np.isfinite(observed_value) or not np.isclose(observed_value, threshold):
            violations.append("affine active arm did not execute its frozen LOO threshold")
            break
        for item in row.get("trace", []):
            gate = item.get("action_transfer_gate", {}) if isinstance(item, dict) else {}
            if not bool(gate.get("local_affine_recovery_considered", False)):
                continue
            gate_threshold = gate.get("local_affine_chart_loo_tolerance")
            try:
                gate_value = float(gate_threshold)
            except (TypeError, ValueError):
                gate_value = float("nan")
            if not np.isfinite(gate_value) or not np.isclose(gate_value, threshold):
                violations.append("affine local release did not use its frozen LOO threshold")
                return violations
    return violations


def audit(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema") != "pointmaze.htsc_active_probe.strict_rgb_closed_loop.v1":
        raise ValueError("audit requires an active-probe strict RGB aggregate")
    split = report.get("split_protocol", {})
    frozen = report.get("frozen_probe", {})
    expected_roots = split.get("held_out_test_roots")
    budget = report.get("split_protocol", {}).get("total_physical_block_budget")
    if budget is None:
        budget = report.get("protocol", {}).get("total_physical_block_budget")
    if budget is None:
        budget = report.get("budget_contract", {}).get("total_physical_block_budget")
    # The older aggregate records this under split_protocol only in newer
    # revisions; a missing budget is an audit rejection, never an assumption.
    if not isinstance(expected_roots, list) or not expected_roots:
        raise ValueError("aggregate lacks held-out roots")
    directions = frozen.get("probe_directions")
    magnitude = frozen.get("probe_magnitude", 0.5)
    trigger = frozen.get("active_probe_trigger", "conditional_state_support_failure")
    ambiguity_margin = frozen.get("active_probe_ambiguity_margin", 0.0)
    if not isinstance(directions, int):
        raise ValueError("aggregate lacks frozen active probe direction count")
    rows = _raw_rows(report)
    evidence = _trace_evidence(rows)
    violations: list[str] = []
    if budget is None:
        violations.append("aggregate lacks its declared shared physical block budget")
    violations.extend(_raw_trace_violations(evidence))
    violations.extend(_affine_loo_contract_violations(rows, frozen))
    validated: dict[int, dict[str, dict[str, Any]]] | None = None
    if budget is not None:
        try:
            validated = validate_active_probe_report(
                {"rows": rows},
                expected_roots=[int(value) for value in expected_roots],
                total_block_budget=int(budget),
                selected_probe_directions=int(directions),
                selected_probe_magnitude=float(magnitude),
                selected_probe_trigger=str(trigger),
                selected_probe_ambiguity_margin=float(ambiguity_margin),
            )
        except (KeyError, TypeError, ValueError) as error:
            if str(error) not in violations:
                violations.append(str(error))
    if violations:
        return {
            "schema": SCHEMA,
            "status": "rejected_trace_contract",
            "trace_contract_passed": False,
            "reason": "; ".join(violations),
            "violations": violations,
            "raw_trace_evidence": evidence,
            "claim_boundary": (
                "Raw-trace protocol audit only. A rejection invalidates the aggregate active-probe mechanism claim; "
                "it does not establish that a conforming task-safe probe would be beneficial or harmful."
            ),
            "expected_contract": {
                "probe_design": _expected_probe_design(),
                "positive_predicted_progress": True,
                "stop_after_nonpositive_observed_progress": True,
                "shared_total_block_budget": None if budget is None else int(budget),
                "finite_calibration_frozen_affine_loo_threshold": (
                    ACTIVE_METHOD in AFFINE_ACTIVE_METHODS
                ),
            },
        }
    if validated is None:  # Defensive: a passing audit must have run the strict validator.
        raise RuntimeError("trace contract audit passed without strict validator output")
    active_rows = [validated[int(root)][ACTIVE_METHOD] for root in expected_roots]
    return {
        "schema": SCHEMA,
        "status": "passed_trace_contract",
        "trace_contract_passed": True,
        "active_root_count": len(active_rows),
        "raw_trace_evidence": evidence,
        "claim_boundary": (
            "Raw-trace protocol audit only. A pass validates declared action safety/accounting; it does not establish local-rerank efficacy."
        ),
        "expected_contract": {
            "probe_design": _expected_probe_design(),
            "positive_predicted_progress": True,
            "stop_after_nonpositive_observed_progress": True,
            "shared_total_block_budget": int(budget),
            "finite_calibration_frozen_affine_loo_threshold": (
                ACTIVE_METHOD in AFFINE_ACTIVE_METHODS
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite trace audit: {args.output}")
    result = audit(_load(args.report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "reason": result.get("reason"), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
