"""Forensic consistency audit for PushT action-conditioned torque reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "pusht.action_conditioned_signed_torque_confirmation_audit.v1"
REPORT_SCHEMA = "pusht.action_conditioned_signed_torque_confirmation.v1"


def _estimate(report: dict[str, Any], representation: str) -> float:
    try:
        return float(report["results"][representation]["test_balanced_accuracy"]["estimate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing test balanced-accuracy estimate for {representation}") from error


def audit(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unexpected action-conditioned torque report schema")
    protocol = report.get("protocol", {})
    selected = protocol.get("confirmation_representation", "geometry_plus_candidate_pair")
    if selected not in {
        "geometry_plus_candidate_pair", "geometry_plus_pair_mechanics", "geometry_plus_rigid_body_mechanics",
    }:
        raise ValueError("unknown confirmation representation")
    delta = report.get("results", {}).get("paired_action_conditioning_delta")
    if not isinstance(delta, dict):
        raise ValueError("missing paired action-conditioning delta")
    expected_direction = f"{selected} minus geometry_history"
    reported_direction = delta.get("direction")
    selected_delta = _estimate(report, str(selected)) - _estimate(report, "geometry_history")
    reported_estimate = float(delta.get("estimate"))
    numerical_match = abs(selected_delta - reported_estimate) <= 1.0e-8
    direction_match = reported_direction == expected_direction
    return {
        "schema": SCHEMA,
        "status": "passed_consistent_report" if direction_match and numerical_match else "rejected_report_inconsistency",
        "report": str(report_path.resolve()),
        "confirmation_representation": selected,
        "reported_direction": reported_direction,
        "expected_direction": expected_direction,
        "reported_estimate": reported_estimate,
        "selected_accuracy_difference": selected_delta,
        "numerical_delta_matches_selected_representation": numerical_match,
        "direction_matches_selected_representation": direction_match,
        "claim_boundary": (
            "Forensic report-consistency audit only. A passed audit does not establish independent replication, "
            "online action selection, pose closure, or planning benefit."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite torque report audit: {args.output}")
    result = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
