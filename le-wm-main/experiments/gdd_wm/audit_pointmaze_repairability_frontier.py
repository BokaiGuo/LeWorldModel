"""Audit a PointMaze repairability-frontier pilot without upgrading its claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split = report.get("split_isolation", {})
    outcome = report.get("outcome", {})
    preselection = report.get("preselection_contract", {})
    checks = {
        "privileged_control_label": report.get("control_contract", {}).get("label") == "privileged_execution_supervised_state_action_control",
        "four_root_disjoint_roles": split.get("root_disjoint") is True and set(split.get("roles", {})) == {"train", "gdd", "calibration", "outcome"},
        "outcome_not_used_for_selection": preselection.get("outcome_cost_used_for_selection") is False and outcome.get("outcome_cost_used_for_selection") is False,
        "prediction_preservation_not_overclaimed": report.get("control_contract", {}).get("prediction_preservation", {}).get("applicable") is False,
        "all_four_conditions": set(outcome.get("mean_regret", {})) == {"psd", "mlp", "naive", "routed"},
        "preselection_metrics_reported": "preselection_repair_precision" in outcome and "preselection_repair_recall" in outcome,
        "preselection_verdict_reported": outcome.get("preselection_prediction_verdict", {}).get("status") in {"confirmed_in_this_pilot", "not_confirmed_in_this_pilot"},
    }
    return {
        "schema": "pointmaze_repairability_frontier_audit.v0.1",
        "status": "passed_contract_audit" if all(checks.values()) else "rejected_contract_audit",
        "checks": checks,
        "claim_boundary": report.get("claim_boundary"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed_contract_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
