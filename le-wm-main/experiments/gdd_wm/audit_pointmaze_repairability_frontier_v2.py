"""Contract audit for the v2 PointMaze repairability frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    contract, split, outcome = report.get("control_contract", {}), report.get("split_isolation", {}), report.get("outcome", {})
    checks = {
        "v2_schema": report.get("schema") == "pointmaze_execution_grounded_repairability_frontier.v2",
        "privileged_label": contract.get("label") == "privileged_execution_supervised_state_action_control_v2",
        "outcome_execution_deferred": contract.get("outcome_execution_deferred_until_frozen") is True,
        "four_disjoint_roles": split.get("root_disjoint") is True and set(split.get("roles", {})) == {"train", "gdd", "calibration", "outcome"},
        "no_outcome_selection": report.get("preselection_contract", {}).get("outcome_cost_used_for_selection") is False and outcome.get("outcome_cost_used_for_selection") is False,
        "crossfit_diagnosed": int(report.get("diagnosis", {}).get("crossfit_train_root_count", 0)) >= 8,
        "feature_ablation_present": bool(report.get("diagnosis", {}).get("gdd_feature_ablation")),
        "verdict_present": outcome.get("preselection_prediction_verdict", {}).get("status") in {"confirmed_in_this_pilot", "not_confirmed_in_this_pilot"},
    }
    return {"schema": "pointmaze_repairability_frontier_v2_audit.v0.1", "status": "passed_contract_audit" if all(checks.values()) else "rejected_contract_audit", "checks": checks, "claim_boundary": report.get("claim_boundary")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed_contract_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
