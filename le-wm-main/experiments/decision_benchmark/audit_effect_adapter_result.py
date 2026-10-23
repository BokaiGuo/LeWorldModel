"""Fail-closed result audit for a materialized PointMaze adapter replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def audit(input_audit: Path, replication: Path, output: Path) -> dict[str, Any]:
    inputs, result = _load(input_audit), _load(replication)
    root_count = int(inputs.get("root_counts", {}).get("outcome", 0))
    protocol = result.get("protocol", {})
    controls = result.get("controls", {}).get("paired_regret_gain", {})
    checks = {
        "input_materialization_valid": inputs.get("status") == "materialized_valid_held_root_replication_inputs",
        "held_root_semantics_declared": inputs.get("replication_semantics") == "held-root replication; not newly collected external data",
        "outcome_excluded_from_fit_and_calibration": protocol.get("outcome_used_for_training_calibration_or_selection") is False,
        "frozen_latent_action_runtime_input": protocol.get("feature_mode") == "frozen_latent_action",
        "three_paired_controls_present": set(controls) == {"bank_only", "shuffled_prediction", "no_dynamics"},
        "paired_control_root_units_match_held_panel": all(
            int(item.get("groups", 0)) == root_count for item in controls.values() if isinstance(item, dict)
        ),
    }
    positive = {
        name: bool(float(item.get("lower", float("-inf"))) > 0.0)
        for name, item in controls.items() if isinstance(item, dict)
    }
    checks["all_paired_controls_have_positive_bootstrap_lower_bound"] = (
        set(positive) == {"bank_only", "shuffled_prediction", "no_dynamics"} and all(positive.values())
    )
    report = {
        "schema": "pointmaze_effect_adapter_replication_result_audit.v0.1",
        "status": "reproduced_bounded_positive_control" if all(checks.values()) else "rejected_fail_closed",
        "input_audit": str(input_audit.resolve()),
        "replication": str(replication.resolve()),
        "checks": checks,
        "paired_regret_gain": controls,
        "claim_boundary": (
            "A root-disjoint held-split intervention-supervised ranking control only; it is not an "
            "independent external collection, frozen predictor-dynamics, closed-loop MPC, or universal transfer result."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-audit", required=True, type=Path)
    parser.add_argument("--replication", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.input_audit, args.replication, args.output)
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
    return 0 if report["status"] == "reproduced_bounded_positive_control" else 2


if __name__ == "__main__":
    raise SystemExit(main())
