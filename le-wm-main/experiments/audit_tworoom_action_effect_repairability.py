"""Fail-closed contract audit for the TwoRoom action-effect repairability run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "tworoom_action_effect_repairability_audit.v1"
REQUIRED_ARMS = {"frozen_visual", "effect_adapter", "action_only", "no_dynamics"}


def audit(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "tworoom_action_effect_repairability.v1":
        raise ValueError("unexpected report schema")
    protocol = report.get("protocol", {})
    root_ids = protocol.get("root_ids", {})
    expected = {"train", "calibration", "outcome"}
    if set(root_ids) != expected or any(not root_ids[name] for name in expected):
        raise ValueError("report must contain non-empty train/calibration/outcome root IDs")
    groups = {name: set(map(int, values)) for name, values in root_ids.items()}
    if groups["train"] & groups["calibration"] or groups["train"] & groups["outcome"] or groups["calibration"] & groups["outcome"]:
        raise ValueError("root partitions overlap")
    candidate = protocol.get("candidate_bank", {})
    if not candidate.get("same_world_fixed_bank_all_roots") or int(candidate.get("count", 0)) < 2:
        raise ValueError("fixed candidate-bank contract is absent")
    if protocol.get("runtime_state_access") is not False or protocol.get("training_outcome_roots_read") is not False:
        raise ValueError("privileged runtime or outcome-training access detected")
    if protocol.get("outcome_used_for_selection") is not False:
        raise ValueError("held outcome selected the method")
    if protocol.get("outcome_execution_order") != (
        "frozen train/calibration adapter fit precedes first outcome counterfactual replay"
    ):
        raise ValueError("outcome execution order is not frozen-before-outcome")
    if not protocol.get("closed_loop", {}).get("all_arms_same_policy_input_and_bank"):
        raise ValueError("closed-loop input/bank contract absent")
    ranking = report.get("ranking", {})
    closed = report.get("closed_loop", {})
    if not REQUIRED_ARMS.issubset(ranking) or set(ranking) != set(closed):
        raise ValueError("missing required comparison arms")
    support_gated = bool(protocol.get("support_gated_repair", False))
    if support_gated:
        if "support_gated" not in ranking or not isinstance(report.get("support_gate"), dict):
            raise ValueError("support-gated run lacks frozen gate evidence")
        if report["support_gate"].get("outcome_labels_read") is not False:
            raise ValueError("support gate must not read outcome labels")
    elif "support_gated" in ranking:
        raise ValueError("support-gated arm is present but protocol did not declare it")
    outcome_count = len(groups["outcome"])
    for arm in REQUIRED_ARMS:
        if len(ranking[arm]["outcome"]["root_regret"]) != outcome_count:
            raise ValueError(f"{arm} lacks held-root ranking outcomes")
        if len(closed[arm]["roots"]) != outcome_count:
            raise ValueError(f"{arm} lacks held-root closed-loop outcomes")
    return {
        "schema": SCHEMA,
        "status": "passed",
        "report": str(report_path.resolve()),
        "root_counts": {name: len(group) for name, group in groups.items()},
        "candidate_count": int(candidate["count"]),
        "support_gated_repair": support_gated,
        "cuda_requested": str(protocol.get("device", "")).startswith("cuda"),
        "claim_boundary": report["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
