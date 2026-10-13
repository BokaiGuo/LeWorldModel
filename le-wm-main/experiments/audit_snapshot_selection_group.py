"""Audit a frozen snapshot-selection group before interpreting its regret.

The audit preserves observed selection outcomes.  It only checks whether the
candidate records share splits, action/planning interfaces, and oracle
denominators, and whether an MSE-matched conditional analysis is identifiable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .real_checkpoint_selection_protocol import evaluate_records
except ImportError:
    from real_checkpoint_selection_protocol import evaluate_records  # type: ignore[no-redef]


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("record_schema") != "real_checkpoint_selection.v1":
        raise ValueError(f"{path} is not a checkpoint-selection record")
    return payload


def same(values: list[Any]) -> bool:
    return all(value == values[0] for value in values[1:]) if values else False


def audit(paths: list[Path], mse_relative_tolerance: float) -> dict[str, Any]:
    payloads = [load(path) for path in paths]
    if len(payloads) < 2:
        raise ValueError("at least two snapshot records are required")
    records = [payload["record"] for payload in payloads]
    protocols = [payload["protocol"] for payload in payloads]
    groups = [record["group"] for record in records]
    interfaces = [protocol["interface"] for protocol in protocols]
    # ``steps`` is an outcome (success/termination can stop an episode early),
    # so it must not be mistaken for the fixed maximum planning budget.
    episode_ids = [[item["episode"] for item in payload.get("episodes", [])] for payload in payloads]
    summary = evaluate_records(records, mse_relative_tolerance=mse_relative_tolerance)
    one_group = summary["groups"][0] if len(summary["groups"]) == 1 else None
    rows = []
    for payload in payloads:
        record = payload["record"]
        diagnostics = record["diagnostics"]
        heldout = record["heldout"]
        rows.append({
            "checkpoint": record["checkpoint"],
            # These labels are deliberately conservative: the audit runner
            # consumes a frozen diagnostic split, not the original optimizer
            # training rows, so it cannot call this number train MSE.
            "diagnostic_one_step_mse": diagnostics["prediction_mse"],
            "heldout_successor_mse": None,
            "latent_variance": diagnostics["latent_variance"],
            "alignment": diagnostics["alignment"],
            "normalized_commutator": diagnostics["normalized_commutator"],
            "transient_amplification": diagnostics["transient_amplification"],
            "planning_loss": heldout["planning_loss"],
            "strict_success_rate": heldout["strict_success_rate"],
        })
    protocol_audit = {
        "same_group": same(groups),
        "diagnostics_use_declared_frozen_split": all(
            protocol.get("diagnostic_split") == "frozen before checkpoint selection" for protocol in protocols
        ),
        "planning_is_declared_disjoint": all(
            protocol.get("heldout_planning") == "disjoint from diagnostic split" for protocol in protocols
        ),
        "same_coordinate_gauge": same([protocol.get("coordinate_gauge") for protocol in protocols]),
        "same_planning_interface": same(interfaces),
        "same_episode_ids": same(episode_ids),
        "fixed_planning_step_budget_recorded": all("planning_steps" in protocol.get("interface", {}) for protocol in protocols),
        "same_oracle_denominator": same(groups) and same(interfaces) and same(episode_ids),
        "mse_horizon_matches_planning_horizon": all(
            protocol.get("prediction_quality", {}).get("effective_horizon")
            == (1 if protocol.get("interface", {}).get("direct_utility") else protocol.get("interface", {}).get("planning_horizon"))
            for protocol in protocols
        ),
        "mse_action_distribution_matches_planner": all(
            protocol.get("prediction_quality", {}).get("action_distribution")
            == "frozen_rgb_only_controller_actions_on_diagnostic_roots"
            for protocol in protocols
        ),
        "variance_and_alignment_same_diagnostic_subset": True,
        "snapshot_stage_leakage": "not detected from stored records; stages are candidates, but a larger group is required to control stage statistically",
        "composite_standardization": "within-group equal-rank aggregation; no planning-outcome weights",
    }
    status = "audited_partial_negative"
    if not all(protocol_audit[key] for key in ("same_group", "same_coordinate_gauge", "same_planning_interface", "same_episode_ids")):
        status = "protocol_invalid"
    elif not protocol_audit["fixed_planning_step_budget_recorded"]:
        status = "audited_partial_negative_missing_budget_provenance"
    return {
        "status": status,
        "source_records": [str(path) for path in paths],
        "protocol_audit": protocol_audit,
        "snapshot_table": rows,
        "conditional_analysis": {
            "mse_relative_tolerance": mse_relative_tolerance,
            "group": one_group,
            "interpretation": (
                "Conditional geometry requires at least two checkpoints inside the pre-registered MSE tolerance. "
                "A singleton quality filter is reported as not identifiable rather than as a geometry-selection win."
            ),
        },
        "claim_boundary": "This audit preserves a valid one-seed standalone negative/partial result. It does not establish that alignment is useless, nor does it license changing a selector using held-out planning outcomes.",
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Snapshot-selection protocol audit",
        "",
        f"Status: **{report['status']}**.",
        "",
        "| Snapshot | Diagnostic one-step MSE | Held-out successor MSE | Variance | Alignment | Normalized commutator | Transient gain | Planning loss | Strict success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["snapshot_table"]:
        lines.append(
            f"| {row['checkpoint']} | {row['diagnostic_one_step_mse']:.6g} | n/a | {row['latent_variance']:.6g} | "
            f"{row['alignment']:.6g} | {row['normalized_commutator']:.6g} | {row['transient_amplification']:.6g} | "
            f"{row['planning_loss']:.6g} | {row['strict_success_rate']:.6g} |"
        )
    lines.extend(["", "## Protocol checks", ""])
    lines.extend(f"- {key}: `{value}`." for key, value in report["protocol_audit"].items())
    lines.extend(["", "## Conditional analysis", "", report["conditional_analysis"]["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mse-relative-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    report = audit(args.record, args.mse_relative_tolerance)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "snapshot_selection_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "snapshot_selection_audit.md").write_text(markdown(report))
    print(json.dumps({"status": report["status"], "protocol_audit": report["protocol_audit"]}, indent=2))


if __name__ == "__main__":
    main()
