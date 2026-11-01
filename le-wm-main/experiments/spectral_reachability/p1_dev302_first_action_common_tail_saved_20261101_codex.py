"""Saved-only verifier for the prospective dev302 common-tail test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import p1_dev301_first_action_common_tail_saved_20261101_codex as common_saved
from . import p1_dev302_first_action_common_tail_20261101_codex as source

NAME = "p1_dev302_first_action_common_tail_saved_20261101_codex"
ROOT = source.ROOT
require, sha = source.require, source.sha


def verify(prepared, prepared_sha, directory, report_sha):
    prepared = Path(prepared).resolve(strict=True)
    directory = Path(directory).resolve(strict=True)
    require(sha(prepared) == prepared_sha and source.base.read(prepared) == source.build(), "contract")
    contract = source.base.read(prepared)
    report_path = directory / "report.json"
    require(sha(report_path) == report_sha, "report SHA")
    report = source.base.read(report_path)
    require(
        report["schema"] == source.NAME + ".report.v1"
        and report["status"] == "recorded_development_holdout"
        and report["prepare_sha256"] == prepared_sha
        and report["errors"] == []
        and report["rule_frozen_from_dev301_before_dev302_physics"] is True
        and report["dev302_physical_outcomes_used_for_selection"] is False
        and report["equal_physical_action_budget"] is True
        and report["independent_confirmation"] is False
        and report["P1_admission"] is False,
        "report contract",
    )
    common_saved.physical_saved.journal(directory / "journal.jsonl", report["journal_tail"])
    specs = {row["role"]: row for row in contract["roles"]}
    rows = []
    for row, role in zip(report["rows"], source.ROLES, strict=True):
        require(
            row["role"] == role
            and row["status"] in ("horizon_completed", "native_success", "native_stopped")
            and row["error"] is None
            and row["warmup_returned"] == 10
            and source.base.read(directory / f"{role}.row.json") == row,
            "row",
        )
        expected = source.base.action_array(specs[role]["actions"])
        actual = [item["action"] for item in row["transitions"]]
        require(
            actual == expected[: row["plan_returned"]].tolist()
            and row["planned_actions_sha256"] == specs[role]["actions_sha256"]
            and row["prefix_metrics"] == row["transitions"][0]["metrics"]
            and row["final_metrics"] == row["transitions"][-1]["metrics"],
            "actions/metrics",
        )
        for point in row["observations"]:
            common_saved.physical_saved.frame(directory, point)
        rows.append({
            "role": role,
            "status": row["status"],
            "native_success": row["final_metrics"]["native_success"],
            "prefix_native_distance": row["prefix_metrics"]["native_distance"],
            "prefix_object_position_error": row["prefix_metrics"]["object_position_error"],
            "final_native_distance": row["final_metrics"]["native_distance"],
            "final_object_position_error": row["final_metrics"]["object_position_error"],
            "contact_steps": [item["step"] for item in row["transitions"] if item["n_contacts"] > 0],
        })
    require(all(sha(path) == digest for path, digest in contract["bindings"].items()), "closure")
    initial = report["rows"][0]["initial_metrics"]
    return {
        "schema": NAME + ".result.v1",
        "status": "saved_dev302_common_tail_verified_development_holdout",
        "prepared": str(prepared),
        "prepare_sha256": prepared_sha,
        "report": str(report_path),
        "report_sha256": report_sha,
        "journal_sha256": report["journal_sha256"],
        "rows": rows,
        "interpretation": {
            "rule_frozen_before_dev302_physics": True,
            "all_prefixes_leave_object_position_unchanged": all(
                row["prefix_object_position_error"] == initial["object_position_error"] for row in rows
            ),
            "all_common_tails_have_zero_contact": all(not row["contact_steps"] for row in rows),
            "all_final_object_positions_unchanged": all(
                row["final_object_position_error"] == initial["object_position_error"] for row in rows
            ),
            "mechanism_replication": False,
            "candidate_or_tail_contact_coverage_failure": True,
            "training_membership": contract["training_membership"],
            "independent_confirmation": False,
            "P1_admission": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha256", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    require(output.parent == ROOT / "results" and not os.path.lexists(output), "exclusive output")
    value = verify(args.prepared, args.prepare_sha256, args.directory, args.report_sha256)
    source.base.write_new(output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
