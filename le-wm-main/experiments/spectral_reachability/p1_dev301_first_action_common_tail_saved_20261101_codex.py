"""Saved-only verification of the dev301 first-action/common-tail diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev301_blind_plan_physical_saved_20261101_codex as physical_saved
from . import p1_dev301_first_action_common_tail_20261101_codex as source

NAME = "p1_dev301_first_action_common_tail_saved_20261101_codex"
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
        and report["status"] == "recorded_development"
        and report["prepare_sha256"] == prepared_sha
        and report["errors"] == []
        and report["physical_outcomes_used_to_design_diagnostic"] is True
        and report["physical_outcomes_used_to_choose_within_run"] is False
        and report["equal_budget_claim"] is True
        and report["independent_confirmation"] is False
        and report["P1_admission"] is False,
        "report contract",
    )
    events = physical_saved.journal(directory / "journal.jsonl", report["journal_tail"])
    require(
        sha(directory / "journal.jsonl") == report["journal_sha256"]
        and events[0]["kind"] == "run_started"
        and events[-1]["kind"] == "run_finished",
        "journal",
    )
    specs = {row["role"]: row for row in contract["roles"]}
    rows = []
    common_tail = None
    for row, role in zip(report["rows"], source.ROLES, strict=True):
        require(
            row["role"] == role
            and row["status"] in ("horizon_completed", "native_success", "native_stopped")
            and row["error"] is None
            and row["warmup_returned"] == 10,
            "row identity",
        )
        require(source.base.read(directory / f"{role}.row.json") == row, "row shard")
        expected = source.base.action_array(specs[role]["actions"])
        actual = np.asarray([item["action"] for item in row["transitions"]], np.float32)
        require(
            np.array_equal(actual, expected[: row["plan_returned"]])
            and row["planned_actions_sha256"] == specs[role]["actions_sha256"]
            and row["planned_actions_sha256"] == hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest(),
            "action identity",
        )
        if common_tail is None:
            common_tail = expected[1:].copy()
        else:
            require(np.array_equal(expected[1:], common_tail), "exact common tail")
        require(
            row["prefix_metrics"] == row["transitions"][0]["metrics"]
            and row["prefix_state"] == row["transitions"][0]["state"]
            and row["final_metrics"] == row["transitions"][-1]["metrics"],
            "prefix/final metrics",
        )
        observations = row["observations"]
        require(
            [point["step"] for point in observations[:3]] == [0, 5, 10]
            and [point["phase"] for point in observations[:3]] == ["warmup"] * 3
            and [point["step"] for point in observations[3:]]
            == list(range(1, row["plan_returned"] + 1)),
            "observation roster",
        )
        for point in observations:
            physical_saved.frame(directory, point)
        rows.append(
            {
                "role": role,
                "status": row["status"],
                "native_success": row["final_metrics"]["native_success"],
                "prefix_native_distance": row["prefix_metrics"]["native_distance"],
                "prefix_object_position_error": row["prefix_metrics"]["object_position_error"],
                "final_native_distance": row["final_metrics"]["native_distance"],
                "final_object_position_error": row["final_metrics"]["object_position_error"],
                "final_angle_error": row["final_metrics"]["wrapped_object_angle_error"],
                "contact_steps": [
                    item["step"] for item in row["transitions"] if item["n_contacts"] > 0
                ],
            }
        )
    require(
        all(sha(path) == digest for path, digest in contract["bindings"].items()),
        "source closure",
    )
    by_role = {row["role"]: row for row in rows}
    d2p, zero, gn = (by_role[role] for role in source.ROLES)
    initial_object = report["rows"][0]["initial_metrics"]["object_position_error"]
    return {
        "schema": NAME + ".result.v1",
        "status": "saved_first_action_common_tail_verified_development_only",
        "prepared": str(prepared),
        "prepare_sha256": prepared_sha,
        "report": str(report_path),
        "report_sha256": report_sha,
        "journal_sha256": report["journal_sha256"],
        "rows": rows,
        "interpretation": {
            "all_prefixes_leave_object_position_unchanged": all(
                row["prefix_object_position_error"] == initial_object for row in rows
            ),
            "d2p_prefix_native_distance_worsens": d2p["prefix_native_distance"]
            > report["rows"][0]["initial_metrics"]["native_distance"],
            "d2p_prefix_improves_common_tail_object_position_vs_zero": d2p["final_object_position_error"]
            < zero["final_object_position_error"],
            "d2p_prefix_improves_common_tail_object_position_vs_gn": d2p["final_object_position_error"]
            < gn["final_object_position_error"],
            "d2p_prefix_advances_first_contact": min(d2p["contact_steps"])
            < min(zero["contact_steps"])
            == min(gn["contact_steps"]),
            "any_native_success": any(row["native_success"] for row in rows),
            "mechanism_development_evidence": True,
            "outcome_informed_design": True,
            "independent_confirmation": False,
            "P1_admission": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
