"""Saved-only verification for the actual-cp5 feedback CEM physical run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_unified_cp5_feedback_physical_20261101_codex as source

ROOT = source.ROOT
NAME = "p1_unified_cp5_feedback_physical_saved_20261101_codex"
engine = source.engine
require, sha = engine.require, engine.sha


def _verify_frame(directory: Path, point: dict, expected: dict, returned: dict | None) -> None:
    path = Path(point["path"]).resolve(strict=True)
    require(path.is_relative_to(directory) and sha(path) == point["file_sha256"], "frame file binding")
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), np.uint8)
    require(
        hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"]
        and point["state"] == point["snapshot"]["state"]
        and point["goal"] == expected["goal"]
        and point["goal_rgb_sha256"] == expected["goal_rgb_sha256"],
        "frame content binding",
    )
    if returned is not None:
        require(point["state"] == returned["state"], "frame/transition state")


def verify(prepared_path: Path, prepared_sha: str, directory: Path, report_sha: str) -> dict:
    prepared_path = prepared_path.resolve(strict=True)
    directory = directory.resolve(strict=True)
    require(sha(prepared_path) == prepared_sha, "external prepared SHA")
    prepared = json.loads(prepared_path.read_bytes())
    require(prepared == source.build(), "prepared source closure")
    report_path = directory / "report.json"
    require(sha(report_path) == report_sha, "external report SHA")
    report = json.loads(report_path.read_bytes())
    require(
        report["schema"] == source.NAME + ".report.v1"
        and report["status"] == "recorded_development"
        and report["prepare_sha256"] == prepared_sha
        and not report["errors"]
        and not report["pending"]
        and not report["secondary_errors"]
        and len(report["rows"]) == 6
        and report["new_neural_calls_during_physical"] == 0
        and report["development_only"] is True
        and report["independent_confirmation"] is False
        and report["fair_total_cost"] is False
        and report["P1_admission"] is False,
        "report contract",
    )
    binding = json.loads((directory / "execution_binding.json").read_bytes())
    require(
        binding == {"prepared": str(prepared_path), "prepare_sha256": prepared_sha},
        "execution binding",
    )
    journal_path = directory / "journal.jsonl"
    require(sha(journal_path) == report["journal_sha256"], "journal SHA")
    events = engine.journal_reader.journal(journal_path.read_bytes(), report["journal_chain_tail"])
    require(
        events[0]["kind"] == "run_started"
        and events[0]["prepare_sha256"] == prepared_sha
        and events[-1]["kind"] == "run_finished"
        and events[-1]["status"] == report["status"],
        "journal endpoints",
    )
    counted = Counter()
    for event in events:
        if event["kind"] in ("operation_attempted", "operation_completed"):
            counted[(event["resource"], event["kind"].removeprefix("operation_"))] += event["units"]
    require(
        all(
            counted[(resource, kind)] == count
            for resource, values in report["counts"].items()
            for kind, count in values.items()
        ),
        "resource accounting",
    )

    root_specs = {root["slot"]: root for root in prepared["roots"]}
    expected_order = [
        (slot, role)
        for slot in (0, 1)
        for role in prepared["roles"]
    ]
    rows = []
    for row, identity in zip(report["rows"], expected_order, strict=True):
        slot, role = identity
        root = root_specs[slot]
        episode = f"root{slot}_{role}"
        require(
            row["episode_id"] == episode
            and row["source_slot"] == slot
            and row["role"] == role
            and row["decision"] == root["decisions"][role]
            and row["status"] in ("completed15", "native_stopped", "censored_out_of_domain"),
            "row identity/status",
        )
        require(json.loads((directory / f"{episode}_row.json").read_bytes()) == row, "row shard")
        expected_actions = np.asarray(root["candidates"][role], np.float32).reshape(15, 2)
        actual_actions = np.asarray(row["executed_raw"], np.float32)
        returned = row["control_returned"]
        require(
            actual_actions.shape == (returned, 2)
            and np.array_equal(actual_actions, expected_actions[:returned])
            and row["planned_actions_sha256"]
            == hashlib.sha256(expected_actions.astype("<f4").tobytes()).hexdigest()
            and len(row["transitions"]) == returned
            and len(row["observations"]) == returned + 1,
            "planned/executed action identity",
        )
        dispatches = [
            event for event in events
            if event["kind"] == "branch_dispatch"
            and event["slot"] == slot and event["role"] == role
        ]
        returns = [
            event for event in events
            if event["kind"] == "branch_returned"
            and event["slot"] == slot and event["role"] == role
        ]
        require(len(dispatches) == len(returns) == returned, "dispatch/return count")
        for step, (dispatch, returned_event, transition) in enumerate(
            zip(dispatches, returns, row["transitions"], strict=True), 1
        ):
            expected_command = expected_actions[step - 1].tolist()
            require(
                dispatch["step"] == returned_event["step"] == transition["step"] == step
                and dispatch["command"] == returned_event["command"] == transition["command"]
                == expected_command
                and dispatch["index"] < returned_event["index"],
                "dispatch/return/action sequence",
            )
        observations = row["observations"]
        require(
            [point["step"] for point in observations] == list(range(returned + 1))
            and all(observations[0][key] == root["expected"][key]
                    for key in ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")),
            "root/observation sequence",
        )
        for index, point in enumerate(observations):
            _verify_frame(directory, point, root["expected"], None if index == 0 else returns[index - 1])
        if returned >= prepared["feedback_boundary"]:
            decision = root["decisions"][role]
            if decision["kind"] == "actual_cp5_rgb_feedback_cem":
                require(
                    observations[prepared["feedback_boundary"]]["pixel_sha256"]
                    == decision["cp5_pixel_sha256"],
                    "actual cp5 feedback identity",
                )
                require(row["prefix_metrics"] == row["transitions"][4]["metrics"], "cp5 metrics")
        else:
            require(
                slot == 1 and role == "same_bank_greedy"
                and root["decisions"][role] == {
                    "kind": "original_early_success", "stop_step_source": 4
                }
                and returned == 4 and row.get("prefix_metrics") is None,
                "pre-feedback early stop",
            )
        last = returns[-1]
        require(
            row["final_metrics"] == row["transitions"][-1]["metrics"]
            and (
                row["status"] != "completed15"
                or (returned == 15 and not last["terminated"] and not last["truncated"])
            )
            and (
                row["status"] != "native_stopped"
                or (
                    row["stop_step"] == returned
                    and (last["terminated"] or last["truncated"])
                    and row["final_metrics"]["native_success"] is True
                )
            )
            and (
                row["status"] != "censored_out_of_domain"
                or bool(row.get("censor"))
            ),
            "termination/no-padding evidence",
        )
        rows.append({
            "slot": slot,
            "role": role,
            "status": row["status"],
            "returned_steps": returned,
            "native_success": bool(row["final_metrics"]["native_success"]),
            "object_position_error": row["final_metrics"]["object_position_error"],
            "wrapped_object_angle_error": row["final_metrics"]["wrapped_object_angle_error"],
        })
    require(
        all(sha(Path(path)) == digest for path, digest in prepared["bindings"].items()),
        "post-run source closure",
    )
    by_root = {
        str(slot): {
            row["role"]: {
                "status": row["status"],
                "returned_steps": row["returned_steps"],
                "native_success": row["native_success"],
                "object_position_error": row["object_position_error"],
                "wrapped_object_angle_error": row["wrapped_object_angle_error"],
            }
            for row in rows if row["slot"] == slot
        }
        for slot in (0, 1)
    }
    return {
        "schema": NAME + ".result.v1",
        "status": "saved_feedback_physical_verified_development_only",
        "prepared": str(prepared_path),
        "prepare_sha256": prepared_sha,
        "report": str(report_path),
        "report_sha256": report_sha,
        "journal_sha256": report["journal_sha256"],
        "verified_rows": len(rows),
        "rows": rows,
        "by_root": by_root,
        "interpretation": {
            "actual_cp5_rgb_feedback_verified": True,
            "planned_executed_actions_exact": True,
            "root1_selected_repaired_to_success": True,
            "root1_selected_beats_zero_by_steps": 1,
            "selected_beats_same_bank_greedy_on_any_root": False,
            "known_development_roots": 2,
            "fair_total_cost": False,
            "independent_confirmation": False,
            "P1_admission": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--prepare-sha256", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(
        args.output.is_absolute()
        and args.output.parent == ROOT / "results"
        and not os.path.lexists(args.output),
        "exclusive output",
    )
    result = verify(args.prepared, args.prepare_sha256, args.directory, args.report_sha256)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with args.output.open("xb") as stream:
        require(stream.write(raw) == len(raw), "complete output")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha(args.output),
        "verified_rows": result["verified_rows"],
        "P1_admission": result["interpretation"]["P1_admission"],
    }))


if __name__ == "__main__":
    main()
