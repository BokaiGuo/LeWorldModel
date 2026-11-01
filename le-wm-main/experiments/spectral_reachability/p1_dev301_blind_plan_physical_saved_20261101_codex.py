"""Strict saved-only verifier for the dev301 three-arm physical execution."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_dev301_blind_plan_physical_20261101_codex as source

NAME = "p1_dev301_blind_plan_physical_saved_20261101_codex"
ROOT = source.ROOT
require, sha = source.require, source.sha


def journal(path, tail):
    raw = Path(path).read_bytes()
    require(raw.endswith(b"\n"), "complete journal")
    rows = [json.loads(line) for line in raw.splitlines()]
    previous = None
    for index, row in enumerate(rows):
        body = {key: value for key, value in row.items() if key != "sha256"}
        require(
            row["index"] == index
            and row["previous"] == previous
            and row["sha256"] == hashlib.sha256(source.canonical(body)).hexdigest(),
            "journal hash chain",
        )
        previous = row["sha256"]
    require(previous == tail, "journal tail")
    return rows


def frame(directory, point, returned_state=None):
    path = (directory / point["file"]).resolve(strict=True)
    require(path.is_relative_to(directory) and sha(path) == point["file_sha256"], "frame file")
    with Image.open(path) as handle:
        pixels = np.asarray(handle.convert("RGB"), np.uint8)
    require(
        source.pixel_sha(pixels) == point["pixel_sha256"]
        and point["state"] == point["snapshot"]["state"],
        "frame pixels/state",
    )
    if returned_state is not None:
        require(point["state"] == returned_state, "frame/transition state")


def verify(prepared, prepared_sha, directory, report_sha):
    prepared = Path(prepared).resolve(strict=True)
    directory = Path(directory).resolve(strict=True)
    require(
        sha(prepared) == prepared_sha and source.read(prepared) == source.build(),
        "prepared source closure",
    )
    contract = source.read(prepared)
    report_path = directory / "report.json"
    require(sha(report_path) == report_sha, "external report SHA")
    report = source.read(report_path)
    require(
        report["schema"] == source.NAME + ".report.v1"
        and report["status"] == "recorded_development"
        and report["prepare_sha256"] == prepared_sha
        and report["errors"] == []
        and report["new_neural_calls"] == 0
        and report["development_only"] is True
        and report["equal_budget_claim"] is False
        and report["independent_confirmation"] is False
        and report["P1_admission"] is False
        and len(report["rows"]) == 3,
        "report contract",
    )
    require(
        source.read(directory / "execution_binding.json")
        == {"prepared": str(prepared), "prepare_sha256": prepared_sha},
        "execution binding",
    )
    events = journal(directory / "journal.jsonl", report["journal_tail"])
    require(
        sha(directory / "journal.jsonl") == report["journal_sha256"]
        and events[0]["kind"] == "run_started"
        and events[-1]["kind"] == "run_finished"
        and events[-1]["status"] == report["status"],
        "journal endpoints",
    )
    grouped = defaultdict(list)
    for event in events:
        if "arm" in event:
            grouped[event["arm"]].append(event)
    source_history = source.read(source.source_directory() / "source_receipt.json")["history"]
    source_frames = {row["step"]: row for row in source_history["frames"]}
    arm_specs = {row["arm"]: row for row in contract["arms"]}
    rows = []
    for row, arm in zip(report["rows"], source.ARMS, strict=True):
        require(
            row["arm"] == arm
            and row["selection_sha256"] == arm_specs[arm]["selection_sha256"]
            and row["status"] in ("horizon_completed", "native_success", "native_stopped")
            and row["error"] is None
            and row["warmup_returned"] == 10,
            "row identity/status",
        )
        require(source.read(directory / f"{arm}.row.json") == row, "row shard")
        expected = source.action_array(arm_specs[arm]["actions"])
        returned = row["plan_returned"]
        actual = np.asarray([item["action"] for item in row["transitions"]], np.float32)
        require(
            actual.shape == (returned, 2)
            and np.array_equal(actual, expected[:returned])
            and row["planned_actions_sha256"] == hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest(),
            "planned/executed action identity",
        )
        arm_events = grouped[arm]
        starts = [event for event in arm_events if event["kind"] == "branch_started"]
        closes = [event for event in arm_events if event["kind"] == "branch_closed"]
        dispatches = [event for event in arm_events if event["kind"] == "action_dispatch"]
        returns = [event for event in arm_events if event["kind"] == "action_return"]
        observations = [event for event in arm_events if event["kind"] == "observation"]
        require(
            len(starts) == len(closes) == 1
            and starts[0]["seed"] == source.SEED
            and closes[0]["status"] == row["status"]
            and len(dispatches) == len(returns) == 10 + returned,
            "branch lifecycle/count",
        )
        for index, (dispatch, result) in enumerate(zip(dispatches, returns, strict=True)):
            require(
                dispatch["index"] < result["index"]
                and dispatch["phase"] == result["phase"]
                and dispatch["step"] == result["step"]
                and dispatch["action"] == result["action"],
                "dispatch/return sequence",
            )
            if index < 10:
                require(dispatch["phase"] == "warmup" and dispatch["action"] == [0.0, 0.0], "warmup action")
            else:
                step = index - 9
                require(
                    dispatch["phase"] == "plan"
                    and dispatch["step"] == step
                    and dispatch["action"] == expected[step - 1].tolist()
                    and result["state"] == row["transitions"][step - 1]["state"],
                    "plan action/return",
                )
        warm_observations = [item for item in row["observations"] if item["phase"] == "warmup"]
        plan_observations = [item for item in row["observations"] if item["phase"] == "plan"]
        require(
            [item["step"] for item in warm_observations] == [0, 5, 10]
            and [item["step"] for item in plan_observations] == list(range(1, returned + 1))
            and len(observations) == len(row["observations"]),
            "observation roster",
        )
        for point in warm_observations:
            reference = source_frames[point["step"]]
            require(
                point["state"] == reference["state"]
                and point["snapshot"] == reference["snapshot"]
                and point["pixel_sha256"] == reference["pixel_sha256"],
                "exact sealed warmup checkpoint",
            )
            frame(directory, point)
        for point, transition in zip(plan_observations, row["transitions"], strict=True):
            frame(directory, point, transition["state"])
        last = row["transitions"][-1]
        require(
            row["final_metrics"] == last["metrics"]
            and (
                row["status"] != "horizon_completed"
                or (returned == 10 and not last["terminated"] and not last["truncated"])
            )
            and (
                row["status"] != "native_success"
                or (last["terminated"] and last["metrics"]["native_success"] and returned < 10)
            ),
            "termination/no-padding",
        )
        rows.append(
            {
                "arm": arm,
                "status": row["status"],
                "returned_steps": returned,
                "native_success": row["final_metrics"]["native_success"],
                "model_terminal_cost": contract["planner"]["model_terminal_cost"].get(arm),
                "native_distance": row["final_metrics"]["native_distance"],
                "pusher_position_error": row["final_metrics"]["pusher_position_error"],
                "object_position_error": row["final_metrics"]["object_position_error"],
                "wrapped_object_angle_error": row["final_metrics"]["wrapped_object_angle_error"],
                "contact_steps": [
                    item["step"] for item in row["transitions"] if item["n_contacts"] > 0
                ],
                "prefix_step5_metrics": row["transitions"][4]["metrics"],
            }
        )
    require(
        all(sha(path) == digest for path, digest in contract["bindings"].items()),
        "post-consume source closure",
    )
    by_arm = {row["arm"]: row for row in rows}
    gn, d2p, zero = (by_arm[arm] for arm in source.ARMS)
    return {
        "schema": NAME + ".result.v1",
        "status": "saved_dev301_physical_verified_development_only",
        "prepared": str(prepared),
        "prepare_sha256": prepared_sha,
        "report": str(report_path),
        "report_sha256": report_sha,
        "journal_sha256": report["journal_sha256"],
        "rows": rows,
        "interpretation": {
            "planned_executed_actions_exact": True,
            "exact_seed_zero_warmup_rgb_state_replay": True,
            "all_three_native_success": False,
            "d2p_object_position_better_than_gn": d2p["object_position_error"] < gn["object_position_error"],
            "d2p_native_distance_better_than_gn": d2p["native_distance"] < gn["native_distance"],
            "d2p_angle_better_than_gn": d2p["wrapped_object_angle_error"] < gn["wrapped_object_angle_error"],
            "model_cost_prefers_gn": gn["model_terminal_cost"] < d2p["model_terminal_cost"],
            "d2p_prefix_has_immediate_object_progress": d2p["prefix_step5_metrics"]["object_position_error"]
            < zero["object_position_error"],
            "non_greedy_reconfiguration_mechanism_established": False,
            "equal_budget_claim": False,
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
    require(output.parent == ROOT / "results" and not os.path.lexists(output), "exclusive result")
    value = verify(args.prepared, args.prepare_sha256, args.directory, args.report_sha256)
    source.write_new(output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
