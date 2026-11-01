"""Saved-only verifier for the dev303 two-arm RGB feedback diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as run
from . import p1_dev301_blind_plan_physical_saved_20261101_codex as saved


def verify(prepared, prepared_sha, directory, report_sha):
    prepared = Path(prepared).resolve(strict=True)
    directory = Path(directory).resolve(strict=True)
    run.require(run.base.sha(prepared) == prepared_sha
                and run.base.read(prepared) == run.build(), "prepared source closure")
    report_path = directory / "report.json"
    run.require(run.base.sha(report_path) == report_sha, "report SHA")
    report = run.base.read(report_path)
    run.require(report["schema"] == run.NAME + ".report.v1"
                and report["status"] == "recorded" and report["errors"] == []
                and report["prepared_sha256"] == prepared_sha
                and report["P1_admission"] is False
                and report["independent_confirmation"] is False, "report boundary")
    run.require(run.base.read(directory / "execution_binding.json") == {
        "prepared": str(prepared), "prepare_sha256": prepared_sha
    }, "execution binding")
    journal_path = directory / "journal.jsonl"
    run.require(run.base.sha(journal_path) == report["journal_sha256"], "journal SHA")
    events = saved.journal(journal_path, report["journal_tail"])
    run.require(events[0]["kind"] == "run_started"
                and events[-1]["kind"] == "run_finished"
                and events[-1]["errors"] == [], "journal endpoints")
    _arrays, receipt, _reset = run.source_data()
    reference = {item["step"]: item for item in receipt["history"]["frames"]}
    rows = report["rows"]
    run.require([row["arm"] for row in rows] == list(run.ARMS), "arm roster")
    results = []
    for row, arm in zip(rows, run.ARMS, strict=True):
        run.require(run.base.read(directory / (arm + ".row.json")) == row
                    and row["warmup_returned"] == 10
                    and row["status"] == "fixed_horizon_completed"
                    and row["control_returned"] == run.HORIZON
                    and len(row["transitions"]) == run.HORIZON, "complete row")
        points = row["observations"]
        warm = [p for p in points if p["phase"] == "warmup"]
        actual = [p for p in points if p["phase"] == "control"]
        run.require([p["step"] for p in warm] == [0, 5, 10]
                    and [p["step"] for p in actual] == list(range(1, run.HORIZON + 1)),
                    "frame roster")
        for point in warm:
            ref = reference[point["step"]]
            run.require(point["state"] == ref["state"]
                        and point["snapshot"] == ref["snapshot"]
                        and point["pixel_sha256"] == ref["pixel_sha256"], "warmup frame")
            saved.frame(directory, point)
        for point, transition in zip(actual, row["transitions"], strict=True):
            saved.frame(directory, point, transition["state"])
        scoped = [event for event in events if event.get("arm") == arm]
        dispatches = [e for e in scoped if e["kind"] == "action_dispatch"]
        returns = [e for e in scoped if e["kind"] == "action_return"]
        decisions = [e for e in scoped if e["kind"] == "controller_decision"]
        run.require(len(dispatches) == len(returns) == 10 + run.HORIZON,
                    "dispatch/return count")
        controller = run.control.RGBChunkApproach(initial_previous_action_is_zero=True) if arm == "rgb_chunk" else None
        prior_sha = None
        for i, (dispatch, returned) in enumerate(zip(dispatches, returns, strict=True)):
            run.require(dispatch["index"] < returned["index"]
                        and dispatch["phase"] == returned["phase"]
                        and dispatch["step"] == returned["step"]
                        and dispatch["action"] == returned["action"], "action receipt")
            if i < 10:
                run.require(dispatch["phase"] == "warmup" and dispatch["action"] == [0.0, 0.0],
                            "zero warmup")
                continue
            step = i - 9
            transition = row["transitions"][step - 1]
            action = np.asarray(dispatch["action"], dtype="<f4")
            action_sha = hashlib.sha256(action.tobytes()).hexdigest()
            run.require(dispatch["phase"] == "control" and dispatch["step"] == step
                        and transition["step"] == step and transition["action"] == dispatch["action"]
                        and transition["action_sha256"] == action_sha
                        and returned["state"] == transition["state"], "control receipt")
            if controller is None:
                run.require(not action.any(), "zero arm action")
            else:
                previous_frame = warm[-1] if step == 1 else actual[step - 2]
                rgb = run.base.image(directory / previous_frame["file"])
                proposal = controller.observe(rgb, frame_id=step - 1, executed_action_sha256=prior_sha)
                run.require(len(decisions) == run.HORIZON
                            and decisions[step - 1]["index"] < dispatch["index"]
                            and decisions[step - 1]["proposal"] == proposal
                            and proposal["action_sha256"] == action_sha, "RGB decision replay")
            prior_sha = action_sha
        results.append({"arm": arm, "control_actions": row["control_returned"],
                        "status": row["status"]})
    return {"status": "saved_feedback_chain_verified", "arms": results,
            "physical_result_scope": "one development root",
            "P1_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("prepared", "prepared-sha", "directory", "report-sha"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.prepared, args.prepared_sha, args.directory, args.report_sha),
                     sort_keys=True))


if __name__ == "__main__":
    main()
