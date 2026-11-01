"""Independent saved-only verification; no new physical or neural calls."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_diag32_fixed_physical_20261101_exclusive as physical
from . import p1_moving_live_feedback_analysis_20261027_v2 as analysis

ROOT = physical.ROOT
PREP = ROOT / "results/codex_p1_diag32_fixed_physical_20261101_exclusive_PREP_CODEXPHYS01.json"
DIRECTORY = ROOT / "results/codex_p1_diag32_fixed_physical_20261101_exclusive_run_CODEXPHYS01"
REPORT_SHA = "01095a298ed35e8ccb6b68dbaa57f63e11f9fb6015e2e39cc36ee1325bcca330"
PREP_SHA = "9a61ec04a71ea3e32d5c7693a30c77fdb8e052b96a02ca56a44bd4ecf1049341"
require, sha = physical.require, physical.sha


def verify():
    require(sha(PREP) == PREP_SHA and sha(DIRECTORY / "report.json") == REPORT_SHA, "pinned artifacts")
    contract = json.loads(PREP.read_bytes())
    require(contract == physical.build(), "contract/source drift")
    report = json.loads((DIRECTORY / "report.json").read_bytes())
    row = report["row"]
    require(report["prepare_sha256"] == PREP_SHA and not report["errors"] and report["status"] == "recorded_development"
            and json.loads((DIRECTORY / "row.json").read_bytes()) == row, "report/row")
    journal_path = DIRECTORY / "journal.jsonl"
    require(sha(journal_path) == report["journal_sha256"], "journal hash")
    events = analysis.journal(journal_path.read_bytes(), report["journal_chain_tail"])
    require(events[0]["kind"] == "run_started" and events[-1]["kind"] == "run_finished"
            and events[-1]["status"] == report["status"] and events[-1]["counts"] == report["counts"], "journal ends")
    pending, counts = {}, {k: {"attempted": 0, "completed": 0} for k in report["counts"]}
    for event in events:
        if event["kind"] == "operation_attempted":
            ident = event["operation_id"]
            require(ident not in pending, "duplicate operation")
            pending[ident] = event
            counts[event["resource"]]["attempted"] += event["units"]
        elif event["kind"] == "operation_completed":
            ident = event["operation_id"]
            require(ident in pending and all(event[k] == pending[ident][k] for k in
                    ("resource", "units", "region", "episode_id")), "operation pairing")
            counts[event["resource"]]["completed"] += event["units"]
            del pending[ident]
    require(not pending and counts == report["counts"] and not report["pending"], "meter")
    dispatch = [e for e in events if e["kind"] == "branch_dispatch"]
    returned = [e for e in events if e["kind"] == "branch_returned"]
    transitions = row["transitions"]
    require(len(dispatch) == len(returned) == len(transitions) == row["control_returned"] <= 15, "step roster")
    commands = np.asarray(contract["commands"], dtype="<f4")
    actual = np.asarray(row["executed_raw"], dtype="<f4")
    require(np.array_equal(actual, commands[:len(actual)]) and
            hashlib.sha256(commands.tobytes()).hexdigest() == contract["commands_float32_le_sha256"], "action bytes")
    goal = np.asarray(contract["root"]["goal"])
    for step, (sent, ret, transition) in enumerate(zip(dispatch, returned, transitions, strict=True), 1):
        require(sent["step"] == ret["step"] == transition["step"] == step
                and sent["command"] == ret["command"] == transition["command"] == commands[step-1].tolist(),
                "dispatch/return action")
        for key in ("state", "n_contacts", "reward", "terminated", "truncated", "metrics"):
            require(ret[key] == transition[key], "returned content")
        metrics = analysis.native_metrics(transition["state"], goal)
        for key in ("native_success", "native_distance", "native_angle_term", "negative_native_angle",
                    "wrapped_allxy_success", "wrapped_object_angle_error", "object_position_error"):
            require(metrics[key] == transition["metrics"][key], "metric mismatch: " + key)
        if step < len(transitions):
            require(not transition["terminated"] and not transition["truncated"], "steps after stop")
    for index, receipt in enumerate(row["observations"]):
        require(receipt["phase"] == "plan" and receipt["step"] == index, "observation order")
        image_path = Path(receipt["path"])
        require(image_path.parent == DIRECTORY and not image_path.is_symlink()
                and sha(image_path) == receipt["file_sha256"], "PNG hash")
        with Image.open(image_path) as image:
            pixels = np.asarray(image)
        require(pixels.shape == (224, 224, 3) and physical.engine.old.array_sha(pixels) == receipt["pixel_sha256"],
                "pixel hash")
    root = row["observations"][0]
    require(all(root[k] == contract["root"][k] for k in contract["root"]) and row["initial_exact"], "root")
    require(row["status"] == "native_stopped" and len(transitions) == 15 and transitions[-1]["terminated"]
            and not transitions[-1]["truncated"] and len(row["observations"]) == 16, "terminal H3")
    require(row["final_metrics"] == transitions[-1]["metrics"], "final metrics")
    return {"status": "verified_saved_physical_development", "report_sha256": REPORT_SHA,
            "contract_sha256": PREP_SHA, "action_sha256": contract["commands_float32_le_sha256"],
            "planned_actions": 15, "returned_actions": len(transitions), "all_actions_returned": True,
            "native_terminated_at": 15, "frames": len(row["observations"]), "counts": counts,
            "note": "source row full_horizon false and endpoint null because native stop handling precedes loop completion; all 15 actions and last observation independently verified",
            "new_neural_calls": 0, "new_physical_calls": 0, "P1_admission": False}


if __name__ == "__main__":
    result = verify()
    output = ROOT / "results/codex_p1_diag32_fixed_physical_verify_20261101_exclusive_CODEXPHYS01.json"
    print(json.dumps({"verification": str(output), "sha256": physical.blind.write_new(output, result)}, sort_keys=True))
