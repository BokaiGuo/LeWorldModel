"""Retrospective audit: no planner access to evaluation-only physical state."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/gbk/.stable_worldmodel/datasets/tworoom.h5")
REFERENCE = ROOT / "results/pc_standard_tworoom_reference_replay_20261105_codex/report.json"
INPUT = ROOT / "results/pc_tworoom_reference_vs_hold_input_20261107_codex.npz"
SOURCES = {
    "cem": ROOT / "results/pc_tworoom_rgb_deadline_cem_20261107_codex_run.json",
    "prefix": ROOT / "results/pc_tworoom_rgb_timed_prefix_feedback_20261107_codex.json",
}
OUTPUT = ROOT / "results/pc_tworoom_success_contract_audit_20261107_codex.json"
ENV_SOURCE = ROOT / ".venv/lib/python3.12/site-packages/stable_worldmodel/envs/two_room/env.py"
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_SUCCESS_CONTRACT_AUDIT_PROTOCOL_20261107_CODEX.md"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_trace(arm):
    trace = arm["trace"]
    if not 1 <= len(trace) <= 25:
        raise ValueError("invalid trace length")
    for index, step in enumerate(trace):
        action = np.asarray(step["action"], dtype=float)
        distance = float(step["distance"])
        if action.shape != (2,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
            raise ValueError("invalid action")
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("invalid distance")
        if bool(step["terminated"]) != (distance < 16.0) or step["truncated"]:
            raise ValueError("termination or truncation mismatch")
        if step["terminated"] and index != len(trace) - 1:
            raise ValueError("execution continued after termination")
    success = any(x["terminated"] for x in trace)
    if success != arm["ever_success"] or arm["truncated"]:
        raise ValueError("summary mismatch")
    if float(arm["terminal_distance"]) != float(trace[-1]["distance"]):
        raise ValueError("terminal distance mismatch")
    return success


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reference = json.loads(REFERENCE.read_text())["records"]
    reports = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    expected_arms = {"cem": {"cem13", "cem300"},
                     "prefix": {"parent", "lookahead", "vector", "shuffled"}}
    with np.load(INPUT, allow_pickle=False) as sealed, h5py.File(DATA, "r") as data:
        sealed_rows = sealed["rows"]
        if len(reference) != 50 or len(sealed_rows) != 50:
            raise ValueError("reference panel size mismatch")
        root_details = []
        max_initial_error = max_goal_error = max_ref_error = 0.0
        for root in range(10, 50):
            ref = reference[root]
            row = int(sealed_rows[root])
            if row != ref["row"] or not ref["completed"]:
                raise ValueError("reference row mismatch")
            initial = np.asarray(data["proprio"][row], dtype=float)
            goal = np.asarray(data["proprio"][row + 25], dtype=float)
            max_initial_error = max(max_initial_error, float(np.max(np.abs(initial - ref["initial_position"]))))
            max_goal_error = max(max_goal_error, float(np.max(np.abs(goal - ref["goal_position"]))))
            for step in ref["steps"]:
                reference_position = np.asarray(step["reference_position"], dtype=float)
                actual_position = np.asarray(step["actual_position"], dtype=float)
                dataset_position = np.asarray(data["proprio"][row + step["step"]], dtype=float)
                max_ref_error = max(max_ref_error, float(np.max(np.abs(reference_position - dataset_position))),
                                    float(np.max(np.abs(actual_position - dataset_position))))
            distance = float(np.linalg.norm(initial - goal))
            root_details.append({"root": root, "row": row, "initial_distance": distance,
                                 "initially_successful": distance < 16.0,
                                 "reference_ever_success": ref["ever_success"]})
        if max(max_initial_error, max_goal_error, max_ref_error) > 1e-5:
            raise ValueError("dataset/reference state mismatch")
        results = {}
        for name, report in reports.items():
            rows = report["rows"]
            if [x["root"] for x in rows] != list(range(10, 50)):
                raise ValueError("feedback root panel mismatch")
            for item, detail in zip(rows, root_details, strict=True):
                if item["row"] != detail["row"] or set(item["arms"]) != expected_arms[name]:
                    raise ValueError("feedback row or arm mismatch")
                for arm_name, arm in item["arms"].items():
                    success = audit_trace(arm)
                    key = f"{name}/{arm_name}"
                    aggregate = results.setdefault(key, {"steps": 0, "success": 0,
                                                          "initially_successful": 0,
                                                          "initially_unsuccessful": 0,
                                                          "success_initially_unsuccessful": 0})
                    aggregate["steps"] += len(arm["trace"])
                    aggregate["success"] += int(success)
                    initial_success = detail["initially_successful"]
                    aggregate["initially_successful"] += int(initial_success)
                    aggregate["initially_unsuccessful"] += int(not initial_success)
                    aggregate["success_initially_unsuccessful"] += int(success and not initial_success)
        output = {"scope": "retrospective exposed-development audit; evaluation-only state",
                  "inputs_sha256": {str(p): digest(p) for p in [REFERENCE, INPUT, *SOURCES.values(),
                                                                ENV_SOURCE, PROTOCOL, Path(__file__)]},
                  "dataset": {"path": str(DATA), "size_bytes": DATA.stat().st_size,
                              "mtime_ns": DATA.stat().st_mtime_ns,
                              "whole_file_hash_computed": False},
                  "max_state_difference": {"initial": max_initial_error, "goal": max_goal_error,
                                           "reference_steps": max_ref_error},
                  "roots": root_details, "arms": results}
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({"arms": results, "max_state_difference": output["max_state_difference"]}, indent=2))


if __name__ == "__main__":
    main()
