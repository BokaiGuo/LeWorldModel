"""Check root1 saved node traces, selections and executed action packets."""
import json
from pathlib import Path

import numpy as np

from . import codex_four_arm_common_rgb_root1_20261030_main as run

DIRECTORY = run.ROOT / "results/codex_four_arm_common_rgb_root1_20261030_main_RUN01"
REPORT_SHA = "11857215e6edbc51a0eaf8500c891168011762454a58820b49d773dfc4a73f61"


def check():
    path = DIRECTORY / "report.json"
    run.require(run.sha(path) == REPORT_SHA, "report SHA")
    report = json.loads(path.read_bytes())
    run.require(report["status"] == "recorded_known_root_not_admitted" and not report["errors"],
                "report status")
    run.require(run.sha(DIRECTORY / "journal.jsonl") == report["journal_sha256"], "journal SHA")
    total = 0
    for row in report["rows"]:
        run.require(row["source_slot"] == 1 and row["initial_exact"] and row["control_returned"] == 20,
                    "root identity and full horizon")
        actions = np.asarray(row["executed_raw"], dtype=np.float32)
        run.require(actions.shape == (20, 2) and np.isfinite(actions).all(), "executed actions")
        nodes = report["nodes"][row["episode_id"]]
        for i, node in enumerate(nodes):
            boundary = i * 5
            run.require(node["status"] == "sealed" and node["boundary"] == boundary, "node roster")
            receipt = node["receipt"]
            trace = Path(receipt["trace"])
            run.require(trace.parent == DIRECTORY / (row["episode_id"] + f"_b{boundary:02}_common")
                        and run.sha(trace) == receipt["trace_sha256"], "trace path and hash")
            lines = [json.loads(line) for line in trace.read_bytes().splitlines()]
            payloads = {e["kind"]: e["payload"] for e in lines if e["kind"] in
                        ("optimized_candidates", "final_selection")}
            run.require(set(payloads) == {"optimized_candidates", "final_selection"}, "events")
            choices = np.asarray(payloads["optimized_candidates"]["raw"], dtype=np.float32)
            index = payloads["final_selection"]["index"]
            commands = np.asarray(receipt["commands"], dtype=np.float32)
            run.require(choices.shape == (8, 3, 5, 2) and type(index) is int and 0 <= index < 8
                        and index == receipt["selected_index"] and commands.shape == (5, 2)
                        and choices[index, 0].tobytes() == commands.tobytes()
                        and commands.tobytes() == actions[boundary:boundary+5].tobytes(),
                        "trace selection and physical packet")
            run.require(payloads["final_selection"]["scores"] == receipt["scores"], "scores")
            total += 1
    run.require(len(report["rows"]) == 4 and total == 16 and run.sha(path) == REPORT_SHA,
                "complete fixed roster")
    return {"status": "passed", "nodes": total, "report_sha256": REPORT_SHA,
            "scope": "trace SHA and selected action to recorded physical packet only"}


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True))
