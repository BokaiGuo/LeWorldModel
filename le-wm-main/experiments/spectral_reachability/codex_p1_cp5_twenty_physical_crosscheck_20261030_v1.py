"""Saved transition/reward crosscheck for fixed twenty-step physical results."""
import argparse
import json
from pathlib import Path

import numpy as np
from . import codex_p1_cp5_twenty_physical_saved_20261030_v1 as saved
from . import p1_backtrack_physical_consume_20261030 as events_reader


def check(directory, pair_receipt, pair_sha, output):
    run, need = saved.run, saved.need
    pairs = run.read(pair_receipt, pair_sha)
    need(pairs["status"] == "24_pairs_saved_verified" and len(pairs["rows"]) == 24, "all pairs")
    directory = Path(directory)
    report = run.read(directory/"run_report.json", pairs["twenty_run_sha256"])
    anchors = run.h10.native.read(run.h10.INPUT, cap=64*1024**2)["anchors"]
    rows = []
    for i, (pair, anchor, shard) in enumerate(zip(pairs["rows"], anchors, report["shards"], strict=True)):
        need(i == pair["index"] == anchor["index"] == shard["index"], "row identity")
        folder = directory/f"row_{i:02d}"
        row_report = run.read(folder/"report.json", shard["report_sha256"])
        need(run.sha(folder/"journal.jsonl") == row_report["journal_sha256"], "journal SHA")
        events = events_reader.events(folder/"journal.jsonl", row_report["journal_chain_tail"])
        goal = [e for e in events if e["kind"] == "goal_attached"]
        need(len(goal) == 1 and goal[0]["goal"] == anchor["goal"], "goal identity")
        transitions = [e for e in events if e["kind"] == "returned_transition"]
        for t in transitions:
            matching = [e for e in events if e["kind"] == "operation_returned" and
                        e["receipt"].get("resource") == "public_step" and
                        e["receipt"].get("phase") == t["phase"] and e["receipt"].get("step") == t["step"]]
            need(len(matching) == 1 and matching[0]["index"] < t["index"]
                 and matching[0]["receipt"]["raw_bits"] == t["requested_bits"] == t["actual_bits"],
                 "public action/transition")
            if t["phase"] != "warmup":
                metric = run.h10.base.previous.metric(anchor["goal"], t["state"])
                need(t["goal_state"] == anchor["goal"] and t["reward"] == -metric["native_distance"]
                     and t["terminated"] == metric["native_success"], "native goal/reward/success")
            if t["phase"] != "probe":
                need(not t["terminated"] and not t["truncated"], "prefix stop")
        probe = [t for t in transitions if t["phase"] == "probe"]
        for response, trans in zip(row_report["row"]["responses"], probe, strict=False):
            delta = np.asarray(trans["state"]) - np.asarray(row_report["row"]["start_state"])
            need(response["delta_native7"] == delta.tolist() and
                 response["delta_angle_wrapped"] == float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))),
                 "response delta")
        need(len(probe) == pair["twenty_step"]["accepted_suffix_steps"] +
             int(pair["twenty_step"]["censored"]), "returned probe steps")
        rows.append({"index": i, "status": pair["twenty_step"]["status"], "returned_steps": len(probe)})
    output = Path(output)
    need(output.is_absolute() and output.parent == run.ROOT/"results" and
         output.name.startswith("codex_p1_cp5_twenty_physical_crosscheck_20261030_v1_"), "exclusive output")
    result = {"schema": "codex_p1_cp5_twenty_physical_crosscheck_20261030_v1",
              "status": "checked", "pair_receipt_sha256": pair_sha, "rows": rows,
              "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False}
    return run.h10._write_new(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--pair-receipt", required=True)
    parser.add_argument("--pair-sha", required=True)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    print(json.dumps({"sha256": check(a.directory, a.pair_receipt, a.pair_sha, a.output)}))
