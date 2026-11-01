"""Additional independent saved transition/goal/receipt consistency checks."""
import argparse
import json
from pathlib import Path
import numpy as np

from . import codex_p1_cp5_physical_saved_20261030_v1 as saved


def check(directory, receipt, receipt_sha, output):
    run, need = saved.run, saved.need
    need(run.sha(receipt) == receipt_sha, "saved receipt SHA")
    data = json.loads(Path(receipt).read_bytes())
    need(data["status"] == "saved_checks_passed" and len(data["rows"]) == 24, "all rows")
    anchors = run.h10.native.read(run.h10.INPUT, cap=64*1024**2)["anchors"]
    result = []
    for i, (row, anchor) in enumerate(zip(data["rows"], anchors, strict=True)):
        folder = Path(directory)/f"row_{i:02d}"
        report = run.h10.native.read(folder/"report.json", cap=64*1024**2)
        need(run.sha(folder/"journal.jsonl") == report["journal_sha256"], "journal SHA")
        events = saved.saved.events(folder/"journal.jsonl", report["journal_chain_tail"])
        attached = [e for e in events if e["kind"] == "goal_attached"]
        need(len(attached) == 1 and attached[0]["goal"] == anchor["goal"], "goal identity")
        transitions = [e for e in events if e["kind"] == "returned_transition"]
        for event in transitions:
            phase, step = event["phase"], event["step"]
            ops = [e for e in events if e["kind"] == "operation_returned" and
                   e["receipt"].get("resource") == "public_step" and e["receipt"].get("phase") == phase
                   and e["receipt"].get("step") == step]
            need(len(ops) == 1 and ops[0]["index"] < event["index"] and
                 ops[0]["receipt"]["raw_bits"] == event["actual_bits"] == event["requested_bits"],
                 "action leaf and returned transition")
            if phase != "warmup":
                need(event["goal_state"] == anchor["goal"], "goal transition")
                metrics = run.h10.base.previous.metric(anchor["goal"], event["state"])
                need(event["reward"] == -metrics["native_distance"] and
                     event["terminated"] == metrics["native_success"], "native reward/success")
            if phase != "probe":
                need(not event["terminated"] and not event["truncated"], "prefix native stop")
        probe = [e for e in transitions if e["phase"] == "probe"]
        for response, event in zip(report["row"]["responses"], probe, strict=False):
            d = np.asarray(event["state"]) - np.asarray(report["row"]["start_state"])
            need(response["delta_native7"] == d.tolist() and
                 response["delta_angle_wrapped"] == float(np.arctan2(np.sin(d[4]), np.cos(d[4]))),
                 "response arithmetic")
        need(row["index"] == i and len(probe) == row["returned_suffix"], "returned count")
        result.append({"index": i, "returned_suffix": len(probe), "status": row["status"]})
    out = Path(output)
    need(out.is_absolute() and out.parent == run.ROOT/"results" and
         out.name.startswith("codex_p1_cp5_physical_crosscheck_20261030_v1_"), "exclusive output")
    result = {"schema": "codex_p1_cp5_physical_crosscheck_20261030_v1", "receipt_sha256": receipt_sha,
              "rows": result, "status": "checked", "P1_admission": False, "independent_confirmation": False,
              "limitation": "Known roots and trusted host; no OS isolation or fairness certification."}
    return run.h10._write_new(out, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--receipt", required=True)
    p.add_argument("--receipt-sha", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps({"sha256": check(a.directory, a.receipt, a.receipt_sha, a.output)}))
