"""Read-only cross-bind of fixed batch, terminal journal and six receipts.

This does not independently recompute neural module arrays or prove FLOPs.
"""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p3_proportional_delivery_20261030_v1 as delivery

ROOT = delivery.ROOT
NAME = "codex_p3_proportional_batch_verify_20261030_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_rows(report, events, load_receipt):
    expected = list(delivery.SLOTS)
    if (report.get("error") is not None or report.get("scientific_pass") is not False
            or [(r.get("point"), r.get("arm")) for r in report.get("rows", [])] != expected):
        raise ValueError("batch roster or terminal")
    if (len(events) != 14 or events[0].get("kind") != "batch_started"
            or events[-1].get("kind") != "batch_terminal"
            or events[-1].get("completed_slots") != 6 or events[-1].get("failure") is not None
            or events[0].get("contract_sha256") != report.get("contract_sha256")):
        raise ValueError("batch journal boundaries")
    summaries = []
    for i, (point, arm) in enumerate(expected):
        attempt, result = events[1+2*i:3+2*i]
        row = report["rows"][i]
        if (attempt.get("kind") != "slot_attempt" or attempt.get("point") != point
                or attempt.get("arm") != arm or attempt.get("run") != row.get("run")
                or result.get("kind") != "slot_result" or result.get("row") != row
                or not row.get("delivered_and_saved") or row["attempt"]["exit_code"] != 0
                or row["attempt"]["deadline_exceeded"] or row["attempt"]["wall_seconds"] > 30):
            raise ValueError("slot event or delivery")
        receipt = load_receipt(row)
        if (receipt.get("status") != "semantic_saved_checked_not_scientifically_admitted"
                or len(receipt.get("rows", [])) != 1
                or receipt.get("scientific_pass") is not False):
            raise ValueError("receipt status")
        actual = receipt["rows"][0]
        if (actual["point"] != point or actual["arm"] != arm
                or actual["allocation"] != row["allocation"] or actual["attempted"] != row["actual"]):
            raise ValueError("receipt versus batch")
        summaries.append({"point": point, "arm": arm, "allocation": actual["allocation"],
                          "weights": actual["weights"], "actual": actual["attempted"],
                          "winners": [{"index": w["index"], "model_cost": w["cost"]}
                                      for w in actual["winners"]],
                          "process_wall_seconds": row["attempt"]["wall_seconds"]})
    return summaries


def build(directory, report_sha):
    directory = Path(directory)
    report_path = directory/"report.json"
    if sha(report_path) != report_sha:
        raise ValueError("external report hash")
    report = json.loads(report_path.read_bytes())
    events = delivery.host.journal.read(directory/"events.jsonl", report["journal_tail"], 2*1024**2)

    def load(row):
        run = ROOT/"results"/(delivery.host.NAME+f"_RUN_WALL_P{row['point']}_{row['arm']}_01")
        receipt = ROOT/"results"/(delivery.host.NAME+f"_CONSUME_WALL_P{row['point']}_{row['arm']}_01.json")
        if (str(run) != row["run"] or str(receipt) != row["receipt"] or run.is_symlink()
                or receipt.is_symlink() or sha(run/"report.json") != row["report_sha256"]
                or sha(receipt) != row["receipt_sha256"]):
            raise ValueError("slot artifact binding")
        host_report = json.loads((run/"report.json").read_bytes())
        if (host_report["point"] != row["point"] or host_report["arm"] != row["arm"]
                or host_report["model_before"] != host_report["model_after"]):
            raise ValueError("host model/identity")
        return json.loads(receipt.read_bytes())

    summaries = verify_rows(report, events, load)
    if sha(report_path) != report_sha:
        raise ValueError("report drift")
    return {"schema": NAME, "source_report_sha256": report_sha, "rows": summaries,
            "status": "batch_bindings_checked", "dynamic_source_closure": "unknown",
            "model_arrays_independently_recomputed": False, "flops": "unknown",
            "physical_certified": False, "scientific_pass": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--report-sha", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if not (out.is_absolute() and out.parent == ROOT/"results" and out.name.startswith(NAME+"_")):
        raise ValueError("exclusive output")
    result = build(a.directory, a.report_sha)
    print(json.dumps({"sha256": delivery.host.write_new(out, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
