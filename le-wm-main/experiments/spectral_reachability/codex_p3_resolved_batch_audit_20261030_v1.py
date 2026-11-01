"""Bound partial-batch audit; never converts a deadline failure to completion."""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p3_resolved_delivery_20261030_v1 as delivery

ROOT = delivery.ROOT
NAME = "codex_p3_resolved_batch_audit_20261030_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(folder, report_sha):
    folder = Path(folder)
    if sha(folder/"report.json") != report_sha:
        raise ValueError("batch report hash")
    report = json.loads((folder/"report.json").read_bytes())
    events = delivery.host.journal.read(folder/"events.jsonl", report["journal_tail"], 2*1024**2)
    if (report["error"] is not None or report["scientific_pass"] is not False or
            [(r["point"], r["arm"]) for r in report["rows"]] != list(delivery.SLOTS) or
            len(events) != 14 or events[0]["kind"] != "batch_started" or
            events[0]["contract_sha256"] != report["contract_sha256"] or
            events[-1]["kind"] != "batch_terminal" or events[-1]["completed_slots"] != 6 or
            [events[2*i+2].get("row") for i in range(6)] != report["rows"]):
        raise ValueError("batch identity or chain")
    rows, sets = [], {}
    for row in report["rows"]:
        point, arm = row["point"], row["arm"]
        item = {"point": point, "arm": arm, "delivered_and_saved": row["delivered_and_saved"],
                "process_wall_seconds": row["attempt"]["wall_seconds"], "allocation": None,
                "scheduler_calls": None, "scheduler_meter_seconds": None, "candidate_counts": None}
        if row["delivered_and_saved"]:
            run = ROOT/"results"/(delivery.host.NAME+f"_RUN_WALL_P{point}_{arm}_01")
            receipt = ROOT/"results"/(delivery.host.NAME+f"_CONSUME_WALL_P{point}_{arm}_01.json")
            if (str(run) != row["run"] or str(receipt) != row["receipt"] or run.is_symlink()
                    or receipt.is_symlink() or sha(run/"report.json") != row["report_sha256"]
                    or sha(receipt) != row["receipt_sha256"]):
                raise ValueError("source identity")
            host_report = json.loads((run/"report.json").read_bytes())
            checked = json.loads(receipt.read_bytes())
            if (host_report["status"] != "panel_completed_unverified" or
                    host_report["model_before"] != host_report["model_after"] or
                    checked["status"] != "semantic_saved_checked_not_scientifically_admitted" or
                    len(checked["rows"]) != 1):
                raise ValueError("slot terminal")
            saved = checked["rows"][0]
            if (saved["point"] != point or saved["arm"] != arm or saved["allocation"] != row["allocation"]
                    or saved["attempted"] != row["actual"] or saved["attempted"]["scheduler"] != 1):
                raise ValueError("slot ledger")
            journal = delivery.host.journal.read(run/"events.jsonl", host_report["journal_tail"],
                                                  delivery.host.CAP)
            candidates = [[e["candidate"]["index"] for e in journal if e["kind"] == "candidate_complete"
                           and e["scope"] == [point, arm, "search", root]] for root in (0, 1)]
            sets[point, arm] = candidates
            attempts = [e["entry"] for e in journal if e["kind"] == "resource_attempt"
                        and e["entry"]["resource"] == "scheduler"]
            returns = [e["entry"] for e in journal if e["kind"] == "resource_return"
                       and e["entry"]["resource"] == "scheduler"]
            if len(attempts) != 1 or len(returns) != 1 or returns[0]["index"] != attempts[0]["index"]:
                raise ValueError("scheduler events")
            item.update(allocation=saved["allocation"], scheduler_calls=1,
                        scheduler_meter_seconds=returns[0]["ended_seconds"]-attempts[0]["started_seconds"],
                        candidate_counts=[len(x) for x in candidates])
            if item["candidate_counts"] != item["allocation"]:
                raise ValueError("effective candidate count")
        else:
            item["failure"] = row["attempt"]
        rows.append(item)
    diff = {}
    for point in (20, 30):
        a, b = sets.get((point, "signal")), sets.get((point, "reverse"))
        if a is not None and b is not None:
            diff[str(point)] = [len(set(a[i]) ^ set(b[i])) for i in (0, 1)]
    complete = all(r["delivered_and_saved"] for r in rows)
    return {"schema": NAME, "status": "complete" if complete else "partial_or_failed",
            "source_report_sha256": report_sha, "rows": rows,
            "signal_reverse_candidate_symmetric_difference": diff,
            "scheduler_timing_scope": "meter dispatch through synchronized return, not pure CPU instruction time",
            "source_or_module_recomputation_certified": False, "scientific_pass": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--report-sha", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    if not (output.is_absolute() and output.parent == ROOT/"results" and output.name.startswith(NAME+"_")):
        raise ValueError("exclusive output")
    result = build(args.directory, args.report_sha)
    digest = delivery.host.write_new(output, result)
    print(json.dumps({"status": result["status"], "sha256": digest}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
