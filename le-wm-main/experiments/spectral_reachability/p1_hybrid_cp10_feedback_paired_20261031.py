"""Pair saved same-prefix open and cp10-feedback physical outcomes."""
import argparse
import json
from pathlib import Path

from . import p1_hybrid_cp10_feedback_saved_20261031 as check
from .codex_p1_h10_paired_table_20261030_main import physical_metrics

ROOT, need, sha = check.run.ROOT, check.run.need, check.run.sha
OLD = ROOT / "results/p1_hybrid_cp5_physical_saved_20261031_RUN01.json"
OLD_SHA = "70068a853c359b37d74dcac17f0121e9978b7227e4c85373582cb74a3b9764ff"
NEW_PREPARE = ROOT / "results/p1_hybrid_cp10_feedback_physical_20261031_prepare_RUN02.json"
NEW_PREPARE_SHA = "50536c52d9d2830356d4f9a11596703b038d6507dbe36cb56fa77780abbfd2c7"
OLD_PREPARE = ROOT / "results/p1_hybrid_cp5_physical_20261031_prepare_RUN01.json"
OLD_PREPARE_SHA = "b91993a25e7c504d16015f0509685ef76566e9655d451a92918c10f06d6b243a"


def build(receipt, receipt_sha):
    for path, digest in ((OLD, OLD_SHA), (NEW_PREPARE, NEW_PREPARE_SHA),
                         (OLD_PREPARE, OLD_PREPARE_SHA), (receipt, receipt_sha)):
        need(sha(path) == digest, "pinned input")
    original, feedback = check.run.old.h.read(OLD), check.run.old.h.read(receipt)
    op, np = check.run.old.h.read(OLD_PREPARE), check.run.old.h.read(NEW_PREPARE)
    need(original["status"] == feedback["status"] == "saved_checks_passed"
         and len(original["rows"]) == len(feedback["rows"]) == 2
         and np["anchors"] == op["anchors"], "fixed roots")
    rows = []
    for anchor, old_item, new_item, a, b, midpoint in zip(
            op["anchors"], op["items"], np["items"], original["rows"], feedback["rows"],
            np["midpoints"], strict=True):
        need(anchor["index"] == old_item["index"] == new_item["index"] == a["index"] == b["index"]
             and bytes.fromhex(old_item["raw10_float32_hex"])[:40] ==
             bytes.fromhex(new_item["raw10_float32_hex"])[:40], "same physical first ten actions")
        goal = anchor["goal"]
        def outcome(row):
            return {"status": row["status"], "returned_suffix": row["returned_suffix"],
                    "full_endpoint": physical_metrics(row["completed_endpoint"], goal),
                    "last_accepted_state_metrics": row["last_accepted_state_metrics"]}
        rows.append({"root_slot": anchor["root_slot"], "task_id": anchor["task_id"],
                     "prefix": physical_metrics(anchor["checkpoint"]["state"], goal),
                     "midpoint": physical_metrics(midpoint["state"], goal),
                     "open": outcome(a), "feedback": outcome(b)})
    return {"schema": "p1_hybrid_cp10_feedback_paired_20261031", "status": "exploratory",
            "rows": rows, "receipt_sha256": receipt_sha, "P1_admission": False,
            "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": "Two known roots; feedback adds four 20-update optimizations and RGB encoding."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--receipt-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith("p1_hybrid_cp10_feedback_paired_20261031_") and not output.exists(),
         "exclusive output")
    result = build(args.receipt, args.receipt_sha)
    print(json.dumps({"sha256": check.run.old.h.engine._write_new(output, result), "rows": result["rows"]}))


if __name__ == "__main__":
    main()
