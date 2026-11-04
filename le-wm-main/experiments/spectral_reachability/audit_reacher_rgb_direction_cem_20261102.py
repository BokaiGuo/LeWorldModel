"""Saved-only validation and fixed-context L2 pairing for RGB-direction CEM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import audit_reacher_response_svd_cp5_cem_20261102 as audit
from . import compare_reacher_q_l2_cp5_20261102 as pairing
from . import reacher_response_svd_rgb_direction_cem_20261102 as method

ROOT = method.base.ROOT
L2_JOURNAL = ROOT / "results/reacher_response_svd_cp5_RUN_20261102_v1/events.jsonl"
L2_JOURNAL_SHA = "e9c933f8d78abc9787198e324240afdb1186475fa08189a2787e39a3b9480f43"
L2_RECEIPT = ROOT / "results/reacher_response_svd_cp5_SAVED_20261102_v1.json"
L2_RECEIPT_SHA = "20c412e0bfd3d7cc0710005af62892ec9ab805b76087c988917e2917b9f44207"


def consume(plan_path, plan_sha, journal_path, journal_sha):
    host = method.base.host
    host.require(method.base.bridge.sha_file(L2_JOURNAL) == L2_JOURNAL_SHA
                 and method.base.bridge.sha_file(L2_RECEIPT) == L2_RECEIPT_SHA, "L2 sources")
    original = audit.experiment.make_plan
    try:
        audit.experiment.make_plan = method.make_plan
        result = audit.consume(plan_path, plan_sha, journal_path, journal_sha)
    finally:
        audit.experiment.make_plan = original
    events = audit.saved.verify_journal(journal_path)
    receipts = [e["result"]["suffix_receipt"] for e in events if e["kind"] == "arm_returned"]
    host.require(len(receipts) == 26 and all(
        r["objective"] == "observed_rgb_direction_plus_l2" and r["direction_weight"] == method.WEIGHT
        and r["additional_encoder_calls"] == 2 and r["forward_rollout_evaluations"] == 32
        for r in receipts), "metric or candidate receipt")
    l2 = json.loads(L2_RECEIPT.read_text())
    host.require(l2["status"] == "complete_saved_consistency" and l2["journal_sha256"] == L2_JOURNAL_SHA,
                 "L2 receipt")
    paired = pairing.paired(audit.saved.verify_journal(L2_JOURNAL), events, l2["results"], result["results"])
    rows = []
    for row in paired["rows"]:
        row = dict(row)
        row["direction_final"] = row.pop("q_final")
        row["direction_minus_l2_final"] = row.pop("q_minus_l2_final")
        rows.append(row)
    result.update(paired_rows=rows, metric="observed_rgb_direction_plus_l2",
                  extra_encoder_calls_reported=52, L2_journal_sha256=L2_JOURNAL_SHA,
                  L2_receipt_sha256=L2_RECEIPT_SHA,
                  limits=["saved consistency only; no independent simulator replay",
                          "extra encoder calls reported, not instrumented",
                          "exposed roots; no total-compute or P1 certification"])
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ("plan", "plan-sha", "journal", "journal-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = consume(args.plan, args.plan_sha, args.journal, args.journal_sha)
    method.base.host.write_new(args.output, result)
    print(json.dumps({"sha256": method.base.bridge.sha_file(Path(args.output)), "rows": result["paired_rows"]}))


if __name__ == "__main__":
    main()
