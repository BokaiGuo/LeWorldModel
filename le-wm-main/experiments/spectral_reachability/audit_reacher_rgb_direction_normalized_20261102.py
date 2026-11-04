"""Validate and pair normalized-direction runs with sealed development baselines."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import audit_reacher_response_svd_cp5_cem_20261102 as audit
from . import audit_reacher_rgb_direction_cem_20261102 as prior
from . import compare_reacher_q_l2_cp5_20261102 as pairing
from . import reacher_rgb_direction_normalized_20261102 as method

ROOT = method.base.ROOT
STRONG_JOURNAL = ROOT / "results/reacher_response_svd_cp5_rgb_direction_RUN_20261102_v1/events.jsonl"
STRONG_JOURNAL_SHA = "13adeb4ba17140ca70de76c6c47d43fae8dc2805d1954faed8bd4c73824d5239"
STRONG_RECEIPT = ROOT / "results/reacher_response_svd_rgb_direction_SAVED_20261102_v2.json"
STRONG_RECEIPT_SHA = "68744fa686f020dd9ea1538c4b2d11638ad456658dffd0569eba8daaacae6250"


def consume(plan_path, plan_sha, journal_path, journal_sha):
    host, bridge = method.base.host, method.base.bridge
    for path, digest in ((prior.L2_JOURNAL, prior.L2_JOURNAL_SHA),
                         (prior.L2_RECEIPT, prior.L2_RECEIPT_SHA),
                         (STRONG_JOURNAL, STRONG_JOURNAL_SHA),
                         (STRONG_RECEIPT, STRONG_RECEIPT_SHA)):
        host.require(bridge.sha_file(path) == digest, "baseline source")
    original = audit.experiment.make_plan
    try:
        audit.experiment.make_plan = method.make_plan
        result = audit.consume(plan_path, plan_sha, journal_path, journal_sha)
    finally:
        audit.experiment.make_plan = original
    events = audit.saved.verify_journal(journal_path)
    receipts = [e["result"]["suffix_receipt"] for e in events if e["kind"] == "arm_returned"]
    host.require(len(receipts) == 26 and all(
        r["objective"] == "observed_rgb_direction_normalized" and r["direction_relative_weight"] == method.WEIGHT
        and r["measured_suffix_calls"] == {"encode_calls": 4, "predict_calls": 4, "predict_samples": 64}
        and math.isfinite(r["measured_suffix_wall_seconds"]) and r["measured_suffix_wall_seconds"] > 0
        for r in receipts), "measured call receipt")
    pairs = {}
    for label, path, receipt in (("l2", prior.L2_JOURNAL, prior.L2_RECEIPT),
                                 ("strong_direction", STRONG_JOURNAL, STRONG_RECEIPT)):
        baseline = json.loads(receipt.read_text())
        host.require(baseline["status"] == "complete_saved_consistency"
                     and baseline["journal_sha256"] == bridge.sha_file(path), "baseline receipt")
        paired = pairing.paired(audit.saved.verify_journal(path), events, baseline["results"], result["results"])
        rows = []
        for row in paired["rows"]:
            row = dict(row)
            row["baseline_final"] = row.pop("l2_final")
            row["normalized_final"] = row.pop("q_final")
            row["normalized_minus_baseline"] = row.pop("q_minus_l2_final")
            rows.append(row)
        pairs[label] = rows
    result.update(pairs=pairs, suffix_encode_calls=sum(r["measured_suffix_calls"]["encode_calls"] for r in receipts),
                  suffix_predict_calls=sum(r["measured_suffix_calls"]["predict_calls"] for r in receipts),
                  suffix_predict_samples=sum(r["measured_suffix_calls"]["predict_samples"] for r in receipts),
                  suffix_wall_seconds=sum(r["measured_suffix_wall_seconds"] for r in receipts),
                  limits=["exposed roots; no P1 or generalization", "saved consistency, not independent simulator replay",
                          "instrumentation covers suffix only, not total method costs"])
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ("plan", "plan-sha", "journal", "journal-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = consume(args.plan, args.plan_sha, args.journal, args.journal_sha)
    method.base.host.write_new(args.output, result)
    print(json.dumps({"sha256": method.base.bridge.sha_file(Path(args.output)),
                      "suffix_encode_calls": result["suffix_encode_calls"],
                      "suffix_predict_samples": result["suffix_predict_samples"]}))


if __name__ == "__main__":
    main()
