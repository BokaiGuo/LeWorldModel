"""Recount a verified fixed-prefix panel; does not admit physical evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load_verified(report_path, receipt_path, receipt_sha):
    receipt_bytes = Path(receipt_path).read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha:
        raise ValueError("receipt SHA mismatch")
    receipt = json.loads(receipt_bytes)
    report_bytes = Path(report_path).read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != receipt.get("report_sha256"):
        raise ValueError("report SHA mismatch")
    if receipt.get("status") != "verified_model_only" or receipt.get("complete") is not True:
        raise ValueError("receipt is not verified and complete")
    return json.loads(report_bytes), receipt


def recount(report, receipt):
    rows = report["rows"]
    if len(rows) != receipt["group_denominator"]:
        raise ValueError("group denominator mismatch")
    counts = {"candidates": 0, "uphill": 0, "better": 0, "witness": 0}
    for row in rows:
        result = row["result"]
        initial, stage, baseline = (result[k] for k in ("initial_Q", "stage_Q", "baseline_Q"))
        if not all(math.isfinite(x) for x in (initial, stage, baseline)):
            raise ValueError("non-finite anchor")
        for candidate in result["iterates"]:
            terminal = candidate["terminal_Q"]
            if not math.isfinite(terminal):
                raise ValueError("non-finite terminal")
            uphill, better = stage > initial, terminal < baseline
            if (candidate["uphill_from_initial"], candidate["better_terminal_than_original"], candidate["witness"]) != (uphill, better, uphill and better):
                raise ValueError("saved candidate predicates disagree with recomputation")
            counts["candidates"] += 1
            counts["uphill"] += uphill
            counts["better"] += better
            counts["witness"] += uphill and better
    if counts["candidates"] != receipt["candidate_denominator"]:
        raise ValueError("candidate denominator mismatch")
    return {"scope": "verified_model_only", "group_count": len(rows), **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--receipt-sha", required=True)
    args = parser.parse_args()
    report, receipt = load_verified(args.report, args.receipt, args.receipt_sha)
    print(json.dumps(recount(report, receipt), sort_keys=True))


if __name__ == "__main__":
    main()
