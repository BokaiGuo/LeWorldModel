"""Independently bind saved per-candidate winners to optimizer return events."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from . import p2_same_bank_host_development_20261102_exclusive as runner


def check_result_events(result, events):
    recorded = {}
    for item in events:
        row = item["record"]
        if row["kind"] != "optimizer_returned":
            continue
        event = row["payload"]
        key = (event["candidate_index"], event["arm"], event["iteration"])
        if key in recorded:
            raise ValueError("duplicate optimizer return")
        if type(event["score"]) not in (int, float) or not math.isfinite(event["score"]):
            raise ValueError("invalid optimizer score")
        recorded[key] = event
    expected = {(i, arm, j) for i in range(16) for arm in runner.native.optimizer.ARMS for j in range(9)}
    if set(recorded) != expected or [row["index"] for row in result["rows"]] != list(range(16)):
        raise ValueError("return or result census mismatch")
    for row in result["rows"]:
        i = row["index"]
        if [entry["arm"] for entry in row["arms"]] != list(runner.native.optimizer.ARMS):
            raise ValueError("result arm order")
        for entry in row["arms"]:
            arm = entry["arm"]
            best = min((recorded[i, arm, j] for j in range(9)),
                       key=lambda e: (e["score"], e["iteration"]))
            raw = bytes.fromhex(entry["raw_float32_hex"])
            if (len(raw) != 80 or entry["winner_iteration"] != best["iteration"] or
                    entry["model_score"] != best["score"] or
                    hashlib.sha256(raw).hexdigest() != best["action_sha256"]):
                raise ValueError("winner not bound to best optimizer event")
    return {"candidate_arm_winners_bound": 32, "optimizer_returns_checked": 288}


def crosscheck(output, report_sha, profile="native_hooks"):
    output = Path(output)
    result = runner.consume(output, report_sha, profile)
    report_bytes = (output / "report.json").read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != report_sha:
        raise ValueError("report changed")
    report = json.loads(report_bytes)
    journal = (output / "journal.jsonl").read_bytes()
    if hashlib.sha256(journal).hexdigest() != report["journal_sha256"]:
        raise ValueError("journal changed")
    checked = check_result_events(report["result"], [json.loads(x) for x in journal.splitlines()])
    return {"status": "saved_winner_events_bound_not_admitted", "report_sha256": report_sha,
            **checked, "winners": result["winners"], "equal_total_compute": False,
            "physical_execution_authorized": False, "native_execution_independently_authenticated": False,
            "P1_admission": False, "P2_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(crosscheck(args.output, args.report_sha256), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
