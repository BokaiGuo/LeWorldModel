"""Saved-chain verification of a fixed full-cross prefix/common-tail run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_v3_cross_common_tail_20261102_exclusive as run

base, require = run.base, run.require
chain = run.replay.verifier


def verify(prepared, prepared_sha, directory, report_sha):
    prepared, directory = Path(prepared).resolve(strict=True), Path(directory).resolve(strict=True)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == run.build(),
            "contract/source closure")
    require(base.sha(directory / "report.json") == report_sha, "external report SHA")
    report = base.read(directory / "report.json")
    require(report["schema"] == run.NAME + ".report.v1" and report["prepare_sha256"] == prepared_sha
            and report["status"] == ("partial_or_failed" if report["errors"] else "recorded_development")
            and all(report[k] is False for k in
                    ("P1_admission", "independent_confirmation", "ledger_verified", "equal_budget_claim")),
            "report boundary")
    require(base.read(directory / "execution_binding.json") ==
            {"prepared": str(prepared), "prepare_sha256": prepared_sha}, "execution binding")
    path = directory / "journal.jsonl"
    require(base.sha(path) == report["journal_sha256"], "journal SHA")
    events = chain.saved.journal(path, report["journal_tail"])
    require(events[0]["kind"] == "run_started" and events[0]["prepare_sha256"] == prepared_sha
            and events[-1]["kind"] == "run_finished" and events[-1]["errors"] == report["errors"],
            "journal boundaries")
    branches = base.read(prepared)["branches"]
    require([r["arm"] for r in report["rows"]] == [r["arm"] for r in branches], "complete ordered roster")
    _, receipt, _ = run.prior.source_data()
    cursor, results = 1, []
    for spec, row in zip(branches, report["rows"], strict=True):
        require(base.read(directory / (spec["arm"] + ".row.json")) == row, "row shard")
        end = cursor
        while end < len(events) - 1 and events[end]["kind"] != "branch_closed":
            end += 1
        require(end < len(events) - 1, "branch closure")
        segment = events[cursor:end + 1]
        require(all(e.get("arm") == spec["arm"] for e in segment), "branch isolation")
        summary = chain.verify_branch(directory, spec, row, segment, receipt["history"])
        for transition in row["transitions"]:
            state = np.asarray(transition["state"], dtype=np.float64)
            require(state.ndim == 1 and len(state) >= 5 and np.isfinite(state).all(),
                    "finite object pose state")
        count = summary["tail_returned"]
        results.append({"arm": spec["arm"], "prefix_role": spec["prefix_role"],
                        "tail_role": spec["tail_role"], "status": row["status"],
                        "tail_returned": count, "tail_unobserved": 10 - count,
                        "censored": row["status"] != "fixed_horizon_completed"})
        cursor = end + 1
    require(cursor == len(events) - 1, "unscoped events")
    require(all({"arm": row["arm"], **row["error"]} in report["errors"]
                for row in report["rows"] if row["error"] is not None), "reported branch errors")
    return {"status": "saved_chain_verified_development_only", "branches": results,
            "run_status": report["status"], "errors": report["errors"], "P1_admission": False,
            "independent_confirmation": False, "ledger_verified": False, "equal_budget_claim": False}


def main():
    parser = argparse.ArgumentParser()
    for name in ("prepared", "prepared-sha", "directory", "report-sha"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.prepared, args.prepared_sha, args.directory, args.report_sha)))


if __name__ == "__main__":
    main()
