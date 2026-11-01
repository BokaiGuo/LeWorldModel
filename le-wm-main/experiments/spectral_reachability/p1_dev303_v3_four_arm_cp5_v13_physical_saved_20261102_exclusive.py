"""Read-only saved verification for four-arm V13 physical replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import p1_dev303_v3_four_arm_cp5_v13_physical_20261102_exclusive as run

base, require = run.base, run.require
chain = run.replay.verifier


def verify(prepared, prepared_sha, directory, report_sha):
    prepared, directory = Path(prepared).resolve(strict=True), Path(directory).resolve(strict=True)
    contract = base.read(prepared)
    require(base.sha(prepared) == prepared_sha
            and contract == run.build(contract["model_report"], contract["model_report_sha256"]),
            "contract/source closure")
    require(base.sha(directory / "report.json") == report_sha, "external report SHA")
    report = base.read(directory / "report.json")
    require(report["schema"] == run.NAME + ".report.v1" and report["prepare_sha256"] == prepared_sha
            and report["status"] == ("partial_or_failed" if report["errors"] else "recorded_development")
            and all(report[k] is False for k in
                    ("P1_admission", "independent_confirmation", "ledger_verified", "equal_full_budget_certified")),
            "report boundary")
    require(base.read(directory / "execution_binding.json") ==
            {"prepared": str(prepared), "prepare_sha256": prepared_sha}, "execution binding")
    path = directory / "journal.jsonl"
    require(base.sha(path) == report["journal_sha256"], "journal SHA")
    events = chain.saved.journal(path, report["journal_tail"])
    require(events[0]["kind"] == "run_started" and events[0]["prepare_sha256"] == prepared_sha
            and events[-1]["kind"] == "run_finished" and events[-1]["errors"] == report["errors"],
            "journal boundaries")
    roles = contract["roles"]
    require([r["arm"] for r in report["rows"]] == list(run.inputs.ROLES)
            and [r["arm"] for r in roles] == list(run.inputs.ROLES), "four-arm roster")
    _, receipt, _ = run.inputs.physical.prior.source_data()
    cursor, verified = 1, []
    for role, row in zip(roles, report["rows"], strict=True):
        require(base.read(directory / (role["arm"] + ".row.json")) == row, "row shard")
        end = cursor
        while end < len(events) - 1 and events[end]["kind"] != "branch_closed":
            end += 1
        require(end < len(events) - 1, "branch closure")
        segment = events[cursor:end + 1]
        require(all(e.get("arm") == role["arm"] for e in segment), "branch isolation")
        verified.append(chain.verify_branch(directory, role, row, segment, receipt["history"]))
        cursor = end + 1
    require(cursor == len(events) - 1, "unscoped events")
    require(all({"arm": row["arm"], **row["error"]} in report["errors"]
                for row in report["rows"] if row["error"] is not None), "unreported failure")
    return {"status": "saved_chain_verified_development_only", "branches": verified,
            "run_status": report["status"], "errors": report["errors"], "P1_admission": False,
            "independent_confirmation": False, "ledger_verified": False,
            "equal_full_budget_certified": False}


def main():
    parser = argparse.ArgumentParser()
    for arg in ("prepared", "prepared-sha", "directory", "report-sha"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.prepared, args.prepared_sha, args.directory, args.report_sha)))


if __name__ == "__main__":
    main()
