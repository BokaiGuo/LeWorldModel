"""Saved-only verification of the exploratory dev303 four-arm prefix run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import p1_dev303_v3_four_arm_physical_20261102_exclusive as run
from . import p1_dev301_blind_plan_physical_saved_20261101_codex as saved

base, require = run.base, run.require


def verify_branch(directory, spec, row, events, history):
    arm = spec["role"]
    require(row["arm"] == arm and row["status"] in
            ("failed", "native_early_stop", "fixed_horizon_completed"), "branch status")
    require(events and events[0]["kind"] == "branch_started"
            and events[-1]["kind"] == "branch_closed"
            and events[-1]["status"] == row["status"]
            and all(event.get("arm") == arm for event in events), "branch boundaries")
    core = events[1:-1]
    branch_errors = [e for e in core if e["kind"] == "branch_error"]
    require((len(branch_errors) == 1 and row["status"] == "failed"
             and row["error"] == branch_errors[0]["error"])
            or (not branch_errors and row["status"] != "failed" and row["error"] is None),
            "failure identity")
    if branch_errors:
        require(core[-1]["kind"] in ("branch_error", "close_error"), "error ordering")
    core = [e for e in core if e["kind"] not in ("branch_error", "close_error")]
    plan = [("reset_verified", None, None, None), ("observation", "warmup", 0, None)]
    for phase, actions in (("warmup", [[0.0, 0.0]] * 10), ("prefix", spec["actions"])):
        for i, action in enumerate(actions, 1):
            plan.extend((kind, phase, i, action)
                        for kind in ("action_dispatch", "action_observed", "action_return"))
            if phase == "prefix" or i in (5, 10):
                plan.append(("observation", phase, i, None))
    require(len(core) <= len(plan), "event count")
    refs = {p["step"]: p for p in history["frames"]}
    transitions, observations, pending, observed = [], [], None, None
    stopped = False
    for event, (kind, phase, step, action) in zip(core, plan):
        require(event["kind"] == kind and (phase is None or
                (event["phase"] == phase and event["step"] == step)), "event sequence")
        if kind == "action_dispatch":
            require(not stopped and event["action"] == action, "frozen dispatch")
            pending = event
        elif kind == "action_observed":
            require(pending is not None, "observed without dispatch")
            observed = event
        elif kind == "action_return":
            require(pending is not None and observed is not None and event["action"] == action
                    and all(observed[k] == event[k] for k in ("action", "terminated", "truncated")),
                    "actual action or stop mismatch")
            transition = {k: event[k] for k in
                          ("phase", "step", "action", "state", "reward", "terminated", "truncated")}
            transitions.append(transition)
            stopped = event["terminated"] or event["truncated"]
            pending = observed = None
        elif kind == "observation":
            point = {k: v for k, v in event.items()
                     if k not in ("index", "previous", "kind", "arm", "sha256")}
            saved.frame(directory, point)
            if phase == "warmup":
                ref = refs[step]
                require(all(point[k] == ref[k] for k in ("state", "snapshot", "pixel_sha256")),
                        "warmup checkpoint")
            elif phase == "prefix":
                require(transitions and point["state"] == transitions[-1]["state"], "frame/state")
            observations.append(point)
    require(transitions == row["transitions"] and observations == row["observations"], "row/journal")
    if row["status"] == "fixed_horizon_completed":
        require(len(core) == len(plan) and not stopped, "complete horizon")
    elif row["status"] == "native_early_stop":
        require(stopped and transitions[-1]["phase"] == "prefix"
                and core[-1]["kind"] == "observation"
                and core[-1]["step"] == transitions[-1]["step"], "native stop")
    return {"arm": arm, "status": row["status"],
            "prefix_returned": sum(t["phase"] == "prefix" for t in transitions)}


def verify(prepared, prepared_sha, directory, report_sha):
    prepared, directory = Path(prepared).resolve(strict=True), Path(directory).resolve(strict=True)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == run.build(),
            "contract and source closure")
    require(base.sha(directory / "report.json") == report_sha, "external report SHA")
    report = base.read(directory / "report.json")
    require(report["schema"] == run.NAME + ".report.v1" and report["prepare_sha256"] == prepared_sha
            and report["status"] == ("partial_or_failed" if report["errors"] else "recorded_development")
            and all(report[k] is False for k in
                    ("ledger_verified", "equal_budget_claim", "P1_admission", "independent_confirmation")),
            "report boundary")
    require(base.read(directory / "execution_binding.json") ==
            {"prepared": str(prepared), "prepare_sha256": prepared_sha}, "execution binding")
    journal = directory / "journal.jsonl"
    require(base.sha(journal) == report["journal_sha256"], "journal SHA")
    events = saved.journal(journal, report["journal_tail"])
    require(events[0]["kind"] == "run_started" and events[0]["prepare_sha256"] == prepared_sha
            and events[-1]["kind"] == "run_finished" and events[-1]["errors"] == report["errors"],
            "journal endpoints")
    specs = base.read(prepared)["arms"]
    require([r["arm"] for r in report["rows"]] == [s["role"] for s in specs], "fixed arm roster")
    _, receipt, _ = run.prior.source_data()
    cursor, verified = 1, []
    for spec, row in zip(specs, report["rows"], strict=True):
        require(base.read(directory / (spec["role"] + ".row.json")) == row, "row shard")
        end = cursor
        while end < len(events) - 1 and events[end]["kind"] != "branch_closed":
            end += 1
        require(end < len(events) - 1, "branch closure")
        verified.append(verify_branch(directory, spec, row, events[cursor:end + 1], receipt["history"]))
        cursor = end + 1
    require(cursor == len(events) - 1, "unscoped events")
    require(all({"arm": r["arm"], **r["error"]} in report["errors"]
                for r in report["rows"] if r["error"] is not None), "unreported error")
    return {"status": "saved_chain_verified_development_only", "branches": verified,
            "run_status": report["status"], "errors": report["errors"], "ledger_verified": False,
            "equal_budget_claim": False, "P1_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser()
    for name in ("prepared", "prepared-sha", "directory", "report-sha"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.prepared, args.prepared_sha, args.directory, args.report_sha)))


if __name__ == "__main__":
    main()
