"""Saved-only verification of prospective dev303 tail diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import p1_dev301_blind_plan_physical_20261101_codex as base
from . import p1_dev301_blind_plan_physical_saved_20261101_codex as saved
from . import p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as prior

require = prior.require


def verify_branch(directory, role, row, events, history):
    arm = role["arm"]
    require(row["arm"] == arm and row["status"] in
            ("failed", "native_early_stop", "fixed_horizon_completed"), "branch status")
    require(events and events[0]["kind"] == "branch_started"
            and events[-1]["kind"] == "branch_closed"
            and events[-1]["status"] == row["status"], "branch boundaries")
    core = events[1:-1]
    errors = [e for e in core if e["kind"] == "branch_error"]
    require((len(errors) == 1 and row["error"] == errors[0]["error"]
             and row["status"] == "failed") or
            (not errors and row["error"] is None and row["status"] != "failed"), "failure record")
    if errors:
        require(core[-1]["kind"] in ("branch_error", "close_error"), "error ordering")
    core = [e for e in core if e["kind"] not in ("branch_error", "close_error")]
    references = {p["step"]: p for p in history["frames"]}
    plan = [("reset_verified", None, None, None), ("observation", "warmup", 0, None)]
    for phase, action_list in (("warmup", [[0.0, 0.0]] * 10),
                               ("prefix", role["prefix"]), ("tail", role["tail"])):
        if phase == "tail":
            plan.append(("cp5_gate_passed", None, None, None))
        for i, action in enumerate(action_list, 1):
            for kind in ("action_dispatch", "action_observed", "action_return"):
                plan.append((kind, phase, i, action))
            if phase != "warmup" or i in (5, 10):
                plan.append(("observation", phase, i, None))
    require(len(core) <= len(plan), "unexpected branch events")
    transitions, observations, pending, observed = [], [], None, None
    stopped = False
    for event, (kind, phase, step, action) in zip(core, plan):
        require(event["kind"] == kind and event["arm"] == arm, "event sequence")
        if phase is not None:
            require(event["phase"] == phase and event["step"] == step, "phase/step sequence")
        if kind == "action_dispatch":
            require(not stopped and event["action"] == action, "frozen dispatch")
            pending = event
        elif kind == "action_observed":
            require(pending is not None, "observed without dispatch")
            observed = event
        elif kind == "action_return":
            require(pending is not None and observed is not None and event["action"] == action
                    and all(observed[k] == event[k]
                            for k in ("action", "terminated", "truncated")), "actual action")
            transitions.append({k: event[k] for k in
                                ("phase", "step", "action", "state", "reward",
                                 "terminated", "truncated")})
            stopped = bool(event["terminated"] or event["truncated"])
            require(not stopped or phase == "tail" or row["status"] == "failed",
                    "unexpected native stop")
            pending = None
        elif kind == "observation":
            point = {k: v for k, v in event.items()
                     if k not in ("index", "previous", "kind", "arm", "sha256")}
            saved.frame(directory, point)
            if phase == "warmup":
                ref = references[step]
            elif phase == "prefix":
                ref = role["prefix_checkpoints"][step - 1]
            else:
                ref = None
            if ref is not None:
                require(all(point[k] == ref[k] for k in ("state", "snapshot", "pixel_sha256")),
                        "replay checkpoint")
            if step and phase in ("prefix", "tail"):
                require(transitions and transitions[-1]["state"] == point["state"],
                        "observation and returned state")
            observations.append(point)
        elif kind == "cp5_gate_passed":
            require(event["pixel_sha256"] == role["cp5_pixel_sha256"]
                    and len([p for p in observations if p["phase"] == "prefix"]) == 5,
                    "cp5 gate")
    require(transitions == row["transitions"] and observations == row["observations"],
            "row/journal consistency")
    if row["status"] == "fixed_horizon_completed":
        require(len(core) == len(plan) and not stopped, "complete horizon")
    elif row["status"] == "native_early_stop":
        require(stopped and transitions[-1]["phase"] == "tail"
                and core[-1]["kind"] == "observation"
                and core[-1]["step"] == transitions[-1]["step"], "early stop")
    return {"arm": arm, "status": row["status"],
            "tail_returned": sum(t["phase"] == "tail" for t in transitions)}


def verify(prepared, prepared_sha, directory, report_sha):
    from . import p1_dev303_cp5_tail_physical_20261102_exclusive as run

    prepared, directory = Path(prepared).resolve(strict=True), Path(directory).resolve(strict=True)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == run.build(),
            "contract and source closure")
    require(base.sha(directory / "report.json") == report_sha, "external report SHA")
    report = base.read(directory / "report.json")
    require(report["schema"] == run.NAME + ".report.v1"
            and report["prepare_sha256"] == prepared_sha
            and report["P1_admission"] is False and report["independent_confirmation"] is False
            and report["classification"] == "development_only"
            and report["status"] == ("partial_or_failed" if report["errors"] else "recorded"),
            "report boundary")
    require(base.read(directory / "execution_binding.json") ==
            {"prepared": str(prepared), "prepare_sha256": prepared_sha}, "execution binding")
    journal_path = directory / "journal.jsonl"
    require(base.sha(journal_path) == report["journal_sha256"], "journal SHA")
    events = saved.journal(journal_path, report["journal_tail"])
    require(events[0]["kind"] == "run_started" and events[0]["prepare_sha256"] == prepared_sha
            and events[-1]["kind"] == "run_finished" and events[-1]["errors"] == report["errors"],
            "journal endpoints")
    roles = base.read(prepared)["source"]["roles"]
    rows = report["rows"]
    require([r["arm"] for r in rows] == [r["arm"] for r in roles], "fixed arms")
    _, receipt, _ = prior.source_data()
    cursor = 1
    verified = []
    for role, row in zip(roles, rows, strict=True):
        require(base.read(directory / (role["arm"] + ".row.json")) == row, "row shard")
        end = cursor
        while end < len(events) - 1 and events[end]["kind"] != "branch_closed":
            end += 1
        require(end < len(events) - 1, "branch closure")
        segment = events[cursor:end + 1]
        require(all(e.get("arm") == role["arm"] for e in segment), "branch isolation")
        verified.append(verify_branch(directory, role, row, segment, receipt["history"]))
        cursor = end + 1
    require(cursor == len(events) - 1, "no unscoped events")
    logged_errors = [{"arm": r["arm"], **r["error"]} for r in rows if r["error"] is not None]
    require(all(item in report["errors"] for item in logged_errors), "reported branch errors")
    return {"status": "saved_chain_verified", "branches": verified,
            "run_status": report["status"], "errors": report["errors"],
            "P1_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("prepared", "prepared-sha", "directory", "report-sha"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.prepared, args.prepared_sha, args.directory, args.report_sha)))


if __name__ == "__main__":
    main()
