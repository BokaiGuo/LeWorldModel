"""Saved physical comparison consumer; never runs physics or a neural model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p3_preoutcome_winner_physical_20261102_codexexclusive as producer

moving = producer.moving
require = producer.require
VALID = ("horizon_completed", "native_success", "native_stopped")


def validate(plan, refs, report, events, directory):
    require(events and events[0]["kind"] == "run_started" and events[0]["prepare"] == plan,
            "embedded prepare")
    require(events[-1]["kind"] == "run_finished" and events[-1]["status"] == report["status"]
            and events[-1]["counts"] == report["counts"] and events[-1]["errors"] == report["errors"],
            "terminal binding")
    require(report["contract"] == plan["contract"] and len(report["rows"]) == len(plan["plans"]) == 4,
            "four fixed slots")
    attempts = Counter(e["resource"] for e in events if e["kind"] == "operation_attempted")
    completed = Counter(e["resource"] for e in events if e["kind"] == "operation_completed")
    require(set(attempts) <= plan["contract"]["caps"].keys()
            and set(completed) <= plan["contract"]["caps"].keys(), "unknown resource")
    for key, cap in plan["contract"]["caps"].items():
        require(report["counts"][key] == {"attempted": attempts[key], "completed": completed[key]}
                and completed[key] <= attempts[key] <= cap, "resource census")
    closures = [e for e in events if e["kind"] == "branch_closed"]
    require(len(closures) <= 4, "branch closure count")
    summaries = []
    previous_end = 0
    for i, (row, fixed) in enumerate(zip(report["rows"], plan["plans"], strict=True)):
        require((row["arm"], row["root_slot"], row["selection_digest"])
                == (fixed["arm"], fixed["root_slot"], fixed["selection_digest"]), "winner binding")
        if i >= len(closures):
            require(row["status"] == "not_started", "unclosed branch not verified")
            summaries.append({"arm": row["arm"], "root_slot": row["root_slot"], "status": "not_started",
                              "endpoint": "censored"})
            continue
        closed = closures[i]
        require(closed["receipt"] == row and closed["index"] > previous_end, "ordered branch closure")
        section = [e for e in events if previous_end < e["index"] < closed["index"]]
        previous_end = closed["index"]
        starts = [e for e in section if e["kind"] == "branch_started"]
        if not starts:
            require(row["status"] == "failed" and row["first_error"], "missing start must be failed")
            summaries.append({"arm": row["arm"], "root_slot": row["root_slot"], "status": "failed",
                              "endpoint": "censored", "failure": row["first_error"]})
            continue
        require(len(starts) == 1 and starts[0]["branch"] == i and starts[0]["plan"] == fixed,
                "branch start plan")
        scoped = [e for e in section if e.get("branch") == i]
        require(starts[0] in scoped, "branch scope")
        branch_attempts = Counter(e["resource"] for e in scoped if e["kind"] == "operation_attempted")
        branch_completed = Counter(e["resource"] for e in scoped if e["kind"] == "operation_completed")
        observations = [e["receipt"] for e in scoped if e["kind"] == "fresh_observation"]
        guards = [e for e in scoped if e["kind"] == "guard_receipt"]
        require(len(observations) == row["PNG"] == len(guards), "observation/guard count")
        for obs, guard in zip(observations, guards, strict=True):
            require((obs["phase"], obs["step"]) == (guard["phase"], guard["step"]), "guard order")
            path = directory / obs["png"]
            require(path.parent == directory and path.name == obs["png"] and not path.is_symlink()
                    and moving.sha(path) == obs["file_sha256"], "PNG byte SHA")
            with Image.open(path) as img:
                pixels = np.asarray(img)
            require(pixels.dtype == np.uint8 and pixels.shape == (224, 224, 3)
                    and moving.moving.image_sha(pixels) == obs["pixel_sha256"], "PNG pixel SHA")
        warm = [e for e in scoped if e["kind"] == "warmup_return"]
        require([e["step"] for e in warm] == list(range(1, len(warm)+1))
                and [e["raw_action"] for e in warm] == starts[0]["raw_warmup"][:len(warm)],
                "warmup action prefix")
        trans = [e for e in scoped if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        dispatched = [e for e in scoped if e["kind"] == "operation_attempted"
                      and e["resource"] == "public_step" and e.get("phase") == "plan"]
        require([e["step"] for e in trans] == list(range(1, row["own_returned"]+1))
                and [e["command"] for e in trans] == fixed["commands"][:len(trans)]
                and [e["step"] for e in dispatched] == list(range(1, row["own_attempted"]+1))
                and [e["raw_action"] for e in dispatched] == fixed["commands"][:len(dispatched)]
                and row["own_returned"] <= row["own_attempted"] <= 10, "physical command prefix")
        if row["initial_exact"]:
            ref = refs[row["root_slot"]]
            checkpoints = [o for o in observations if o["phase"] == "warmup" and o["step"] in (0, 5, 10)]
            require(len(warm) == 10 and len(checkpoints) == 3, "initial history")
            for j, obs in enumerate(checkpoints):
                require(obs["step"] == (0, 5, 10)[j]
                        and obs["pixel_sha256"] == moving.moving.image_sha(ref["history"][j])
                        and obs["state"] == ref["states"][j].tolist()
                        and obs["snapshot"] == ref["dynamics"][j], "initial RGB/state/snapshot")
            attached = [e for e in scoped if e["kind"] == "goal_attached"]
            require(len(attached) == 1 and attached[0]["goal"] == row["goal"] == ref["goal"].tolist()
                    and row["initial_metrics"] == moving.metric(row["goal"], row["root_state"])
                    and (not trans or attached[0]["index"] < trans[0]["index"]), "root/goal")
        require(not trans or row["initial_exact"], "action before exact root")
        require(not any(e["terminated"] or e["truncated"] for e in trans[:-1]), "continued after stop")
        if trans and (trans[-1]["terminated"] or trans[-1]["truncated"]):
            require(not any(e["index"] > trans[-1]["index"] for e in dispatched), "dispatch after stop")
        if trans:
            require(trans[-1]["state"] == row["final_state"]
                    and row["final_metrics"] == moving.metric(row["goal"], row["final_state"])
                    and (row["terminated"], row["truncated"])
                    == (trans[-1]["terminated"], trans[-1]["truncated"]), "terminal state/flags")
        if row["status"] in VALID:
            require(row["first_error"] is None and row["initial_exact"] and trans
                    and all(g["accepted"] and g["render_preserved_dynamics"] for g in guards), "valid row guards")
            if row["status"] == "horizon_completed":
                require(len(trans) == 10 and not row["terminated"] and not row["truncated"], "horizon completion")
            elif row["status"] == "native_success":
                require(row["terminated"] and row["native_success"] and not row["truncated"], "native success")
            else:
                require(row["terminated"] or row["truncated"], "native stop")
        else:
            require(row["status"] == "failed" and row["first_error"], "failed row retained")
        summaries.append({"arm": row["arm"], "root_slot": row["root_slot"], "status": row["status"],
                          "attempted": row["own_attempted"], "returned": row["own_returned"],
                          "endpoint": ("censored_native_stop" if row["status"] == "native_stopped"
                                       else row.get("endpoint_scope", "censored") if row["status"] in VALID
                                       else "censored"),
                          "failure": row["first_error"], "final_metrics": row.get("final_metrics"),
                          "physical_counts": {k: {"attempted": branch_attempts[k], "completed": branch_completed[k]}
                                              for k in plan["contract"]["caps"]}})
    statuses = [r["status"] for r in report["rows"]]
    require(report["native_successes"] == statuses.count("native_success"), "success census")
    require(report["status"] == ("moving_history_physical_recorded"
            if all(x in VALID for x in statuses) and not report["errors"] else "partial_or_failed"), "report status")
    return {"status": "saved_physical_records_verified" if report["status"] == "moving_history_physical_recorded"
            else "partial_saved_records_verified", "branches": summaries,
            "physical_counts": report["counts"], "physical_wall_seconds": report["wall_seconds"],
            "model_accounting": plan["model_accounting"], "errors": report["errors"],
            "independent_physics_attestation": False, "independent_outcome": False,
            "equal_total_compute": False, "P3_admission": False, "C3_admission": False}


def consume(prepared, prepare_sha, directory, report_sha):
    plan, data = producer.verify(prepared, prepare_sha)
    directory = Path(directory)
    require(directory.is_absolute() and directory.parent == producer.ROOT / "results"
            and directory.name.startswith(producer.NAME + "_RUN_") and directory.is_dir()
            and not directory.is_symlink(), "run path")
    require(moving.read(directory / "execution_binding.json")
            == {"prepared": str(Path(prepared).resolve()), "prepare_sha256": prepare_sha}, "execution binding")
    require(moving.sha(directory / "report.json") == report_sha, "report SHA")
    report = moving.read(directory / "report.json")
    require(moving.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal SHA")
    events = moving.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    result = validate(plan, data["references"], report, events, directory)
    result.update(prepare_sha256=prepare_sha, report_sha256=report_sha, journal_sha256=report["journal_sha256"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--prepare-sha", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--report-sha", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = consume(a.prepared, a.prepare_sha, a.run, a.report_sha)
    output = producer.output(a.output, "_CONSUME_")
    print(json.dumps({"status": result["status"], "path": str(output),
                      "sha256": producer.producer.host.write_new(output, result)}))


if __name__ == "__main__":
    main()
