"""Saved-only prefix-null audit; never opens the scientific gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from . import codex_p1_prefix_null_runner_20261030_v1 as runner
from . import p1_backtrack_physical_consume_20261030 as previous

base = runner.base
need = base.require
AXES = base.contract.contract()["caps"]


def account(report, journal, prepared):
    """Check complete saved intent/result bookkeeping, including failures."""
    rows = report["rows"]
    need(len(rows) == len(prepared["plans"]) == 16 and report["fixed_rows"] == 16,
         "fixed denominator")
    for i, (row, plan) in enumerate(zip(rows, prepared["plans"], strict=True)):
        need(row["index"] == i and row["plan"] == plan and
             row["status"] in ("not_started", "engineering_failed", "native_stopped", "horizon_completed"),
             "row identity or status")
    intents, endings = {}, {}
    for event in journal:
        kind = event["kind"]
        if kind not in ("operation_dispatch_intent", "operation_returned", "operation_failed"):
            continue
        receipt = event["receipt"]
        idx = receipt["index"]
        need(type(idx) is int and idx >= 0 and receipt["resource"] in AXES, "operation identity")
        if kind == "operation_dispatch_intent":
            need(idx not in intents and receipt["status"] == "pending" and idx == len(intents),
                 "duplicate or unordered intent")
            intents[idx] = receipt
        else:
            need(idx in intents and idx not in endings and
                 receipt["status"] == ("returned" if kind == "operation_returned" else "failed"),
                 "orphan or duplicate operation end")
            clean = {k: v for k, v in receipt.items() if k not in ("status", "error")}
            need(clean == {k: v for k, v in intents[idx].items() if k != "status"},
                 "operation identity drift")
            endings[idx] = receipt
    need(set(intents) == set(endings), "unresolved attempt")
    need([endings[i] for i in range(len(intents))] == report["receipts"], "receipt reconciliation")
    expected = {k: {"attempted": 0, "completed": 0} for k in AXES}
    for receipt in endings.values():
        counts = expected[receipt["resource"]]
        counts["attempted"] += 1
        counts["completed"] += receipt["status"] == "returned"
    need(expected == report["counts"] and all(expected[k]["attempted"] <= AXES[k] for k in AXES),
         "operation counts or caps")
    return expected


def observed_checkpoints(row, branch, source_root, directory=None):
    """Bind saved state and RGB to journal/source; report only observed checkpoints."""
    frames = [e for e in branch if e["kind"] == "fresh_observation"]
    transitions = [e for e in branch if e["kind"] == "returned_transition" and e["phase"] == "plan"]
    expected_frames = [(phase, step, overlay)
                       for phase, steps in (("warmup", range(11)), ("plan", range(len(transitions) + 1)))
                       for step in steps for overlay in (0, 1, 2)]
    need([(e["receipt"]["phase"], e["receipt"]["step"], e["receipt"]["overlay_index"])
          for e in frames] == expected_frames, "frame triplet coverage")
    for e in frames:
        f = e["receipt"]
        need(e["pre"] == e["post"] and f["snapshot"] == e["post"]["snapshot"] and
             f["state"] == f["snapshot"]["state"], "frame/snapshot mismatch")
        if f["phase"] == "warmup":
            reference = base.native.reference_for(source_root["warm_trace"], f["step"], f["overlay_index"])
            if reference is not None:
                base.native.check_warm_frame({"receipt": f, "pre": e["pre"], "post": e["post"]}, reference)
    selected_overlay = 1 if row["condition"] == "on" else 0
    own_frames = [e["receipt"] for e in frames if e["receipt"]["phase"] == "plan" and
                  e["receipt"]["overlay_index"] == selected_overlay]
    need(own_frames == [f for f in row["trajectory"] if f["phase"] == "plan"],
         "trajectory is not source journal frames")
    need(own_frames[0]["state"] == row["root_state"], "initial frame/root mismatch")
    if len(transitions) >= 6:
        guards = [e for e in branch if e["kind"] == "cp5_guard_verified"]
        need(len(guards) == 1 and directory is not None, "saved cp5 guard missing")
        five = [e["receipt"] for e in frames if e["receipt"]["phase"] == "plan" and
                e["receipt"]["step"] == 5]
        need(five[0]["pixel_sha256"] == five[2]["pixel_sha256"] and
             all(x["state"] == five[0]["state"] and x["snapshot"] == five[0]["snapshot"] for x in five),
             "cp5 triplet identity")
        prefix = row["plan"]["raw15_float32_hex"][:80]
        warm = base.native.moving.warmup_commands().astype("<f4", copy=False).tobytes().hex()
        payload = base.guard.identity(task_id=row["task_id"], condition=row["condition"],
                                      prefix_hex=prefix, state=five[0]["state"], snapshot=five[0]["snapshot"],
                                      pixels={"off": five[0]["pixel_sha256"], "on": five[1]["pixel_sha256"]},
                                      history_hex=warm + prefix)
        key = hashlib.sha256((row["task_id"] + row["condition"] + prefix).encode()).hexdigest()
        path = directory / ("cp5_" + key + ".json")
        need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024 and
             path.read_bytes() == payload and guards[0]["reference_key"] == key and
             guards[0]["identity_sha256"] == hashlib.sha256(payload).hexdigest(), "cp5 guard file mismatch")
    for step, transition in enumerate(transitions, 1):
        frame = own_frames[step]
        need(frame["step"] == step and frame["state"] == transition["state"] and
             frame["snapshot"] == transition["snapshot"], "transition/frame state mismatch")
        response = row["responses"][step - 1]
        need(type(response["terminated"]) is bool and type(response["truncated"]) is bool and
             response["terminated"] == transition["terminated"] and
             response["truncated"] == transition["truncated"], "terminal flags mismatch")
        if step < len(transitions):
            need(not response["terminated"] and not response["truncated"], "dispatch after native stop")
    if row["status"] == "native_stopped":
        need(transitions and (transitions[-1]["terminated"] or transitions[-1]["truncated"]),
             "unproven native stop")
    else:
        need(len(transitions) == 15 and not transitions[-1]["terminated"] and
             not transitions[-1]["truncated"], "horizon status mismatch")
    result = []
    for cp in row["checkpoints"]:
        step = cp["step"]
        f = own_frames[step]
        need(all(cp[k] == f[k] for k in ("step", "png", "file_sha256", "pixel_sha256", "state", "snapshot")),
             "checkpoint is not observed frame")
        state = np.asarray(cp["state"], dtype=np.float64)
        goal = np.asarray(row["goal"], dtype=np.float64)
        need(state.shape == goal.shape == (7,) and np.isfinite(state).all() and np.isfinite(goal).all(),
             "invalid state or goal")
        bodies = cp["snapshot"]["bodies"]
        need(np.array_equal(np.asarray(bodies["agent"]["position"]), state[:2]) and
             np.array_equal(np.asarray(bodies["block"]["position"]), state[2:4]) and
             np.array_equal(np.asarray(bodies["agent"]["velocity"]), state[5:7]), "snapshot state mismatch")
        delta = state[4] - goal[4]
        result.append({"step": step, "object_position_error": float(np.linalg.norm(state[2:4] - goal[2:4])),
                       "wrapped_angle_error": abs(math.atan2(math.sin(delta), math.cos(delta))),
                       "pusher_position": state[:2].tolist(), "pusher_velocity": state[5:7].tolist(),
                       "block_velocity": bodies["block"]["velocity"],
                       "block_angular_velocity": bodies["block"]["angular_velocity"]})
    return result


def check(directory, report, prepared, data):
    need(report["schema"] == base.NAME + ".report" and report["contract"] == prepared["contract"] and
         report["P1_admission"] is False and report["independent_confirmation"] is False and
         report["new_neural_calls"] == 0, "report contract")
    need(base.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal digest")
    journal = previous.events(directory / "journal.jsonl", report["journal_chain_tail"])
    need(journal[0]["kind"] == "run_started" and
         journal[0]["binding"] == report["execution_binding"] and
         journal[0]["contract"] == prepared["contract"] and
         journal[-1]["status"] == report["status"] and journal[-1]["counts"] == report["counts"] and
         journal[-1]["errors"] == report["errors"],
         "terminal journal binding")
    counts = account(report, journal, prepared)
    closed = [e["receipt"] for e in journal if e["kind"] == "branch_closed"]
    started = [r for r in report["rows"] if r["status"] != "not_started"]
    need(closed == started and
         all(r["trajectory"] == [] and r["responses"] == [] and
             r["missing_checkpoints"] == [0, 5, 10, 15] for r in report["rows"]
             if r["status"] == "not_started"), "branch closure or unstarted padding")
    summaries = [{"index": r["index"], "status": r["status"],
                  "first_error": r["first_error"], "missing_checkpoints": r["missing_checkpoints"]}
                 for r in report["rows"]]
    if report["status"] != "physical_recorded_not_independently_consumed":
        return {"status": "blocked_incomplete", "rows": summaries, "counts": counts,
                "P1_admission": False, "independent_confirmation": False}
    previous.check(directory, report, prepared)
    checkpoints = []
    for row in report["rows"]:
        branch = [e for e in journal if e.get("branch") == row["index"]]
        task = prepared["task_rows"][row["plan"]["slot"]]
        need(row["root_sha256"] == task["native_root_sha256"] and
             row["root_state"] == task["native_root_state"] and row["goal"] == task["native_goal"] and
             row["goal_pixel_sha256"] == task["goal_pixel_sha256"], "root or goal source mismatch")
        for e in branch:
            if e["kind"] == "fresh_observation":
                frame = e["receipt"]
                path = base.native.paired.flat(directory, frame["png"])
                with Image.open(path) as image:
                    pixels = np.asarray(image.convert("RGB"))
                base.native.moving.guard(pixels, np.asarray(frame["state"]))
        checkpoints.append({"index": row["index"],
                            "observed": observed_checkpoints(row, branch, data["roots"][row["root_slot"]], directory),
                            "missing": row["missing_checkpoints"]})
    return {"status": "saved_checks_passed_exploratory_only", "rows": summaries,
            "checkpoints": checkpoints, "counts": counts, "P1_admission": False,
            "independent_confirmation": False, "freshness": False, "equal_budget_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-sha", required=True)
    args = parser.parse_args()
    try:
        directory = Path(args.output)
        need(directory.is_absolute() and directory.parent == runner.roster.ROOT / "results" and
             directory.name.startswith(runner.NAME + "_run_") and directory.is_dir() and
             not directory.is_symlink(), "run directory")
        base.native.actual_hash(args.report_sha)
        path = directory / "report.json"
        need(path.is_file() and not path.is_symlink() and base.sha(path) == args.report_sha, "report SHA")
        report = base.native.read(path, cap=64 * 1024**2)
        binding = report["execution_binding"]
        prepared, data = runner.verify(binding["prepared"], binding["prepare_sha256"])
        need(binding == base.native.read(directory / "execution_binding.json") and
             binding["join_sha256"] == runner.roster.JOIN_SHA and base.sha(path) == args.report_sha,
             "execution binding")
        result = check(directory, report, prepared, data)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "saved_checks_passed_exploratory_only" else 3
    except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        print(json.dumps({"status": "blocked_invalid", "reason": str(exc), "P1_admission": False}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
