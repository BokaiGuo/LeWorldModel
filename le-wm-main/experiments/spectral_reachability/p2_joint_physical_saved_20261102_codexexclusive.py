"""Consume fixed-winner physical records, including genuine stop censoring."""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import p2_joint_physical_20261102_codexexclusive as run

physical = run.physical
RESULT = run.SESSION / (physical.NAME + "_JOINT_01")
RECEIPT = run.RESULTS / (run.NAME + "_CONSUME_01.json")


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def check_stop(row, transitions, warm):
    stopped = (transitions and (transitions[-1]["terminated"] or transitions[-1]["truncated"])) or (
        warm and (warm[-1]["terminated"] or warm[-1]["truncated"])) or (
        row.get("stop_reason") in ("warmup already terminal", "original root already task-terminal"))
    need(bool(stopped) and row["own_returned"] <= 25, "uncorroborated censoring")


def audit_events(events, report, p, digest, directory):
    need(events and events[0]["kind"] == "joint_started" and events[0]["plan_sha256"] == digest,
         "start identity")
    need(events[-1]["kind"] == "joint_finished" and
         all(events[-1][k] == report[k] for k in ("status", "errors", "counts")), "final event")
    need(report["status"] == "recorded_development_unconsumed" and report["errors"] == []
         and len(report["rows"]) == 2 and report["consumer_verified"] is False
         and report["new_neural_calls"] == 0 and all(report[k] is False for k in
         ("historical_sra_authenticated", "independent_confirmation", "equal_total_compute", "P1_admission", "P2_admission"))
         and type(report["wall_seconds"]) in (int, float) and math.isfinite(report["wall_seconds"])
         and report["wall_seconds"] >= 0, "report status/flags")
    allowed = {"joint_started", "joint_finished", "branch_closed", "operation_dispatch_intent",
               "operation_returned", "placement_recipe", "goal_attached", "native_history_guard",
               "feedback_prefix_guard", "fresh_observation", "returned_transition", "own_response"}
    tail, active, stack, intents, returns = None, 0, [], [], {}
    for index, event in enumerate(events):
        need(event["index"] == index and event["previous_sha256"] == tail
             and event["event_sha256"] == physical.digest({k: v for k, v in event.items() if k != "event_sha256"})
             and event["kind"] in allowed, "event chain")
        tail = event["event_sha256"]
        kind = event["kind"]
        if kind in ("joint_started", "joint_finished"):
            need(index == (0 if kind == "joint_started" else len(events)-1), "bookend")
            continue
        need(active < 2, "branch order")
        row = report["rows"][active]
        if kind == "branch_closed":
            need(not stack and event["receipt"] == row, "branch close")
            active += 1
            continue
        scope = event["receipt"] if kind.startswith("operation_") else event
        need(scope.get("branch") == active and all(scope.get(k) == row[k] for k in
             ("root_slot", "condition", "arm", "selection_digest")), "event branch scope")
        if kind == "operation_dispatch_intent":
            op = event["receipt"]
            need(op["index"] == len(intents) and op["status"] == "pending"
                 and op["resource"] in physical.contract()["caps"], "operation intent")
            intents.append(op)
            stack.append(op["index"])
        elif kind == "operation_returned":
            op = event["receipt"]
            need(stack and stack.pop() == op["index"] and op["index"] not in returns
                 and op["status"] == "returned" and dict(op, status="pending") == intents[op["index"]],
                 "operation return")
            returns[op["index"]] = op
    need(active == 2 and not stack and len(returns) == len(intents), "unfinished execution")
    ops = [returns[i] for i in range(len(returns))]
    need(ops == report["operation_receipts"], "operation receipts")
    count = Counter(o["resource"] for o in ops)
    need(report["counts"] == {k: {"attempted": count[k], "completed": count[k]}
                              for k in physical.contract()["caps"]}
         and all(count[k] <= limit for k, limit in physical.contract()["caps"].items()), "caps")
    seen = set()
    for event in events:
        if event["kind"] != "fresh_observation":
            continue
        rec = event["receipt"]
        name = rec["png"]
        need(isinstance(name, str) and Path(name).name == name and name not in seen, "PNG name")
        path = directory / name
        need(path.is_file() and not path.is_symlink(), "PNG file")
        raw = path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == rec["file_sha256"], "PNG SHA")
        with Image.open(io.BytesIO(raw)) as im:
            need(im.format == "PNG" and im.mode == "RGB" and im.size == (224, 224), "PNG RGB224")
            pixel = np.asarray(im).copy()
        need(physical.moving.image_sha(pixel) == rec["pixel_sha256"]
             and event["pre"] == event["post"] and rec["snapshot"] == event["post"]["snapshot"], "RGB binding")
        seen.add(name)
    for branch, (row, plan) in enumerate(zip(report["rows"], p["plans"], strict=True)):
        need(row["index"] == branch and row["plan"] == plan and row["policy"] == plan["policy"]
             and row["selection_digest"] == plan["selection_digest"] and row["first_error"] is None
             and row["secondary_errors"] == [] and row["status"] in ("native_stopped", "horizon_completed"),
             "branch result")
        scoped = [e for e in events if e.get("branch", e.get("receipt", {}).get("branch")) == branch]
        physical.audit_feedback(row, plan, scoped, p)
        physical.audit_labels(row, plan, scoped, p)
        observations = [e for e in scoped if e["kind"] == "fresh_observation"]
        trajectory = [e["receipt"] for e in observations if e["receipt"]["overlay_index"] == 0]
        need(row["PNG"] == len(observations) and row["trajectory"] == trajectory,
             "RGB trajectory census")
        checkpoints = []
        for step in physical.CP:
            found = [x for x in trajectory if x["phase"] == "plan" and x["step"] == step]
            if found:
                checkpoints.append({k: found[-1][k] for k in
                                    ("step", "png", "file_sha256", "pixel_sha256", "state", "snapshot")})
        need(row["checkpoints"] == checkpoints and row["missing_checkpoints"] ==
             [step for step in physical.CP if step not in [c["step"] for c in checkpoints]],
             "checkpoint censoring")
        for phase, label, actions in (("warmup", "warmup", physical.moving.warmup_commands()),
                                      ("plan", "own", physical.raw25(plan["commands"]))):
            dispatch = [e for e in scoped if e["kind"] == "operation_dispatch_intent"
                        and e["receipt"]["resource"] == "public_step" and e["receipt"]["phase"] == phase]
            transitions = [e for e in scoped if e["kind"] == "returned_transition" and e["phase"] == phase]
            need(len(dispatch) == len(transitions) == row[label + "_attempted"] == row[label + "_returned"]
                 and len(transitions) <= len(actions), "action census")
            for step, (intent, transition, action) in enumerate(zip(dispatch, transitions, actions), 1):
                bits = action.tobytes().hex()
                need(intent["receipt"]["step"] == transition["step"] == step and
                     intent["receipt"]["raw_bits"] == transition["requested_bits"] == transition["actual_bits"] == bits
                     and transition["actual_dtype"] == "<f4" and intent["index"] < transition["index"],
                     "literal action binding")
                if step < len(transitions):
                    need(not transition["terminated"] and not transition["truncated"], "post-stop padding")
        transitions = [e for e in scoped if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        responses = [e for e in scoped if e["kind"] == "own_response"]
        need(len(responses) == len(transitions) and [e["response"] for e in responses] == row["responses"],
             "response census")
        for t, e in zip(transitions, responses, strict=True):
            r = e["response"]
            calc = physical.native.metrics(row["goal"], r["state"])
            need(t["index"] < e["index"] and all(r[k] == t[k] for k in
                 ("step", "state", "reward", "terminated", "truncated"))
                 and r["native_metrics"] == calc["native"] and r["reward"] == calc["native7_reward"]
                 and r["native_success"] == calc["native_success"], "native response")
        if responses:
            need(row["final_state"] == responses[-1]["response"]["state"]
                 and row["final_metrics"] == responses[-1]["response"]["native_metrics"], "final metric")
        if row["status"] == "horizon_completed":
            need(row["initial_exact"] and row["warmup_returned"] == 10 and row["own_returned"] == 25
                 and row["feedback_guard"] is not None and not row["terminated"] and not row["truncated"], "horizon")
        else:
            warm = [e for e in scoped if e["kind"] == "returned_transition" and e["phase"] == "warmup"]
            check_stop(row, transitions, warm)
            if row.get("stop_reason") in ("warmup already terminal", "original root already task-terminal"):
                phase = "warmup" if row["stop_reason"] == "warmup already terminal" else "plan"
                observed = [e for e in observations if e["receipt"]["phase"] == phase]
                need(observed and (phase != "plan" or row["own_returned"] == 0), "terminal observation")
                view = observed[-1]["post"]
                need(physical.native.metrics(view["goal_state"], view["snapshot"]["state"])["native_success"]
                     is True, "pre-action terminal evidence")
    names = {"binding.json", "journal.jsonl", "report.json"} | seen
    need(all(x.is_file() and not x.is_symlink() for x in directory.iterdir())
         and {x.name for x in directory.iterdir()} == names, "artifact inventory")
    return {"journal_tail_sha256": tail, "events": len(events), "png_verified": len(seen),
            "branches": [{"policy": r["policy"], "status": r["status"],
                           "warmup_returned": r["warmup_returned"], "plan_returned": r["own_returned"],
                           "missing_checkpoints": r["missing_checkpoints"], "final_metrics": r["final_metrics"],
                           "native_success": r["native_success"], "terminated": r["terminated"],
                           "truncated": r["truncated"]} for r in report["rows"]]}


def consume(digest, report_sha):
    plan = run.verify(digest)
    need(RESULT.is_dir() and not RESULT.is_symlink(), "result directory")
    binding = json.loads((RESULT / "binding.json").read_bytes())
    need(binding == {"plan_sha256": digest, "old_plan_sha256": run.prior.old.PREP_SHA}, "binding")
    report = json.loads(run.prior.checked(RESULT / "report.json", report_sha))
    need(report["schema"] == run.NAME + ".report_v1" and report["plan_sha256"] == digest
         and run.host.base.file_sha(RESULT / "journal.jsonl") == report["journal_sha256"], "report identity")
    raw = (RESULT / "journal.jsonl").read_bytes()
    need(raw.endswith(b"\n") and b"\n\n" not in raw and b"\r" not in raw, "journal framing")
    events = [physical.native.decode(line) for line in raw.splitlines()]
    original = json.loads(run.prior.checked(run.prior.old.PREP, run.prior.old.PREP_SHA))
    summary = audit_events(events, report, run.derived(original, plan), digest, RESULT)
    need(run.host.base.file_sha(RESULT / "report.json") == report_sha
         and run.host.base.file_sha(RESULT / "journal.jsonl") == report["journal_sha256"], "postcheck")
    return {"schema": run.NAME + ".consume_v1", "status": "verified_saved_known_root_development",
            "plan_sha256": digest, "report_sha256": report_sha, "journal_sha256": report["journal_sha256"],
            "summary": summary, "historical_sra_authenticated": False, "equal_total_compute": False,
            "independent_confirmation": False, "P1_admission": False, "P2_admission": False,
            "limitations": ["saved bytes do not prove physical execution independently",
                            "one exposed B0 root", "no external producer completion certificate",
                            "source manifest not full transitive closure"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--report-sha256", required=True)
    a = p.parse_args()
    need(not RECEIPT.exists() and not RECEIPT.is_symlink(), "exclusive receipt")
    result = consume(a.plan_sha256, a.report_sha256)
    print(json.dumps({"receipt": str(RECEIPT), "sha256": run.host.base.write_new(RECEIPT, result)}))


if __name__ == "__main__":
    main()
