"""Read-only consumer for the actual tmpfix finalization contract.

This is a development-record audit, not the old B0 runner's completion
certificate, an independent experiment, or a scientific admission.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import p2_conditional_physical_saved_consumer_20261102_exclusive as prior
from . import p2_conditional_physical_executor_tmpfix_20261102_exclusive as executor

physical = executor.physical
RESULT = prior.RESULT
REPORT_SHA = prior.REPORT_SHA
JOURNAL_SHA = prior.JOURNAL_SHA
PLAN_SHA = prior.PLAN_SHA


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_lifecycle(report, events, plans, prepared):
    require(report["status"] == "recorded_development_unconsumed" and report["errors"] == [],
            "report status/errors")
    require(report["consumer_verified"] is False and report["new_neural_calls"] == 0 and
            all(report[k] is False for k in
                ("independent_confirmation", "equal_total_compute", "P1_admission", "P2_admission")),
            "report claim flags")
    require(type(report["wall_seconds"]) in (int, float) and
            math.isfinite(report["wall_seconds"]) and 0 <= report["wall_seconds"],
            "report wall time")
    require(events[0]["kind"] == "conditional_started" and
            events[0]["recipe_sha256"] == prior.RECIPE_SHA, "start binding")
    end = events[-1]
    require(end["kind"] == "conditional_finished" and
            all(end[k] == report[k] for k in ("status", "errors", "counts")),
            "final event/report mismatch")
    require(len(report["rows"]) == len(plans) == 2, "fixed paired rows")
    allowed = {"conditional_started", "conditional_finished", "branch_closed",
               "operation_dispatch_intent", "operation_returned", "placement_recipe",
               "goal_attached", "native_history_guard", "feedback_prefix_guard",
               "fresh_observation", "returned_transition", "own_response"}
    active, stack, intents, returns, tail = 0, [], [], {}, None
    for index, event in enumerate(events):
        require(type(event["index"]) is int and event["index"] == index and
                event["previous_sha256"] == tail and event["kind"] in allowed and
                event["event_sha256"] == prior.digest(
                    {k: v for k, v in event.items() if k != "event_sha256"}),
                f"journal chain/event {index}")
        tail = event["event_sha256"]
        kind = event["kind"]
        if kind in ("conditional_started", "conditional_finished"):
            require((index == 0 if kind == "conditional_started" else index == len(events)-1),
                    "lifecycle bookend")
            continue
        require(active < 2, "event after both branches closed")
        row = report["rows"][active]
        if kind == "branch_closed":
            require(not stack and event["receipt"] == row, "branch closure/receipt")
            active += 1
            continue
        scope = event["receipt"] if kind.startswith("operation_") else event
        require(scope.get("branch") == active and
                all(scope.get(k) == row[k] for k in
                    ("root_slot", "condition", "arm", "selection_digest")),
                f"active branch scope {index}")
        if kind == "operation_dispatch_intent":
            op = event["receipt"]
            require(op["index"] == len(intents) and op["status"] == "pending" and
                    op["resource"] in physical.contract()["caps"], "operation intent")
            intents.append(op)
            stack.append(op["index"])
        elif kind == "operation_returned":
            op = event["receipt"]
            require(stack and stack.pop() == op["index"] and op["index"] not in returns and
                    op["status"] == "returned" and
                    dict(op, status="pending") == intents[op["index"]],
                    "operation return and nesting")
            returns[op["index"]] = op
    require(active == 2 and not stack and len(returns) == len(intents), "unfinished operations")
    ops = [returns[i] for i in range(len(returns))]
    require(ops == report["operation_receipts"], "operation receipts/report")
    counts = {k: {"attempted": 0, "completed": 0} for k in physical.contract()["caps"]}
    for op in ops:
        counts[op["resource"]]["attempted"] += 1
        counts[op["resource"]]["completed"] += 1
    require(counts == report["counts"] and
            all(v["attempted"] <= physical.contract()["caps"][k] for k, v in counts.items()),
            "operation counts/caps")
    for branch, (row, plan) in enumerate(zip(report["rows"], plans, strict=True)):
        require(row["index"] == branch and row["plan"] == plan and
                row["policy"] == plan["policy"] and row["selection_digest"] == plan["selection_digest"] and
                row["status"] == "horizon_completed" and row["initial_exact"] is True and
                row["first_error"] is None and row["secondary_errors"] == [],
                "completed exact bound row")
        scoped = [e for e in events if e.get("branch", e.get("receipt", {}).get("branch")) == branch]
        # Reuse the historical pure ABI checks, not its old-witness validator:
        # the sealed old preparation remains unchanged and plans come from the
        # separately rebuilt new-witness recipe.
        physical.audit_feedback(row, plan, scoped, prepared)
        physical.audit_labels(row, plan, scoped, prepared)
        for phase, label, actions in (
                ("warmup", "warmup", physical.moving.warmup_commands()),
                ("plan", "own", physical.raw25(plan["commands"]))):
            dispatch = [e for e in scoped if e["kind"] == "operation_dispatch_intent"
                        and e["receipt"]["resource"] == "public_step"
                        and e["receipt"]["phase"] == phase]
            transitions = [e for e in scoped if e["kind"] == "returned_transition" and e["phase"] == phase]
            require(len(dispatch) == len(transitions) == len(actions) ==
                    row[label + "_attempted"] == row[label + "_returned"], "action count")
            for step, (intent, transition, action) in enumerate(zip(dispatch, transitions, actions, strict=True), 1):
                bits = action.tobytes().hex()
                require(intent["receipt"]["step"] == transition["step"] == step and
                        intent["receipt"]["raw_bits"] == transition["requested_bits"] ==
                        transition["actual_bits"] == bits and transition["actual_dtype"] == "<f4" and
                        intent["index"] < transition["index"] and
                        transition["terminated"] is False and transition["truncated"] is False,
                        f"{phase} literal action/termination/chronology")
        responses = [e for e in scoped if e["kind"] == "own_response"]
        plan_returns = [e for e in scoped if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        require(len(responses) == 25 and [e["response"] for e in responses] == row["responses"],
                "response census")
        for transition, event in zip(plan_returns, responses, strict=True):
            response = event["response"]
            require(transition["index"] < event["index"] and all(
                response[k] == transition[k] for k in ("step", "state", "reward", "terminated", "truncated")),
                "response tied to actual transition")
            calc = physical.native.metrics(row["goal"], response["state"])
            require(response["native_metrics"] == calc["native"] and response["reward"] == calc["native7_reward"]
                    and response["native_success"] == calc["native_success"], "native metric recalculation")
        require(row["final_state"] == responses[-1]["response"]["state"] and
                row["final_metrics"] == responses[-1]["response"]["native_metrics"], "final labels")
    return tail


def audit():
    # Existing consumer performs pinned report/journal hashes, PNG byte/pixel
    # reads, recipe rebuild; its completion inference is explicitly discarded.
    base = prior.audit()
    executor.verify(PLAN_SHA)
    report = prior.load_json(RESULT / "report.json")
    raw = (RESULT / "journal.jsonl").read_bytes()
    require(raw.endswith(b"\n") and b"\n\n" not in raw and b"\r" not in raw, "journal framing")
    events = [physical.native.decode(line) for line in raw.splitlines()]
    original = json.loads(executor.handoff.checked(executor.handoff.old.PREP, prior.OLD_PLAN_SHA))
    new_recipe = executor._checked_recipe()
    derived = executor._derived(original, new_recipe)
    tail = check_lifecycle(report, events, new_recipe["plans"], derived)
    names = {"binding.json", "journal.jsonl", "report.json"}
    for event in events:
        if event["kind"] == "fresh_observation":
            names.add(event["receipt"]["png"])
    files = list(RESULT.iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in files) and
            {p.name for p in files} == names and len(names) == 225, "saved artifact inventory")
    require(prior.sha(RESULT / "report.json") == REPORT_SHA and
            hashlib.sha256((RESULT / "journal.jsonl").read_bytes()).hexdigest() == JOURNAL_SHA,
            "final source unchanged")
    return {
        "schema": "p2_conditional_tmpfix_finalization_consumer_v1",
        "status": "tmpfix_saved_record_verified_known_root_not_admitted",
        "report_sha256": REPORT_SHA, "journal_sha256": JOURNAL_SHA,
        "journal_chain_tail": tail, "journal_events": len(events),
        "png_bytes_and_pixels_checked": base["png_verified"],
        "operation_receipts_checked": len(report["operation_receipts"]),
        "per_arm_warmup_steps": 10, "per_arm_plan_steps": 25,
        "producer_finalization": "conditional_finished followed by report.json",
        "separate_completion_contract": "not present in tmpfix producer contract",
        "separate_completion_file_present": (RESULT / "completion.json").exists(),
        "external_report_sha_source": "previously published tmpfix physical result document",
        "limitations": [
            "no separate producer completion certificate or independently measured post-report wall bound",
            "saved evidence consistency only; cannot prove real environment execution from bytes alone",
            "reuses historical pure validation helpers and upstream recipe reconstruction",
            "one previously exposed development root; unequal total model-side costs",
        ],
        "independent_confirmation": False, "equal_total_compute": False,
        "P1_admission": False, "P2_admission": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(audit(), sort_keys=True))


if __name__ == "__main__":
    main()
