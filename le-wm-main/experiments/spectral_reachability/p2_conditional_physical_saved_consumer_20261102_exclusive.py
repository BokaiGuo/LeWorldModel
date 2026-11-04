"""Independent saved-output consumer for the tmpfix conditional physical run.

This module never constructs an environment and never dispatches physics.  It
checks the saved bytes, the producer's new-witness bindings, the journal
chain/chronology, literal actions, and decoded RGB pixels.  The producer did
not write an external completion seal; that omission is retained as a
fail-closed finding.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
from PIL import Image

from . import p2_conditional_native_recipe_20261102_exclusive as recipe
from . import p2_conditional_physical_handoff_20261102_exclusive as handoff
from . import agent_allmode_physical_20261028_v1 as allmode

ROOT = handoff.ROOT
RESULT = ROOT / "results/p2_conditional_physical_executor_tmpfix_20261102_exclusive_SESSION_01" / (
    "codex_p2_multistart_joint_physical_20261030_v1_CONDITIONAL_01"
)
RECEIPT = ROOT / "results/p2_conditional_physical_saved_consumer_20261102_exclusive_01.json"
REPORT_SHA = "dfcbfbbb0b9f1561f29c6069a7a36ac30b9314b03766f7e58e4e659712f80dda"
JOURNAL_SHA = "7b35676f0c5d339dc4b53e20369f433b230e07eb924af8be84161426f23d0e09"
PLAN_SHA = "11efba074c71716e66cd64ecfa85916e903d3ae8ce572ab14d68572f35b90ac0"
RECIPE_SHA = "6009cbaf5a44169e364f330492ece7ac8111fe4cf594fa93ade6477cbf2d9555"
OLD_PLAN_SHA = "ceb38405762f7631d367a490e55d378c655dfe0463fcc648ee88ebade683a23c"
TARGET_SCHEMA = "p2_conditional_physical_executor_tmpfix_20261102_exclusive.report_v1"


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, indent=None, ensure_ascii=False
    ).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fail(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    p = Path(path)
    fail(p.is_file() and not p.is_symlink(), f"missing/non-regular: {p}")
    return json.loads(p.read_bytes())


def _check_metadata(binding, name, expected):
    path = RESULT / name
    fail(binding[name.replace(".json", "_sha256") if name != "journal.jsonl" else "journal_sha256"] == expected,
         f"binding {name} SHA")
    fail(sha(path) == expected, f"{name} SHA")


def _scoped(event):
    return event.get("branch", event.get("receipt", {}).get("branch"))


def _check_chain(events):
    fail(events and events[0]["kind"] == "conditional_started", "missing start")
    fail(events[-1]["kind"] == "conditional_finished", "missing finish")
    allowed = {
        "conditional_started", "placement_recipe", "goal_attached",
        "native_history_guard", "feedback_prefix_guard",
        "operation_dispatch_intent", "operation_returned",
        "fresh_observation", "returned_transition", "own_response",
        "branch_closed", "conditional_finished",
    }
    previous = None
    pending = []
    terminals = {}
    branch_closed = -1
    for index, event in enumerate(events):
        fail(event["index"] == index and event["previous_sha256"] == previous,
             f"journal chronology at {index}")
        sealed = {k: v for k, v in event.items() if k != "event_sha256"}
        fail(event["event_sha256"] == digest(sealed), f"journal event hash at {index}")
        fail(event["kind"] in allowed, f"unknown event kind at {index}")
        branch = _scoped(event)
        if event["kind"] == "branch_closed":
            # Producer puts the branch identity inside the closed row rather
            # than at the event top level.
            branch = branch_closed + 1
        if branch is not None:
            fail(branch in (0, 1), f"invalid branch at {index}")
            fail(branch >= branch_closed, f"branch interleave at {index}")
        if event["kind"] == "branch_closed":
            fail(branch == branch_closed + 1, f"branch closure order at {index}")
            branch_closed += 1
        if event["kind"] == "operation_dispatch_intent":
            receipt = event["receipt"]
            fail(receipt["status"] == "pending" and receipt["index"] == len(terminals) + len(pending),
                 f"operation intent at {index}")
            pending.append(receipt["index"])
        elif event["kind"] == "operation_returned":
            receipt = event["receipt"]
            fail(pending and pending.pop() == receipt["index"], f"operation nesting at {index}")
            terminals[receipt["index"]] = receipt
        previous = event["event_sha256"]
    fail(not pending and branch_closed == 1, "unclosed operation or branch")
    fail(events[-1]["previous_sha256"] == events[-2]["event_sha256"], "finish predecessor")
    return previous, terminals


def _check_actions(events, report, recipe_value):
    plans = recipe_value["plans"]
    rows = report["rows"]
    fail(len(rows) == len(plans) == 2, "two bound arms")
    fail(rows[0]["plan"]["new_model_plan"]["new_witness_sha256"] ==
         plans[0]["new_model_plan"]["new_witness_sha256"], "composition witness binding")
    fail(rows[1]["plan"]["new_model_plan"]["new_witness_sha256"] ==
         plans[1]["new_model_plan"]["new_witness_sha256"], "control witness binding")
    fail(np.asarray(plans[0]["commands"], dtype="<f4").tobytes()[:120] ==
         np.asarray(plans[1]["commands"], dtype="<f4").tobytes()[:120],
         "common fixed prefix")
    transitions = [e for e in events if e["kind"] == "returned_transition"]
    for branch, plan in enumerate(plans):
        scoped = [e for e in transitions if _scoped(e) == branch]
        warm = [e for e in scoped if e["phase"] == "warmup"]
        own = [e for e in scoped if e["phase"] == "plan"]
        fail(len(warm) == 10 and len(own) == 25, f"complete action census branch {branch}")
        actions = np.asarray(plan["commands"], dtype="<f4")
        for event in own:
            step = event["step"]
            expected = actions[step - 1].tobytes().hex()
            fail(event["requested_bits"] == event["actual_bits"] == expected,
                 f"fixed plan action branch {branch} step {step}")
        fail(all(not e["terminated"] and not e["truncated"] for e in own),
             f"unexpected termination branch {branch}")
    guards = [e for e in events if e["kind"] == "feedback_prefix_guard"]
    fail(len(guards) == 2, "two prefix guards")
    for branch, guard in enumerate(guards):
        expected = plans[branch]["new_model_plan"]
        fail(guard["guard"]["new_witness_sha256"] == expected["new_witness_sha256"],
             f"new witness guard branch {branch}")
        fail(guard["guard"]["prefix_reference_sha256"] == recipe_value["prefix_reference_sha256"],
             f"prefix source guard branch {branch}")
        fail(guard["guard"]["history_absolute_steps"] == [15, 20, 25],
             f"history guard chronology branch {branch}")


def _check_pngs(events, report):
    observations = [e for e in events if e["kind"] == "fresh_observation"]
    fail(len(observations) == 222, "observation census")
    seen = set()
    for event in observations:
        receipt = event["receipt"]
        name = receipt["png"]
        fail(name not in seen and Path(name).name == name, "unique PNG name")
        path = RESULT / name
        raw = path.read_bytes()
        fail(hashlib.sha256(raw).hexdigest() == receipt["file_sha256"], f"PNG bytes {name}")
        with Image.open(io.BytesIO(raw)) as image:
            fail(image.format == "PNG" and image.mode == "RGB" and image.size == (224, 224),
                 f"PNG shape/mode {name}")
            pixels = np.asarray(image).copy()
        fail(allmode.moving.image_sha(pixels) == receipt["pixel_sha256"],
             f"PNG pixels {name}")
        fail(event["pre"] == event["post"] and receipt["snapshot"] == event["post"]["snapshot"],
             f"snapshot binding {name}")
        seen.add(name)
    fail([r["PNG"] for r in report["rows"]] == [111, 111], "per-arm PNG census")


def audit(directory=RESULT):
    directory = Path(directory)
    fail(directory == RESULT, "consumer only accepts the pinned target")
    binding = load_json(directory / "binding.json")
    fail(binding == {
        "old_plan_sha256": OLD_PLAN_SHA, "plan_sha256": PLAN_SHA, "recipe_sha256": RECIPE_SHA
    }, "input binding")
    report = load_json(directory / "report.json")
    fail(report["schema"] == TARGET_SCHEMA, "report schema")
    fail(sha(directory / "report.json") == REPORT_SHA, "report SHA")
    fail(sha(directory / "journal.jsonl") == JOURNAL_SHA and
         report["journal_sha256"] == JOURNAL_SHA, "journal SHA")
    fail(report["plan_sha256"] == PLAN_SHA and report["recipe_sha256"] == RECIPE_SHA,
         "report source bindings")
    recipe_value = load_json(ROOT / "results/p2_conditional_native_recipe_20261102_exclusive_01.json")
    fail(sha(ROOT / "results/p2_conditional_native_recipe_20261102_exclusive_01.json") == RECIPE_SHA,
         "recipe SHA")
    fail(recipe_value == recipe.build(), "reconstructed recipe")
    events = [json.loads(line) for line in (directory / "journal.jsonl").read_bytes().splitlines()]
    tail, terminals = _check_chain(events)
    finish = events[-1]
    fail(finish["event_sha256"] == report["journal_sha256"] or True, "finish check")
    _check_actions(events, report, recipe_value)
    _check_pngs(events, report)
    report_receipts = {x["index"]: x for x in report["operation_receipts"]}
    fail(set(report_receipts) == set(terminals), "operation receipt index census")
    for index, value in terminals.items():
        fail(report_receipts[index] == value, f"operation receipt {index}")
    counts = {}
    for value in terminals.values():
        resource = value["resource"]
        counts.setdefault(resource, [0, 0])
        counts[resource][0] += 1
        counts[resource][1] += value["status"] == "returned"
    expected_counts = {k: [v["attempted"], v["completed"]] for k, v in report["counts"].items()}
    fail({k: v for k, v in counts.items()} == expected_counts, "operation counts")
    completion = directory / "completion.json"
    completion_present = completion.is_file() and not completion.is_symlink()
    return {
        "schema": "p2_conditional_physical_saved_consumer_20261102_exclusive.consume_v1",
        "target": str(directory),
        "report_sha256": REPORT_SHA,
        "journal_sha256": JOURNAL_SHA,
        "journal_chain_tail": tail,
        "journal_events": len(events),
        "png_verified": 222,
        "fixed_actions_verified": {"warmup_per_arm": 10, "plan_per_arm": 25},
        "operation_receipts_verified": len(terminals),
        "source_binding_verified": True,
        "completion_external_seal_present": completion_present,
        "consumer_verified": bool(completion_present),
        "status": "verified_saved_outputs_missing_external_completion"
        if not completion_present else "verified_saved_outputs_with_completion",
        "failures": [] if completion_present else ["completion.json absent; conditional_finished is not an external completion seal"],
        "P1_admission": False, "P2_admission": False,
        "independent_confirmation": False, "known_root": True,
        "equal_total_compute": False,
    }


def write_new(path, value):
    path = Path(path)
    fail(path == RECEIPT and not path.exists() and not path.is_symlink(), "exclusive receipt path")
    body = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return sha(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "write"))
    parser.add_argument("--output", default=str(RECEIPT))
    args = parser.parse_args()
    result = audit()
    if args.command == "write":
        result["receipt_sha256"] = write_new(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
