"""Strict saved-ledger census; no model, GPU, physics, or launch authority."""
import hashlib
import json
import math
from pathlib import Path

from . import p2_common_objective_ledger_20261102_exclusive as ledger
from . import p2_same_bank_native_objective_20261102_exclusive as native

ARMS = native.optimizer.ARMS
MODULES = ("action_encoder", "predict", "predictor", "pred_proj")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(path, profile):
    """Reconstruct the full schedule against the real fixed bank.

    `native_hooks` validates recorded hooks, not actual hardware, CUDA execution,
    loader integrity or production authorization. `cpu_fixture` is not neural
    evidence. Incomplete or failed logs never return a completed census.
    """
    require(profile in ("cpu_fixture", "native_hooks"), "unknown profile")
    fixed = native.verified_input()
    path = Path(path)
    summary = ledger.verify_ledger(path)
    require(not summary["outstanding"], "pending ledger operations")
    raw = path.read_bytes()
    events = [json.loads(line)["record"] for line in raw.splitlines()]
    require(events[0]["kind"] == "ledger_open", "ledger open")
    expected = [(i, arm, iteration)
                for i in range(16) for arm in ARMS for iteration in range(9)]
    cursor, active = 0, None
    completed = {arm: {"scores": 0, "gradients": 0} for arm in ARMS}
    for event in events[1:]:
        kind, payload, category = event["kind"], event["payload"], event["category"]
        if kind == "optimizer_attempt":
            require(active is None and cursor < len(expected), "unexpected optimizer attempt")
            index, arm, iteration = expected[cursor]
            want_grad = iteration < 8
            require(category is None and payload["kind"] == "attempt" and
                    (payload["candidate_index"], payload["arm"], payload["iteration"]) ==
                    (index, arm, iteration) and payload["gradient_required"] is want_grad,
                    "optimizer sequence or gradient")
            action_sha = payload["action_sha256"]
            require(isinstance(action_sha, str) and len(action_sha) == 64, "action SHA")
            if iteration == 0:
                require(action_sha == hashlib.sha256(fixed["bank"][index]).hexdigest(),
                        "initial action mismatch")
            active = {"index": index, "arm": arm, "iteration": iteration, "grad": want_grad,
                      "action_sha": action_sha, "objective": "pending", "axes": {}}
        elif kind == "objective_attempt":
            require(active is not None and active["objective"] == "pending" and
                    category == "common_" + active["arm"] and payload["arm"] == active["arm"] and
                    payload["gradient"] is active["grad"], "objective attempt")
            active["objective"] = "running"
        elif kind in ("attempt", "returned"):
            require(active is not None and active["objective"] == "running" and
                    category == "common_" + active["arm"], "operation outside objective")
            axis = payload["axis"]
            require(axis in native.CAPS, "unexpected operation axis")
            if kind == "attempt":
                active["axes"][axis] = active["axes"].get(axis, 0) + 1
        elif kind == "objective_returned":
            require(active is not None and active["objective"] == "running" and
                    category == "common_" + active["arm"] and payload["arm"] == active["arm"] and
                    payload["gradient"] is active["grad"], "objective return")
            expected_axes = {"transition_packet": 2}
            if active["grad"]:
                expected_axes["autograd_grad"] = 1
            if profile == "native_hooks":
                expected_axes.update({axis: 2 for axis in MODULES})
            require(active["axes"] == expected_axes, "operation attempts incomplete or unexpected")
            active["objective"] = "returned"
        elif kind == "optimizer_returned":
            require(active is not None and active["objective"] == "returned" and category is None and
                    payload["kind"] == "returned" and payload["candidate_index"] == active["index"] and
                    payload["arm"] == active["arm"] and payload["iteration"] == active["iteration"] and
                    payload["gradient_required"] is active["grad"] and
                    payload["action_sha256"] == active["action_sha"] and
                    type(payload["score"]) in (int, float) and math.isfinite(payload["score"]),
                    "optimizer return")
            completed[active["arm"]]["scores"] += 1
            completed[active["arm"]]["gradients"] += int(active["grad"])
            active = None
            cursor += 1
        else:
            raise ValueError("unexpected, failed, or unsupported ledger event: " + kind)
    require(active is None and cursor == len(expected), "incomplete schedule")
    require(all(v == {"scores": 144, "gradients": 128} for v in completed.values()), "count mismatch")
    return {"schema": "p2_same_bank_ledger_consumer_20261102_exclusive_v1",
            "profile": profile, "ledger_sha256": hashlib.sha256(raw).hexdigest(),
            "ledger_tail_sha256": summary["tail_sha256"], "callbacks_returned": completed,
            "candidate_count_per_arm": 16, "context_sha256": fixed["context_sha256"],
            "goal_sha256": fixed["goal_sha256"],
            "native_execution_authenticated": False, "composition_mechanism_authenticated": False,
            "production_execution_authorized": False, "equal_total_compute": False,
            "P1_admission": False, "P2_admission": False}
