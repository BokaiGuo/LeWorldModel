"""Fixed-bank, fixed-optimizer CPU fixture bridge; not a native experiment.

Both labels see exactly the same 16 saved initial actions. The callback can
model different routes in a fixture, but this code does not authenticate a
composition mechanism, common native objective, or total resource equality.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p2_sra_common_optimizer_bridge_20261102_exclusive as optimizer

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "results/p2_b0_fixed_candidate_handoff_20261102_exclusive_run01.json"
HANDOFF_SHA = "8555a3ba70f696695f168fb196bccc8e86cc0ac839ac7edb41a923910d69bd9c"
PAIR = ROOT / "results/p2_shared_rgb_pair_wiring_20261102_codexexclusive_RUN_20261102_p2continuation_host01/pair.json"
PAIR_SHA = "3f3bdfc70bec1a32e62a7df9093d0fd90e5bd5d43e2960a23fd59e5d27244f88"
ARMS = optimizer.ARMS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_fixed(handoff_raw, pair_raw):
    """Fail closed on any byte, source-bank, context, or goal mismatch."""
    if sha(handoff_raw) != HANDOFF_SHA or sha(pair_raw) != PAIR_SHA:
        raise ValueError("pinned handoff or pair SHA mismatch")
    handoff, pair = json.loads(handoff_raw), json.loads(pair_raw)
    source = handoff["source"]
    if (handoff["schema"] != "p2_b0_fixed_candidate_handoff_20261102_exclusive_v1" or
            source["pair_sha256"] != PAIR_SHA or
            sha(canonical(pair["context"])) != source["context_sha256"] or
            sha(canonical(pair["goal"])) != source["goal_sha256"]):
        raise ValueError("source/context/goal mismatch")
    if ([x["arm"] for x in source["arms"]] != ["joint_multistart", "prefix_chain"] or
            [x["arm"] for x in pair["branches"]] != ["joint_multistart", "prefix_chain"]):
        raise ValueError("source arm census")
    for bank, branch in zip(source["arms"], pair["branches"], strict=True):
        if len(bank["candidates"]) != 16 or len(branch["candidates"]) != 16:
            raise ValueError("candidate census")
        for index, (candidate, original) in enumerate(zip(bank["candidates"], branch["candidates"], strict=True)):
            raw_hex = (original["raw_float32_hex"] if branch["arm"] == "joint_multistart"
                       else original["first_raw_float32_hex"] + original["second_raw_float32_hex"])
            if candidate["index"] != index or original["index"] != index or candidate["raw_float32_hex"] != raw_hex:
                raise ValueError("candidate binding")
            raw = bytes.fromhex(raw_hex)
            if len(raw) != 80:
                raise ValueError("candidate action length")
            optimizer.raw_actions(np.frombuffer(raw, dtype="<f4").reshape(10, 2))
    actions = tuple(bytes.fromhex(c["raw_float32_hex"]) for c in source["arms"][0]["candidates"])
    return {"bank": actions, "context_json": canonical(pair["context"]),
            "goal_json": canonical(pair["goal"]), "context_sha256": source["context_sha256"],
            "goal_sha256": source["goal_sha256"], "handoff_sha256": HANDOFF_SHA,
            "pair_sha256": PAIR_SHA}


def load_real():
    return load_fixed(HANDOFF.read_bytes(), PAIR.read_bytes())


def run_fixture(fixed, evaluator, emit):
    """Apply identical optimizer and bank to both labels; callback is fixture only.

    evaluator(arm, context, goal, raw_actions, gradient_required) has to
    implement a fixture, not a native CUDA/physics operation. Any exception
    stops the whole schedule; attempted callback events are retained by emit.
    """
    if not callable(evaluator) or not callable(emit):
        raise ValueError("fixture callbacks required")
    if fixed != load_real():
        raise ValueError("bound bank/context/goal drift")
    bank = fixed["bank"]
    if len(bank) != 16 or sha(fixed["context_json"]) != fixed["context_sha256"] or sha(fixed["goal_json"]) != fixed["goal_sha256"]:
        raise ValueError("bank/context/goal drift")
    counts = {a: {"scores": 0, "gradients": 0} for a in ARMS}
    results = []
    for index, raw in enumerate(bank):
        if not isinstance(raw, bytes) or len(raw) != 80:
            raise ValueError("invalid bank entry")
        action = optimizer.raw_actions(np.frombuffer(raw, dtype="<f4").reshape(10, 2))
        initial = {arm: action.copy() for arm in ARMS}
        def evaluate(arm, x, need_gradient):
            counts[arm]["scores"] += 1
            counts[arm]["gradients"] += int(need_gradient)
            return evaluator(arm, json.loads(fixed["context_json"]),
                             json.loads(fixed["goal_json"]), x, need_gradient)
        def event(record):
            emit({"candidate_index": index, **record})
        result = optimizer.optimize_pair(initial, evaluate, event)
        results.append({"candidate_index": index, "initial_action_sha256": sha(raw),
                        "arms": result["rows"]})
    if any(c != {"scores": 144, "gradients": 128} for c in counts.values()):
        raise ValueError("callback census mismatch")
    return {"schema": "p2_same_bank_common_optimizer_20261102_exclusive.fixture_v1",
            "rows": results, "counts": counts, "source_handoff_sha256": fixed["handoff_sha256"],
            "context_sha256": fixed["context_sha256"], "goal_sha256": fixed["goal_sha256"],
            "fixture_only": True, "native_objective_authenticated": False,
            "composition_mechanism_authenticated": False, "production_execution_authorized": False,
            "equal_total_compute": False, "P1_admission": False, "P2_admission": False}
