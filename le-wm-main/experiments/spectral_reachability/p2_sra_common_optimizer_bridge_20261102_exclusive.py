"""Known-root development bridge: fixed SRA budget and common optimizer core.

The optimizer callback is an ABI only. Production execution requires separately
authenticated native initializer receipts, a guarded CUDA model adapter, and
durable full-resource metering. This module never loads or invokes a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p2_cp15_capture_builder_20261102_codexexclusive as capture
from . import p2_shared_capture_launch_20261102_exclusive as shared
from . import codex_sra_online_run_bank_20261030 as sra

NAME = "p2_sra_common_optimizer_bridge_20261102_exclusive"
ARMS = ("composition", "no_composition_sra")
STEPS = 8
LEARNING_RATE = 0.05
SRA_SOURCE_SHA = sra.native.SRA_SHA


def need(ok, message):
    if not ok:
        raise ValueError(message)


def raw_actions(value):
    array = np.asarray(value)
    need(array.shape == (10, 2) and array.dtype == np.dtype("float32")
         and np.isfinite(array).all() and (np.abs(array) <= 1).all(),
         "finite bounded raw float32 10x2")
    return np.array(array, dtype="<f4", copy=True)


def from_sra_search(search, bank):
    """Extract original v2 joint-H2 winner after its existing bank validation.

    Does not authenticate the enclosing source, event ledger or CUDA model;
    caller must separately use the existing strict SRA consumer.
    """
    _, winner_id = sra.audit_candidates(search, bank)
    found = [a for a in search["attempts"] if a["candidate_id"] == winner_id]
    need(len(found) == 1, "unique SRA winner")
    raw = np.asarray(found[0]["actions"], dtype=np.float32)
    return {"method": "joint_h2_no_composition_sra_v2",
            "candidate_id": winner_id, "raw_actions": raw_actions(raw)}


def optimize_pair(initializers, evaluate, emit):
    """Same 8-step projected gradient and 9 scores per arm; no model imports.

    evaluate(arm, raw_copy, gradient_required) -> (finite loss, float32 gradient
    or None). emit is called before and after each callback. The best evaluated
    iterate (including the initial and final iterate) is selected; ties favor
    the earlier iterate. Any exception stops the whole pair and cannot refund
    the already emitted attempt. A callback ABI is not a native resource meter.
    """
    need(type(initializers) is dict and tuple(initializers) == ARMS and callable(evaluate)
         and callable(emit), "fixed arms and callbacks")
    result = []
    for arm in ARMS:
        x = raw_actions(initializers[arm])
        observations = []
        for index in range(STEPS + 1):
            requested_gradient = index < STEPS
            encoded = hashlib.sha256(x.tobytes()).hexdigest()
            emit({"kind": "attempt", "arm": arm, "iteration": index,
                  "action_sha256": encoded, "gradient_required": requested_gradient})
            try:
                score, grad = evaluate(arm, x.copy(), requested_gradient)
                need(type(score) in (int, float) and np.isfinite(score), "finite scalar score")
                if requested_gradient:
                    need(isinstance(grad, np.ndarray) and grad.shape == (10, 2)
                         and grad.dtype == np.float32 and np.isfinite(grad).all(), "float32 gradient")
                else:
                    need(grad is None, "final score must not return gradient")
            except BaseException as exc:
                emit({"kind": "failed", "arm": arm, "iteration": index,
                      "error_type": type(exc).__name__})
                raise
            emit({"kind": "returned", "arm": arm, "iteration": index,
                  "action_sha256": encoded, "score": float(score),
                  "gradient_required": requested_gradient})
            observations.append((float(score), index, x.copy()))
            if requested_gradient:
                x = np.clip(x - np.float32(LEARNING_RATE) * grad,
                            np.float32(-1), np.float32(1)).astype("<f4")
        cost, index, selected = min(observations, key=lambda row: (row[0], row[1]))
        result.append({"arm": arm, "winner_iteration": index, "model_score": cost,
                       "raw_float32_hex": selected.tobytes().hex(),
                       "attempted_score_calls": STEPS + 1, "returned_score_calls": STEPS + 1,
                       "attempted_gradients": STEPS, "returned_gradients": STEPS})
    return {"schema": NAME + ".pair_fixture_or_adapter.v1", "rows": result,
            "callback_is_native_meter": False, "equal_total_compute": False,
            "independent_confirmation": False, "P1_admission": False, "P2_admission": False}


def contract():
    source = capture.verify(shared.CAPTURE, shared.CAPTURE_SHA)
    bundle, _, _, _ = shared.wire.load_capture(shared.CAPTURE / "capture.json", shared.CAPTURE_SHA)
    actual_sra_sha = shared.wire.checkpoint.sha(Path(sra.native.SRA_SOURCE))
    need(actual_sra_sha == SRA_SOURCE_SHA and bundle["base"] == 0, "native SRA source and B0")
    return {"schema": NAME + ".contract.v1", "capture_sha256": shared.CAPTURE_SHA,
            "bridge_source_sha256": shared.wire.checkpoint.sha(__file__),
            "capture_source": source, "root_sha256": bundle["source"]["root_sha256"],
            "task_id": bundle["source"]["task_id"], "arms": list(ARMS),
            "sra": {"source_sha256": actual_sra_sha, "candidate_count": 40,
                    "predict_and_action_encoder_calls": 88, "jacobian": 1,
                    "svd": 2, "bounded_solve": 1, "vjp_directions": 192,
                    "scope": "existing native SRA initializer including winner replay"},
            "common_optimizer": {"algorithm": "projected_gradient_incumbent_preserving",
                                 "steps": STEPS, "learning_rate": LEARNING_RATE,
                                 "raw_action_shape": [10, 2], "bounds": [-1, 1],
                                 "score_callbacks_per_arm": 9, "gradient_callbacks_per_arm": 8,
                                 "tie": "earliest", "post_update_final_evaluation": True},
            "total_budget": {"status": "blocked", "reason": "composition initializer and native optimizer internal resource costs unbound",
                             "unknown": ["flops", "energy", "io", "engine_backward_calls"]},
            "production_execution_authorized": False, "fixture_results_are_evidence": False,
            "independent_confirmation": False, "P1_admission": False, "P2_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    value = contract()
    if args.output:
        path = Path(args.output).absolute()
        need(path.parent == shared.RESULTS and path.name.startswith(NAME + "_CONTRACT_"),
             "exclusive contract path")
        print(json.dumps({"path": str(path), "sha256": shared.write_new(path, value)}, sort_keys=True))
    else:
        print(json.dumps(value, sort_keys=True, allow_nan=False))
