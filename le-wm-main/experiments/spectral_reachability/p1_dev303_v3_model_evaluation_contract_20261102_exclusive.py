"""Development-only preseal and score-consumption interface; no model execution.

Seven fixed five-action programs: four RGB geometric approaches plus zero,
the existing RGB direct-support baseline, and its seeded random control.
This does not authorize evaluation or establish prospective independence.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_rgb_boundary_static_v3_20261102_exclusive as static
from . import p1_dev302_mixed_prefix_bank_20261101_codex as original_bank

NAME = "p1_dev303_v3_model_evaluation_contract_20261102_exclusive"
ROOT = Path(__file__).resolve().parents[2]
MODEL_PREP = ROOT / "results/p1_dev303_blind_planners_20261101_codex_PREP_MAIN01.json"
MODEL_PREP_SHA = "e02f17ef79dd8a90d5af6a654a93a7533eb6cd075dae15b22efc9b2a41d503ff"
LEDGER = ROOT / "experiments/spectral_reachability/d2p_planning_compute_ledger.py"
LEDGER_SHA = "57ee00c0efd885c692b737316f5f8a59ddf65ac94706479ceeb895448fea5c48"
SELECTION_SCORE = "prefix_cp5_latent_goal_mse"
ROLES = ("selected", "zero", "task_greedy_direct", "random")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _row(index, label, actions, *, selectable):
    try:
        arr = np.asarray(actions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid candidate actions") from exc
    require(arr.shape == (5, 2) and np.isfinite(arr).all() and (np.abs(arr) <= 1).all(),
            "finite bounded 5x2 actions")
    arr = arr.astype("<f4")
    return {"index": index, "label": label, "actions": arr.tolist(),
            "actions_sha256": digest(arr.tobytes()), "eligible_for_selection": selectable}


def build(manifest_raw, npz_raw, prep_raw, ledger_raw, motion_sha):
    require(digest(prep_raw) == MODEL_PREP_SHA and digest(ledger_raw) == LEDGER_SHA,
            "frozen model preparation/ledger source SHA")
    prep = json.loads(prep_raw)
    require(prep["schema"] == "p1_dev303_blind_planners_20261101_codex.prepare"
            and prep["slots"][0]["task_sha256"] ==
            "3787aba5dfe49c97b9c738e347e228c8089449af0df84a5f91788ceb769b2b12"
            and len(prep["slots"]) == 1 and prep["contract"]["P1_admission"] is False,
            "model task binding")
    checked = static.evaluate(manifest_raw, npz_raw, motion_sha)
    result = checked["result"]
    require(result["status"] == "bounded_approach_witnesses"
            and len(result["candidates"]) == 4, "v3 must produce four candidates or abstain")
    with np.load(BytesIO(npz_raw), allow_pickle=False) as archive:
        current = archive["rgb_history"][-1].copy()
        goal = archive["goal_rgb"].copy()
    zeros = np.zeros((5, 2), dtype=np.float32)
    bank = original_bank.build(current, goal, zeros, zeros)
    original = {item["label"]: item for item in bank["candidates"]}
    require("rgb_goal_support_direct" in original and "random_control" in original,
            "original control labels")
    rows = [_row(i, f"rgb_boundary_approach_{i}", c["actions"], selectable=True)
            for i, c in enumerate(result["candidates"])]
    for row, candidate in zip(rows, result["candidates"]):
        require(row["actions_sha256"] == candidate["actions_sha256"], "v3 action hash")
    rows.extend([
        _row(4, "zero", zeros, selectable=False),
        _row(5, "task_greedy_direct", original["rgb_goal_support_direct"]["actions"],
             selectable=False),
        _row(6, "random_control", original["random_control"]["actions"], selectable=False),
    ])
    require(rows[5]["actions_sha256"] == original["rgb_goal_support_direct"]["actions_sha256"]
            and rows[6]["actions_sha256"] == original["random_control"]["actions_sha256"],
            "original baseline action identity")
    return {
        "schema": NAME + ".prepare.v1",
        "status": "development_prepared_not_executable",
        "classification": "post_development_not_independent",
        "blind_npz_sha256": checked["blind_npz_sha256"],
        "manifest_sha256": checked["manifest_sha256"],
        "config_sha256": checked["config_sha256"],
        "current_rgb_sha256": result["current_rgb_sha256"],
        "goal_rgb_sha256": result["goal_rgb_sha256"],
        "model_prepare_sha256": MODEL_PREP_SHA,
        "model_sha256_expected_from_prepare": prep["expected_model_hash"],
        "model_policy": prep["policy"], "task": prep["slots"][0],
        "model_source_bindings_declared_not_reverified": prep["bindings_sha256"],
        "ledger_source_sha256": LEDGER_SHA,
        "ledger_interface": "d2p_planning_compute_ledger.Binding/Resources/Ledger",
        "ledger_status": "source_bound_not_instrumented",
        "producer_sources": {
            str(Path(static.v3.__file__).resolve()): digest(Path(static.v3.__file__).read_bytes()),
            str(Path(original_bank.__file__).resolve()): digest(Path(original_bank.__file__).read_bytes()),
            str(Path(__file__).resolve()): digest(Path(__file__).read_bytes()),
        },
        "candidates": rows,
        "selection": {"eligible_indices": [0, 1, 2, 3], "score": SELECTION_SCORE,
                      "rule": "minimum finite score, stable lowest candidate index",
                      "control_indices": {"zero": 4, "task_greedy_direct": 5, "random": 6},
                      "task_greedy_direct_definition": "existing rgb_goal_support_direct prefix, not latent greedy",
                      "random_definition": "existing random_control seed 3021101"},
        "evaluation_contract": {"prefix_steps": 5, "per_candidate_rollout": "one prefix only",
                                "score": SELECTION_SCORE, "candidate_count": 7,
                                "budget": {"action_encoder_calls": 7, "predict_calls": 7,
                                           "predict_samples": 7},
                                "assumptions": "one native predict and action encoder call per 5-step prefix; must be verified before execution"},
        "failure_policy": {"missing_or_failed_candidate": "abstain_all_no_release",
                           "ledger_or_model_mismatch": "abstain_all_no_release",
                           "preserve_partial_and_attempted": True, "retry": False,
                           "replace": False, "outcome_access": False},
        "executor_integrated": False, "model_execution_authorized": False,
        "physical_execution_authorized": False, "source_chain_reverified": False,
        "new_model_calls": 0, "new_physics_calls": 0,
        "independent_confirmation": False, "P1_admission": False,
        "candidate_semantics": "RGB geometry approach, not spectral programs or proven contact",
    }


def select(prepared, scores):
    """Pure selection only; score provenance/ledger verification is caller-owned."""
    require(prepared["schema"] == NAME + ".prepare.v1"
            and prepared["status"] == "development_prepared_not_executable"
            and prepared["P1_admission"] is False
            and prepared["physical_execution_authorized"] is False, "prepare boundary")
    labels = [f"rgb_boundary_approach_{i}" for i in range(4)] + [
        "zero", "task_greedy_direct", "random_control"]
    rows = prepared["candidates"]
    require(len(rows) == 7, "candidate roster")
    for i, row in enumerate(rows):
        require(row == _row(i, labels[i], row["actions"], selectable=i < 4),
                "candidate identity or SHA")
    require(not np.asarray(rows[4]["actions"]).any(), "zero baseline")
    require(len(scores) == 7 and all(type(x) in (int, float) and math.isfinite(x) for x in scores),
            "complete finite seven-candidate scores")
    index = min(range(4), key=lambda i: (scores[i], i))
    return {"selected_index": index, "control_indices": {"zero": 4, "task_greedy_direct": 5,
                                                         "random": 6},
            "score_provenance_verified": False, "ledger_verified": False,
            "physical_execution_authorized": False, "P1_admission": False}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not os.path.lexists(args.output), "exclusive output")
    value = build(static.baseline.MANIFEST.read_bytes(), static.baseline.NPZ.read_bytes(),
                  MODEL_PREP.read_bytes(), LEDGER.read_bytes(),
                  digest(static.baseline.MOTION_SOURCE.read_bytes()))
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({"output": str(args.output), "sha256": digest(payload),
                      "status": value["status"]}))


if __name__ == "__main__":
    main()
