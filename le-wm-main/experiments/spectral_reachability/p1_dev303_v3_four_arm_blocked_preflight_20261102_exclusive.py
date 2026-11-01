"""Read-only identity audit for a blocked development four-arm prefix proposal.

This is deliberately not a physical executor or an authentication mechanism.
Matching local hashes and logs cannot certify who produced a score.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/p1_dev303_v3_cuda_executor_20261102_exclusive_RUN_DEV01"
MODEL_PREP = ROOT / "results/p1_dev303_v3_model_evaluation_contract_20261102_exclusive_PREP_02.json"
CUDA_PREP = ROOT / "results/p1_dev303_v3_cuda_executor_20261102_exclusive_PREP_02.json"
PINS = {
    "report": "e9bb449bcb44aff75df4061b867f6dacd9ebcb17202dea0c4b7fa0078fb6f15c",
    "events": "5727e49cd70666c0b2aeed96447b97974672a3f2d35f925e1a951aa12b7ec35c",
    "model_prepare": "fafd1d2de1d211af9166c6ff4966a61ecf29b83c78fda6e2e0a3bffaa7e44345",
    "cuda_prepare": "213345b6680e7cd139768b50857180864a0d512213ba0da535ce77f5c3dfde35",
}
MODEL_HASH = "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546"
ROLES = (("selected", 3), ("zero", 4), ("task_greedy_direct", 5), ("random", 6))
LABELS = tuple(f"rgb_boundary_approach_{i}" for i in range(4)) + (
    "zero", "task_greedy_direct", "random_control")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(raw):
    """Audit exact pinned bytes, returning a non-authorizing proposed contract."""
    require(set(raw) == set(PINS), "missing or extra source")
    for name, expected in PINS.items():
        require(hashlib.sha256(raw[name]).hexdigest() == expected, f"{name} SHA mismatch")
    report = json.loads(raw["report"])
    prep = json.loads(raw["model_prepare"])
    events = [json.loads(line) for line in raw["events"].splitlines()]
    require(report["schema"] == "p1_dev303_v3_cuda_executor_20261102_exclusive.report.v1"
            and report["status"] == "model_only_complete"
            and report["contract_sha256"] == PINS["cuda_prepare"], "report identity")
    require(report["P1_admission"] is False and report["physical_execution_authorized"] is False
            and report["independent_confirmation"] is False, "report boundaries")
    require(report["model_before"] == report["model_after"]
            and report["model_before"]["original_state_dict"] == MODEL_HASH, "model hash")
    rows, candidates = report["rows"], prep["candidates"]
    require(len(rows) == len(candidates) == 7, "complete roster")
    for i, (row, candidate) in enumerate(zip(rows, candidates)):
        actions = np.asarray(candidate["actions"], dtype=np.float64)
        require(actions.shape == (5, 2) and np.isfinite(actions).all()
                and (np.abs(actions) <= 1).all(), "bounded actions")
        action_sha = hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest()
        require(candidate["index"] == row["index"] == i
                and candidate["label"] == row["label"] == LABELS[i]
                and candidate["eligible_for_selection"] is (i < 4)
                and candidate["actions_sha256"] == row["actions_sha256"] == action_sha,
                "candidate identity")
        require(type(row["score"]) in (int, float) and math.isfinite(row["score"]), "score")
    require(not np.asarray(candidates[4]["actions"]).any(), "zero identity")
    selection = report["selection"]
    require(selection["selected_index"] == 3
            and min(range(4), key=lambda i: (rows[i]["score"], i)) == 3
            and selection["control_indices"] == dict(ROLES[1:]), "selection identity")
    expected = [{"kind": "start", "run_contract_sha256": PINS["cuda_prepare"]}]
    for row in rows:
        i = row["index"]
        expected.extend([
            {"kind": "candidate_start", "index": i, "actions_sha256": row["actions_sha256"]},
            {"kind": "dispatch", "index": i, "resource": "action_encoder"},
            {"kind": "dispatch", "index": i, "resource": "predict", "batch": 1},
            {"kind": "candidate_complete", **row},
        ])
    expected.append({"kind": "complete", "report_sha256": PINS["report"]})
    require(events == expected, "model event sequence")
    require(report["attempts"] == {"action_encoder": 7, "predict": 7}
            and report["hooks"] == {"action_encoder": 7, "predictor": 7,
                                    "predictor_samples": 7, "image_encoder": 0,
                                    "image_projector": 0}, "model counters")
    return {
        "schema": "p1_dev303_v3_four_arm_blocked_preflight.v1",
        "classification": "development_only_post_development_not_independent",
        "status": "blocked_uncertified_scorer_and_ledger",
        "observed_local_sha256_not_external_authentication": PINS.copy(),
        "model_state_dict_sha256_reported": MODEL_HASH,
        "arms_fixed_before_physics": [
            {"role": role, "index": i, "label": candidates[i]["label"],
             "actions": candidates[i]["actions"], "actions_sha256": candidates[i]["actions_sha256"]}
            for role, i in ROLES
        ],
        "report_score_provenance_verified": selection["score_provenance_verified"],
        "report_ledger_verified": selection["ledger_verified"],
        "physical_execution_authorized": False,
        "P1_admission": False,
        "independent_confirmation": False,
    }


def audit_local():
    return audit({"report": (RUN / "report.json").read_bytes(),
                  "events": (RUN / "events.jsonl").read_bytes(),
                  "model_prepare": MODEL_PREP.read_bytes(),
                  "cuda_prepare": CUDA_PREP.read_bytes()})


if __name__ == "__main__":
    print(json.dumps(audit_local(), sort_keys=True, indent=2, allow_nan=False))
