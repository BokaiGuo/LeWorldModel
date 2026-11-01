import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev303_v12_adapter_20261101_exclusive as subject,
)


def put(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def planner_fixture(tmp_path, monkeypatch):
    prep = {
        "schema": "p1_dev303_blind_planners_20261101_codex.prepare",
        "slots": [{"order": 0, "task_export": str(subject.TASK),
                   "task_sha256": subject.TASK_SHA}],
        "expected_model_hash": "model", "contract": {}, "policy": "policy",
    }
    prep_path = tmp_path / "prep.json"
    prep_sha = put(prep_path, prep)
    rows, artifacts = [], {}
    for arm in subject.ARMS:
        name = f"root_0_{arm}.plan.json"
        plan = {"arm": arm, "task_sha256": subject.TASK_SHA,
                "selection_sha256": arm, "commands": np.zeros((10, 2)).tolist()}
        artifacts[name] = put(tmp_path / name, plan)
        rows.append({"order": 0, "arm": arm, "status": "model_plan_sealed",
                     "plan_file": name, "selection_sha256": arm})
    report = {
        "schema": "p1_constructive_blind_planners_cuda_20261027_v1.report",
        "status": "model_plans_sealed_not_admitted", "commands_released": True,
        "errors": [], "source_verified": True, "model_unchanged": True,
        "all_plans_before_physics": True, "new_physics": 0, "P1_admission": False,
        "prepared": str(prep_path), "prepare_sha256": prep_sha,
        "model_hash_before": "model", "model_hash_after": "model",
        "contract": {}, "rows": rows, "artifact_sha256": artifacts,
    }
    path = tmp_path / "report.json"
    report_sha = put(path, report)
    monkeypatch.setattr(subject, "REPORT_SHA", report_sha)
    monkeypatch.setattr(subject, "PLAN_SHA", {
        arm: artifacts[f"root_0_{arm}.plan.json"] for arm in subject.ARMS
    })
    plans = [json.loads((tmp_path / f"root_0_{arm}.plan.json").read_text())
             for arm in subject.ARMS]
    monkeypatch.setattr(subject, "consume_planner", lambda *_: {
        "status": "model_commands_verified_for_separate_MAIN_executor",
        "new_physics": 0, "P1_admission": False, "plans": plans,
    })
    return path, report_sha


def test_real_rgb_candidate_contract_with_fixture_plans(tmp_path, monkeypatch):
    path, sha = planner_fixture(tmp_path, monkeypatch)
    result = subject.prepare(path, sha)
    assert result["P1_admission"] is False
    assert result["selector_execution_authorized"] is False
    assert result["budget"]["predict_samples"] == 54417
    assert len(result["prefix_bank"]["candidates"]) == 17
    assert result["source_receipt_sha256"] == subject.RECEIPT_SHA


def test_report_and_plan_drift_fail_closed(tmp_path, monkeypatch):
    path, sha = planner_fixture(tmp_path, monkeypatch)
    (tmp_path / f"root_0_{subject.ARMS[0]}.plan.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA mismatch"):
        subject.prepare(path, sha)
    with pytest.raises(ValueError, match="report SHA"):
        subject.prepare(path, "0" * 64)


def test_selector_ties_abstention_and_baseline_identity():
    rows = [{"index": i, "contact_feasible_prior": i in (3, 5)} for i in range(17)]
    scores = [{"robust_score": 1., "cp5_q": 1.} for _ in rows]
    selected = subject.select(rows, scores)
    assert selected["selected_index"] == 3
    assert selected["latent_greedy_index"] == 0
    assert selected["task_greedy_direct_index"] == 4
    rows[3]["contact_feasible_prior"] = rows[5]["contact_feasible_prior"] = False
    with pytest.raises(ValueError, match="abstain"):
        subject.select(rows, scores)
    scores[0]["cp5_q"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        subject.select(rows, scores)


def test_forbidden_and_symlink_inputs(tmp_path):
    path = tmp_path / "physical.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="forbidden"):
        subject.read_fixed(path, hashlib.sha256(b"{}").hexdigest())
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular input"):
        subject.read_fixed(link, hashlib.sha256(b"{}").hexdigest())
