import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_v3_four_arm_cp5_v13_inputs_20261102_exclusive as inputs
from experiments.spectral_reachability import p1_dev303_v3_four_arm_cp5_v13_model_20261102_exclusive as model
from experiments.spectral_reachability import p1_dev303_v3_four_arm_cp5_v13_physical_20261102_exclusive as physical


def test_real_four_arm_inputs_and_budget():
    result = inputs.build()
    assert [r["role"] for r in result["roles"]] == list(inputs.ROLES)
    assert len({r["actual_prefix_sha256"] for r in result["roles"]}) == 4
    assert all(len(r["prefix_checkpoints"]) == 5 for r in result["roles"])
    assert result["ledger_verified"] is False
    assert result["budget_target_per_role"]["predict_samples"] == 3200


def test_frozen_input_drift_fails(monkeypatch):
    original = inputs.base.read
    def modified(path):
        value = original(path)
        if path == model.PREP:
            value = copy.deepcopy(value)
            value["roles"][0]["actual_prefix"][0][0] += 1
        return value
    monkeypatch.setattr(inputs.base, "read", modified)
    with pytest.raises(ValueError, match="drift"):
        model.protocol()


def test_kernel_common_seed_and_counts():
    calls = []
    def score(bank):
        calls.append(bank.shape)
        return np.square(bank).sum(axis=(2, 3, 4))
    first = inputs.planner.plan_one([1.0, 0.0], score)
    second = inputs.planner.plan_one([1.0, 0.0], score)
    assert first == second and len(calls) == 10
    assert set(calls) == {(64, 5, 2, 5, 2)}


def test_no_implicit_cuda_authority():
    with pytest.raises(ValueError, match="authority"):
        model.run_cuda("missing", "missing", "missing", None)


def test_physical_handoff_synthetic_report_and_role_tamper(tmp_path):
    source = inputs.base.read(model.PREP)
    tail = np.zeros((2, 5, 2), dtype="<f4")
    rows = [{"role": r["role"], "cp5_pixel_sha256": r["cp5_pixel_sha256"],
             "actual_prefix_sha256": r["actual_prefix_sha256"],
             "initial_bank_sha256": "a" * 64,
             "counts": model.protocol()["per_arm_target"],
             "hooks": {"image_encoder": 1, "image_projector": 1,
                       "action_encoder": 10, "predictor": 10},
             "iterations": [{"iteration": i} for i in range(5)],
             "selected": {"actions": tail.tolist()},
             "tail_actions_sha256": inputs.digest(tail.tobytes())} for r in source["roles"]]
    model_state = {"original_state_dict": source["model_sha256"]}
    report = {"schema": model.NAME + ".report.v1", "status": "model_only_complete",
              "prepared_sha256": physical.MODEL_PREP_SHA, "input_sha256": model.PREP_SHA,
              "model_before": model_state, "model_after": model_state, "rows": rows,
              "ledger_verified": False, "equal_full_budget_certified": False,
              "physical_actions_authorized": False, "P1_admission": False,
              "independent_confirmation": False}
    path = tmp_path / "synthetic-report.json"
    path.write_text(json.dumps(report))
    contract = physical.build(path, inputs.base.sha(path))
    assert [r["arm"] for r in contract["roles"]] == list(inputs.ROLES)
    report["rows"][0]["cp5_pixel_sha256"] = "0" * 64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="role/model counts"):
        physical.build(path, inputs.base.sha(path))


def test_no_physics_authority():
    with pytest.raises(ValueError, match="authority"):
        physical.execute("missing", "missing", "missing", None)
