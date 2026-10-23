from __future__ import annotations

import copy

import pytest

from experiments.gdd_wm.evaluate_pointmaze_shared_future_v8_htsc import evaluate


def _action(value: float) -> list[list[float]]:
    return [[value, 0.0] for _ in range(20)]


def _root(root: str, *, reversed_prediction: bool = False) -> dict:
    prediction = (0.9, 0.1) if reversed_prediction else (0.1, 0.9)
    return {
        "instance_id": root,
        "candidates": [
            {"candidate_id": "a", "candidate_action_hash": "a", "action_sequence": _action(-0.2), "predicted_cost": prediction[0], "realized_cost": 0.1},
            {"candidate_id": "b", "candidate_action_hash": "b", "action_sequence": _action(0.2), "predicted_cost": prediction[1], "realized_cost": 0.9},
        ],
    }


def _arm(name: str) -> dict:
    return {
        "name": name,
        "calibration_candidate_costs": [_root(f"cal-{index}") for index in range(192)],
        "outcome_candidate_costs": [_root(f"out-{index}") for index in range(64)],
    }


def _report() -> dict:
    return {
        "schema": "pointmaze_planning_shaped_execution_control.v0.4",
        "control_contract": {
            "action_representation": "full_ordered_20x2", "shared_future_control_included": True,
            "v8_static_preflight": {
                "verified": True, "root_counts": {"train": 128, "calibration": 192, "outcome": 64},
                "action_representation": "full_ordered_20x2",
            },
        },
        "split_isolation": {"train_root_count": 128, "calibration_root_count": 192, "outcome_root_count": 64, "root_disjoint": True},
        "candidate_objectives": [_arm("cost_regression"), _arm("planning_shaped"), _arm("shared_future_planning")],
    }


def _preflight() -> dict:
    return {
        "schema": "pointmaze_shared_future_v8_static_preflight.v1",
        "status": "passed_static_preflight",
        "candidate_bank": {"shared_candidate_in_support": True, "scorer_candidate_bank_protocol": "held_exact_shared_bank"},
    }


def test_v8_htsc_uses_calibration_only_thresholds_and_reports_fixed_bank_release_metrics():
    result = evaluate(_report(), _preflight(), alpha=0.1, risk_quantile=0.9)
    assert result["status"] == "completed_frozen_v8_htsc_audit"
    assert set(result["arm_evaluations"]) == {"cost_regression", "planning_shaped", "shared_future_planning"}
    shared = result["arm_evaluations"]["shared_future_planning"]
    assert shared["calibration"]["contrast"]["calibration_root_count"] == 192
    assert shared["outcome"]["root_count"] == 64
    assert shared["outcome"]["released_root_count"] == 64
    assert shared["outcome"]["wrong_release_rate"] == 0.0
    assert shared["outcome"]["mean_utility_gain_over_fallback"] == 0.0


def test_v8_htsc_rejects_cross_arm_action_bank_drift():
    report = _report()
    report["candidate_objectives"][2]["outcome_candidate_costs"][0]["candidates"][0]["action_sequence"] = _action(-0.5)
    with pytest.raises(ValueError, match="exact action sequence"):
        evaluate(report, _preflight())


def test_v8_htsc_reports_wrong_release_only_after_the_frozen_choice():
    report = _report()
    for row in report["candidate_objectives"][2]["outcome_candidate_costs"]:
        row["candidates"][0]["predicted_cost"] = 0.9
        row["candidates"][1]["predicted_cost"] = 0.1
    result = evaluate(report, _preflight())
    shared = result["arm_evaluations"]["shared_future_planning"]["outcome"]
    assert shared["released_root_count"] == 64
    assert shared["wrong_release_count"] == 64
    assert shared["wrong_release_rate"] == 1.0
    assert shared["mean_utility_gain_over_fallback"] < 0.0


def test_v8_htsc_rejects_report_without_exact_supported_candidate_contract():
    preflight = copy.deepcopy(_preflight())
    preflight["candidate_bank"]["shared_candidate_in_support"] = False
    with pytest.raises(ValueError, match="not declared in support"):
        evaluate(_report(), preflight)


def test_v8_htsc_rejects_a_report_bound_to_a_different_static_ledger():
    with pytest.raises(ValueError, match="different static preflight ledger"):
        evaluate(_report(), _preflight(), expected_preflight_sha256="different")
