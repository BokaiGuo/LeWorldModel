from __future__ import annotations

import numpy as np

from experiments.decision_benchmark.audit_rgb_task_contrast import (
    _candidate_collapse_diagnostics,
    _override_predicted_rows,
    _selective_outcome,
    _support_for_root,
)


def test_prediction_report_overrides_only_candidate_predictions(tmp_path):
    report = tmp_path / "adapter.json"
    report.write_text(
        '{"rows": [{"instance_id": "r0", "candidate_id": 0, '
        '"predicted_terminal_projection": [2.0, 3.0]}]}'
    )
    base = {
        ("r0", 0): {
            "predicted_terminal_projection": [0.0, 0.0],
            "realized_terminal_projection": [1.0, 1.0],
        }
    }
    merged = _override_predicted_rows(base, report)
    assert merged[("r0", 0)]["predicted_terminal_projection"] == [2.0, 3.0]
    assert merged[("r0", 0)]["realized_terminal_projection"] == [1.0, 1.0]


def test_support_gate_is_train_only_and_coordinatewise():
    support = {
        "state_min": np.array([0.0, 0.0, -1.0, -1.0]),
        "state_max": np.array([1.0, 1.0, 1.0, 1.0]),
        "action_min": np.array([-1.0, -1.0]),
        "action_max": np.array([1.0, 1.0]),
    }
    rows = [{"source_state_raw": [0.5, 0.5, 0.0, 0.0], "action_sequence": [[0.5, -0.5]] * 4}]
    assert _support_for_root(rows, support)["passed"] is True
    rows[0]["action_sequence"] = [[1.2, 0.0]] * 4
    assert _support_for_root(rows, support)["passed"] is False


def test_selective_release_reports_coverage_and_regret():
    rows = [
        {"instance_id": "r0", "candidate_id": 0, "predicted_cost": 0.1, "realized_rgb_cost": 0.1, "real_task_cost": 0.1},
        {"instance_id": "r0", "candidate_id": 1, "predicted_cost": 0.8, "realized_rgb_cost": 0.8, "real_task_cost": 0.8},
        {"instance_id": "r1", "candidate_id": 0, "predicted_cost": 0.1, "realized_rgb_cost": 0.8, "real_task_cost": 0.8},
        {"instance_id": "r1", "candidate_id": 1, "predicted_cost": 0.2, "realized_rgb_cost": 0.1, "real_task_cost": 0.1},
    ]
    collapse = {
        "r0": {"candidate_collapse_gate": {"passed": True}},
        "r1": {"candidate_collapse_gate": {"passed": True}},
    }
    report = _selective_outcome(rows, {"r0": {"passed": True}, "r1": {"passed": True}}, collapse, bound=0.2)
    assert report["root_count"] == 2
    assert report["released_root_count"] == 1
    assert report["coverage"] == 0.5
    assert report["released_mean_regret"] == 0.0


def test_joint_support_rejects_temporally_unsupported_candidate():
    train = np.asarray([[0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1]], dtype=np.float64)
    support = {
        "state_min": np.zeros(4), "state_max": np.ones(4),
        "action_min": np.zeros(4), "action_max": np.ones(4),
        "state_train": train, "action_train": train,
        "state_scale": np.ones(4), "action_scale": np.ones(4),
        "state_joint_threshold": 0.2, "action_joint_threshold": 0.2,
    }
    rows = [{"source_state_raw": [0.05] * 4, "action_sequence": [[0.05, 0.05]] * 2}]
    assert _support_for_root(rows, support)["passed"] is True
    rows[0]["action_sequence"] = [[0.9, 0.9]] * 2
    assert _support_for_root(rows, support)["passed"] is False


def test_candidate_collapse_gate_detects_constant_predictions():
    rows = [
        {"instance_id": "r0", "candidate_id": 0, "split": "outcome", "action_sequence": [[0.0, 0.0]]},
        {"instance_id": "r0", "candidate_id": 1, "split": "outcome", "action_sequence": [[1.0, 0.0]]},
        {"instance_id": "r0", "candidate_id": 2, "split": "outcome", "action_sequence": [[0.0, 1.0]]},
    ]
    rgb = {
        ("r0", 0): {"predicted_terminal_projection": [0.0, 0.0], "realized_terminal_projection": [0.0, 0.0]},
        ("r0", 1): {"predicted_terminal_projection": [0.0, 0.0], "realized_terminal_projection": [1.0, 0.0]},
        ("r0", 2): {"predicted_terminal_projection": [0.0, 0.0], "realized_terminal_projection": [0.0, 1.0]},
    }
    report = _candidate_collapse_diagnostics(rows, rgb)
    assert report["r0"]["spread_ratio"] == 0.0
    assert report["r0"]["candidate_collapse_gate"]["passed"] is False
