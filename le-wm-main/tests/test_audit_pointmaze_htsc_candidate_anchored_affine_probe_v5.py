from __future__ import annotations

from experiments.audit_pointmaze_htsc_candidate_anchored_affine_probe_v5 import _trace_violations


def test_forensic_audit_marks_an_executed_backward_pulse_invalid() -> None:
    violations = _trace_violations([{
        "episode_id": 10035,
        "probe_index": 3,
        "probe_candidate_index": 1,
        "action": [0.1, -0.2],
        "action_goal_projection": -0.1486,
        "predicted_goal_progress": 0.1957,
        "observed_goal_progress": -0.0537,
    }])
    assert violations == [{
        "episode_id": 10035,
        "probe_index": 3,
        "probe_candidate_index": 1,
        "action": [0.1, -0.2],
        "action_goal_projection": -0.1486,
        "predicted_goal_progress": 0.1957,
        "observed_goal_progress": -0.0537,
        "violation": "nonpositive_execution_time_goal_projection",
    }]


def test_forensic_audit_accepts_only_strictly_positive_execution_time_safety() -> None:
    assert _trace_violations([{
        "episode_id": 7,
        "probe_index": 0,
        "action_goal_projection": 1.0e-6,
        "predicted_goal_progress": 1.0e-6,
    }]) == []
