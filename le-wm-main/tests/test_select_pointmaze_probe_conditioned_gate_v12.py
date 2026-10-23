from experiments.decision_benchmark.select_pointmaze_probe_conditioned_gate_v12 import (
    _released_count,
    select,
    validate_abstain_equivalence,
)


def _candidate(*, label: str, released: int, success: int, distance: float) -> dict:
    return {
        "label": label,
        "released_root_count": released,
        "success_delta": success,
        "mean_final_distance_delta": distance,
        "eligible": released >= 4 and success >= 0 and distance <= 0.0,
    }


def test_selector_refuses_empty_or_harmful_calibration_repair() -> None:
    result = select([
        _candidate(label="empty", released=0, success=0, distance=0.0),
        _candidate(label="harmful", released=8, success=-1, distance=-0.1),
    ])
    assert result["deployment_authorized"] is False
    assert result["selected_label"] is None


def test_selector_prefers_task_noninferior_nonempty_gate_before_outcomes() -> None:
    result = select([
        _candidate(label="a", released=4, success=0, distance=-0.02),
        _candidate(label="b", released=6, success=1, distance=-0.01),
    ])
    assert result == {
        "deployment_authorized": True,
        "status": "authorized_calibration_only",
        "selected_label": "b",
    }


def test_selector_treats_auditable_preprobe_abstention_as_zero_release() -> None:
    assert _released_count({1: {
        "active_probe_trace": [{
            "probe_decision": True,
            "probe_conditioned_visual_margin": 0.2,
            "probe_triggered": False,
        }],
    }}) == 0


def test_selector_treats_task_safety_probe_skip_as_zero_release() -> None:
    assert _released_count({1: {
        "active_probe_trace": [
            {
                "probe_decision": True,
                "probe_conditioned_visual_margin": 0.01,
                "probe_triggered": True,
            },
            {
                "probe_decision": True,
                "probe_skipped": True,
                "reason": "no_task_safe_probe_under_calibration_dynamics",
            },
        ],
    }}) == 0


def test_selector_rejects_an_abstention_that_changes_physical_actions() -> None:
    baseline = {1: {"trace": [{"selected_action": [0.1, 0.0]}], "success": True, "final_distance": 0.1}}
    candidate = {1: {
        "active_probe_trace": [{
            "probe_decision": True,
            "probe_conditioned_visual_margin": 0.2,
            "probe_triggered": False,
        }],
        "trace": [{"selected_action": [0.2, 0.0]}], "success": True, "final_distance": 0.1,
    }}
    import pytest
    with pytest.raises(ValueError, match="different physical action"):
        validate_abstain_equivalence(baseline, candidate)
