import pytest

from experiments.pc_tworoom_directional_replay_contract_20261107_codex import compare_trace, validate_window


def test_trace_checks_strict_success_boundary_and_post_success():
    saved = [
        {"distance": 15.0, "terminated": True, "truncated": False},
        {"distance": 16.0, "terminated": False, "truncated": False},
    ]
    result = compare_trace(saved, saved)
    assert result["ever_success"] is True
    assert result["final_success"] is False
    assert result["lost_success"] is True
    assert result["max_distance_error"] == 0.0


def test_trace_rejects_mismatched_flags_and_lengths():
    valid = [{"distance": 16.0, "terminated": False, "truncated": False}]
    with pytest.raises(ValueError, match="success"):
        compare_trace(valid, [{"distance": 16.0, "terminated": True, "truncated": False}])
    with pytest.raises(ValueError, match="length"):
        compare_trace(valid, [])


def test_window_rejects_episode_and_clock_discontinuities():
    validate_window([4, 4, 4], [9, 10, 11])
    with pytest.raises(ValueError, match="episode"):
        validate_window([4, 5, 5], [9, 10, 11])
    with pytest.raises(ValueError, match="clock"):
        validate_window([4, 4, 4], [9, 11, 12])
