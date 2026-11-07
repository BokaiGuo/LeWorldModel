import pytest

from experiments.pc_pusht_shared_prefix_physical_identity_20261107_codex import physical_differences


def test_detects_packet_discrepancy():
    base = [[0.0] * 7, [1.0] * 7]
    other = [[0.0] * 7, [1.0] * 7]
    other[1][3] += 0.25
    candidates = [{"packet_states": base}, {"packet_states": other}]
    trace = [{"observed_state": [0.0] * 7} for _ in range(10)]
    trace[9]["observed_state"] = [1.0] * 7
    result = physical_differences(candidates, 0, trace)
    assert result["candidate_count_exceeding_1e_4_by_packet"] == [0, 1]
    assert result["max_candidate_state_abs_error_by_packet"] == [0.0, 0.25]
    assert result["selected_online_state_abs_error_by_packet"] == [0.0, 0.0]


def test_rejects_invalid_state_dimension():
    with pytest.raises(ValueError, match="packet states"):
        physical_differences([{"packet_states": [[0.0] * 6] * 2}], 0, [])
