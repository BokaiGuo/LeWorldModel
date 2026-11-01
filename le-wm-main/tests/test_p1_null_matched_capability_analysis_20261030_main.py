import pytest

from experiments.spectral_reachability.p1_null_matched_capability_analysis_20261030_main import analyze


def fixtures():
    anchors, rows = [], []
    for i in range(16):
        anchors.append(dict(index=i, root_slot=i//4, condition="on", arm=f"arm{i//2}",
                            role="selected_prefix" if i % 2 == 0 else "zero_prefix",
                            task_id=str(i//2), physical_root_sha256=str(i//2),
                            prefix5_float32_hex=str(i % 2),
                            root_state=[0]*7, goal=[0, 0, 1, 0, 0, 0, 0]))
        rows.append(dict(anchor=i, endpoints=[[0]*7 for _ in range(21)]))
    return anchors, dict(rows=rows)


def test_fixed_pairs_and_missing_endpoint():
    anchors, verified = fixtures()
    verified["rows"][0]["endpoints"][1] = None
    rows, pairs = analyze(anchors, verified)
    assert rows[0]["position_singular_values"] is None
    assert pairs[0]["selected_minus_zero_angle_response_l2"] is None
    assert len(pairs) == 8
    assert len({p["duplicate_group"] for p in pairs}) == 8


def test_rejects_mismatched_goal():
    anchors, verified = fixtures()
    anchors[1]["goal"][2] = 3
    with pytest.raises(ValueError, match="pair source"):
        analyze(anchors, verified)


def test_same_probe_differences_and_censoring():
    anchors, verified = fixtures()
    verified["rows"][0]["endpoints"][1][2] = 0.5
    verified["rows"][1]["endpoints"][1][2] = 0.2
    verified["rows"][0]["endpoints"][3] = None
    _, pairs = analyze(anchors, verified)
    pair = pairs[0]
    assert pair["selected_minus_zero_zero_relative_position_error_gains"][1] == pytest.approx(0.3)
    assert pair["selected_minus_zero_zero_relative_position_error_gains"][3] is None
    assert pair["selected_minus_zero_position_goal_response"][0] == pytest.approx(0.6)
    assert pair["selected_minus_zero_position_goal_response"][1] is None


def test_rejects_mismatched_root_state():
    anchors, verified = fixtures()
    anchors[1]["root_state"][2] = 0.1
    with pytest.raises(ValueError, match="pair source"):
        analyze(anchors, verified)
