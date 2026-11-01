import math

import pytest

from experiments.spectral_reachability import codex_p1_h10_paired_table_20261030_main as table


def test_metrics_keep_pusher_joint_and_object_separate():
    goal = [0., 0., 0., 0., 0., 0., 0.]
    state = [30., 0., 0., 0., 0., 0., 0.]
    metrics = table.physical_metrics(state, goal)
    assert metrics["object_position_error"] == 0
    assert metrics["pusher_position_error"] == 30
    assert metrics["native_joint_xy_error"] == 30
    assert metrics["native_success"] is False
    assert table.physical_metrics(None, goal) is None


def test_native_and_wrapped_angle_are_separate():
    goal = [0.] * 7
    state = [0., 0., 0., 0., 2 * math.pi + 0.2, 0., 0.]
    metrics = table.physical_metrics(state, goal)
    assert metrics["object_angle_error_wrapped"] == pytest.approx(0.2)
    assert metrics["native_angle_term"] == pytest.approx(-0.2)


def test_rejects_missing_full_panel():
    with pytest.raises(ValueError):
        table.build({"anchors": []}, {"groups": []}, [])


def test_paired_difference_preserves_missing_and_metric_sign():
    goal = [0.] * 7
    first = table.physical_metrics([10., 0., 4., 0., 0., 0., 0.], goal)
    second = table.physical_metrics([20., 0., 7., 0., 0., 0., 0.], goal)
    result = table.paired_difference(first, second)
    assert result["object_position_error"] == -3
    assert result["pusher_position_error"] == -10
    assert table.paired_difference(first, None) is None
    assert table.paired_difference(None, second) is None


def test_full_roster_pairs_responses_and_keeps_censored_null():
    zero = [0.] * 7
    anchors, groups, states = [], [], []
    for group in range(8):
        for role in table.ROLES:
            anchors.append({"role": role, "root_state": zero, "goal": zero,
                            "checkpoint": {"state": [1., 0., 0., 0., 0., 0., 0.]}})
            states.append([[0., 0., 3. if role == "selected" else 7., 0., 0., 0., 0.]
                           for _ in range(42)])
        probes = []
        for j in range(42):
            response = {"object_response_xy": [1., 0.], "object_response_angle": 0.,
                        "position_error_improvement": 2., "angle_error_improvement": 0.}
            roles = {role: {"status": "probe_completed", "metrics": {"endpoint": 1},
                            "response_vs_own_zero": response}
                     for role in table.ROLES}
            probes.append({"index": j, "probe": str(j), "raw10_sha256": "a" * 64, "roles": roles})
        groups.append({"logical_pair": group, "root_slot": 0, "condition": "fixed",
                       "duplicate_group": group, "arm": "fixed", "probes": probes})
    states[0][1] = None
    groups[0]["probes"][1]["roles"]["selected"].update(
        status="censored_out_of_domain", metrics=None, response_vs_own_zero=None)
    result = table.build({"anchors": anchors}, {"groups": groups, "status_counts": {}}, states)
    assert len(result["rows"]) == 1008
    assert result["rows"][0]["prefix_pusher_position_error_change"] == 1
    assert result["rows"][0]["prefix_native_angle_term_change"] == 0
    assert result["rows"][0]["prefix_native_success_transition"] == {"root": True, "prefix": True}
    assert result["rows"][0]["same_suffix_endpoint_minus_zero"]["object_position_error"] == -4
    assert result["rows"][0]["same_prefix_endpoint_minus_own_zero_suffix"]["object_position_error"] == 0
    assert result["rows"][0]["response_vs_own_zero_suffix_minus_zero_prefix"]["position_error_improvement"] == 0
    assert result["rows"][3]["same_suffix_endpoint_minus_zero"] is None
    assert result["rows"][3]["same_prefix_endpoint_minus_own_zero_suffix"] is None
    assert result["rows"][3]["independently_optimized_suffix"] is None
    anchors[1]["root_state"] = [0., 1., 0., 0., 0., 0., 0.]
    with pytest.raises(ValueError, match="paired root state and goal"):
        table.build({"anchors": anchors}, {"groups": groups, "status_counts": {}}, states)
