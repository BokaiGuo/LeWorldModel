import math

import pytest

from experiments.spectral_reachability import p1_dev303_v3_cross_tail_object_analysis_20261102_exclusive as a


def test_periodic_angle_and_object_not_pusher():
    value = a.object_metrics([100, 200, 3, 4, 2 * math.pi - 0.1],
                             [0, 0, 0, 0, 0.1])
    assert value["goal_xy_error"] == 5
    assert value["goal_angle_error_rad"] == pytest.approx(0.2)
    assert value["goal_signed_angle_error_rad"] == pytest.approx(-0.2)


def row(arm, count, status, *, terminated=False):
    state = [0, 0, 1, 0, 0]
    tail = [{"phase": "tail", "step": i, "state": state,
             "terminated": terminated and i == count, "truncated": False}
            for i in range(1, count + 1)]
    obs = [{"phase": "prefix", "step": 5, "state": state}]
    if count == 10:
        obs.append({"phase": "tail", "step": 10, "state": state})
    return {"arm": arm, "status": status, "observations": obs, "transitions": tail}


def test_early_stop_is_censored_not_padded_and_no_delta():
    specs = [{"arm": "x", "prefix_role": "p", "tail_role": "x"},
             {"arm": "z", "prefix_role": "p", "tail_role": "zero"}]
    rows = [row("x", 3, "native_early_stop", terminated=True),
            row("z", 10, "fixed_horizon_completed")]
    x, z = a.summarize(specs, rows, [0] * 5)
    assert x["native_success_observed"] is True
    assert x["censored_horizon10"] and x["step10_object"] is None
    assert x["relative_to_zero"] is None
    assert z["relative_to_zero"]["goal_xy_error_difference"] == 0


def test_native_stop_at_tenth_step_observes_horizon():
    specs = [{"arm": "z", "prefix_role": "p", "tail_role": "zero"}]
    item = a.summarize(specs, [row("z", 10, "native_early_stop", terminated=True)], [0] * 5)[0]
    assert item["native_success_observed"] is True
    assert item["horizon10_observed"] is True
    assert item["censored_horizon10"] is False


def test_failed_tenth_step_is_not_certified():
    specs = [{"arm": "z", "prefix_role": "p", "tail_role": "zero"}]
    item = a.summarize(specs, [row("z", 10, "failed")], [0] * 5)[0]
    assert item["step10_object"] is None and item["native_success_observed"] is None


def test_real_saved_twenty_branches():
    result = a.analyze()
    assert len(result["rows"]) == 20
    assert all(not row["censored_horizon10"] for row in result["rows"])
    assert result["P1_admission"] is False
