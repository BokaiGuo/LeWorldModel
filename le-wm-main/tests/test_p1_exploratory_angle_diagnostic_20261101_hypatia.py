import math

from experiments.spectral_reachability import p1_exploratory_angle_diagnostic_20261101_hypatia as mod


def test_native_can_be_negative_but_wrapped_is_positive():
    goal = [0, 0, 0, 0, -1.5, 0, 0]
    state = [0, 0, 0, 0, 5.5, 0, 0]
    value = mod.metrics(goal, state)
    assert value["native_angle_expression"] < 0
    assert value["strict_wrapped_absolute_angle"] > math.pi / 9
    assert value["native_formula_success"] is True
    assert value["wrapped_diagnostic_success"] is False


def test_saved_data_remains_censored():
    result = mod.build()
    assert len(result["values"]) == 2
    assert all(x["completed_steps"] == 0 for x in result["observed_slot_statuses"])
    assert result["freshness"] == "unknown"
