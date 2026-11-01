"""Contrast-definition tests only; no neural experiment."""
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_waypoint_reference_margins import summarize


def fixture():
    return {"root_seeds": list(range(8)), "rows": [{"seed": i, "original_planned_prefix_margin": .1,
        "prefix_observed_costs": {"native_dispatch_replay": 2., "native_dispatch_tracking": 3., "zero_replay": 2.5, "zero_tracking": 4.}} for i in range(8)]}


def test_feedback_zero_cannot_redefine_fixed_neutral_condition():
    value = fixture()
    expected = summarize(value)
    for row in expected["rows"]:
        assert row["tracking_minus_fixed_neutral_margin"] == .5
        assert row["tracking_minus_tracking_zero_margin"] == -1.
        assert row["same_action_model_fidelity_comparison_valid"] is False
    value["rows"].reverse()
    assert summarize(value) == expected


@pytest.mark.parametrize("kind", ["partial", "duplicate", "nan", "negative"])
def test_invalid_contract_rejected(kind):
    value = fixture()
    if kind == "partial":
        value["rows"].pop()
    elif kind == "duplicate":
        value["rows"][-1]["seed"] = 0
    else:
        value["rows"][0]["prefix_observed_costs"]["zero_replay"] = float("nan") if kind == "nan" else -1.
    with pytest.raises(ValueError):
        summarize(value)
