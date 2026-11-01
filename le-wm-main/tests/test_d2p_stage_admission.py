import pytest

from experiments.spectral_reachability.d2p_stage_admission_hardened_20261101_main import P0_INTEGRITY_KEYS, p1_admission
from experiments.spectral_reachability.d2p_stage_admission_hardened_20261101_main import P1_INTEGRITY_KEYS, P1_MECHANISM_KEYS, p2_admission


def gates():
    return {key: True for key in P0_INTEGRITY_KEYS}


def test_strict_affirmative_outcome_required():
    assert p1_admission({"ci95": [-.5, -.1]}, calibration_only=False, implementation_gates=gates())[0]
    for interval in ([0, 0], [-.1, 0], [-.1, .1], [.1, .5]):
        assert not p1_admission({"ci95": interval}, calibration_only=False, implementation_gates=gates())[0]


@pytest.mark.parametrize("key", P0_INTEGRITY_KEYS)
def test_every_failed_integrity_key_closes_gate(key):
    value = gates()
    value[key] = False
    assert not p1_admission({"ci95": [-.5, -.1]}, calibration_only=False, implementation_gates=value)[0]
    del value[key]
    assert not p1_admission({"ci95": [-.5, -.1]}, calibration_only=False, implementation_gates=value)[0]


@pytest.mark.parametrize("interval", (None, [], [1], [float("nan"), -1], [-2, -float("inf")], [-1, -2], ["invalid", 0]))
def test_malformed_interval_closes_gate(interval):
    assert not p1_admission({"ci95": interval}, calibration_only=False, implementation_gates=gates())[0]


def test_missing_receipt_or_calibration_cannot_admit():
    assert not p1_admission(None, calibration_only=False, implementation_gates=gates())[0]
    assert not p1_admission({"ci95": [-.5, -.1]}, calibration_only=True, implementation_gates=gates())[0]
    assert not p1_admission({"ci95": [-.5, -.1]}, calibration_only=False, implementation_gates=None)[0]


@pytest.mark.parametrize("value", [None, 0, "", []])
def test_unknown_calibration_flag_never_admits(value):
    assert not p1_admission({"ci95": [-.5, -.1]}, calibration_only=value, implementation_gates=gates())[0]
    args = p2_arguments()
    args["calibration_only"] = value
    assert not p2_admission(**args)[0]


def p2_arguments():
    controls = ("zero_prefix_two_stage_d2p", "greedy_two_stage_d2p", "cem_full_horizon")
    return {"comparisons": {name: {"ci95": [-.5, -.1]} for name in controls},
        "calibration_only": False, "eligible_roots": 8, "min_eligible_roots": 4,
        "implementation_gates": {key: True for key in (*P1_INTEGRITY_KEYS, P1_MECHANISM_KEYS[0])},
        "required_comparisons": controls, "mechanism_gate": P1_MECHANISM_KEYS[0]}


def test_strict_p1_outcome_can_admit_p2_but_zero_touch_cannot():
    args = p2_arguments()
    assert p2_admission(**args)[0]
    for interval in ([0, 0], [-.1, 0], [-.1, .1], [float("nan"), -1]):
        args["comparisons"]["cem_full_horizon"]["ci95"] = interval
        assert not p2_admission(**args)[0]


@pytest.mark.parametrize("key", (*P1_INTEGRITY_KEYS, P1_MECHANISM_KEYS[0]))
def test_missing_p1_integrity_or_mechanism_closes_p2(key):
    args = p2_arguments()
    del args["implementation_gates"][key]
    assert not p2_admission(**args)[0]


def test_p2_requires_every_predeclared_control_and_no_calibration_efficacy():
    args = p2_arguments()
    args["calibration_only"] = True
    assert not p2_admission(**args)[0]
    args["calibration_only"] = False
    args["eligible_roots"] = 3
    assert not p2_admission(**args)[0]
    args["eligible_roots"] = 8
    del args["comparisons"]["cem_full_horizon"]
    assert not p2_admission(**args)[0]
    args["required_comparisons"] = ("zero_prefix_two_stage_d2p",)
    assert not p2_admission(**args)[0]


def test_source_matched_additional_comparison_cannot_be_omitted_or_tied():
    args = p2_arguments()
    args["required_comparisons"] += ("source_matched_greedy_two_stage_d2p",)
    assert not p2_admission(**args)[0]
    args["comparisons"]["source_matched_greedy_two_stage_d2p"] = {"ci95": [0, 0]}
    assert not p2_admission(**args)[0]


@pytest.mark.parametrize("field", ["eligible_roots", "min_eligible_roots"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 4.0, True, None, "4"])
def test_p2_rejects_non_integer_root_counts(field, value):
    args = p2_arguments()
    args[field] = value
    assert not p2_admission(**args)[0]


@pytest.mark.parametrize("value", [None, "cem_full_horizon", [None], [{"bad": "value"}]])
def test_p2_rejects_malformed_required_comparisons(value):
    args = p2_arguments()
    args["required_comparisons"] = value
    assert not p2_admission(**args)[0]
