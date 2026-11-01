import copy

import pytest

from experiments.spectral_reachability.d2p_stage_admission import (
    P0_INTEGRITY_KEYS, p0_predict_samples_matched, validate_p0_admission_report,
)

GATE = "actual_planning_predict_samples_matched_to_cem"


def rows():
    result = []
    for index in range(2):
        root, goal = f"{index+1:064x}", "a"*64
        methods = {}
        for name in ("d2p_goal_progress", "cem"):
            methods[name] = {
                "execution_identity": {"reset_rgb_sha256": root, "goal_rgb_sha256": goal},
                "receipt": {
                    "predict_accounting_scope": "planner_only_excludes_post_execution_diagnostic_encoder_and_backward",
                    "actual_planning_predict": {"phase": "planning", "method": name,
                        "horizon_tokens": 2, "predict_calls": 4, "predict_samples": 32,
                        "context_lengths": {"3": 4}, "failed_predict_calls": 0, "completed": True}}}
        result.append({"seed": 100+index, "root_index": index, "split": "outcome",
                       "executed_context": {"root_rgb_sha256": root, "goal_rgb_sha256": goal},
                       "methods": methods})
    return result


def report():
    return {"schema": "d2p_reacher_p0_goal_progress.v1", "stage_gate": {"p1_unlocked": True},
            "protocol": {"calibration_only": False}, "rows": rows(),
            "comparisons_outcome_only": {"cem": {"ci95": [-2, -1]}},
            "gates": dict.fromkeys(P0_INTEGRITY_KEYS, True)}


def test_valid_and_scope_is_samples_not_calls_or_total_compute():
    value = rows()
    value[0]["methods"]["cem"]["receipt"]["actual_planning_predict"]["predict_calls"] = 2
    value[0]["methods"]["cem"]["receipt"]["actual_planning_predict"]["context_lengths"] = {3: 2}
    assert p0_predict_samples_matched(value)
    assert p0_predict_samples_matched(list(reversed(value)))
    assert validate_p0_admission_report(report())


@pytest.mark.parametrize("field,value", [
    ("completed", False), ("completed", 1), ("failed_predict_calls", 1),
    ("failed_predict_calls", False), ("phase", "evaluation"), ("method", "other"),
    ("horizon_tokens", 3), ("horizon_tokens", True), ("horizon_tokens", 0),
    ("predict_samples", 31), ("predict_samples", 32.0), ("predict_samples", True),
    ("predict_samples", float("nan")), ("predict_samples", 0),
    ("predict_calls", 0), ("predict_calls", 33), ("error", "failure"),
    ("errors", ["failure"]), ("exception", "RuntimeError"),
])
def test_bad_actual_receipt(field, value):
    panel = rows()
    panel[0]["methods"]["cem"]["receipt"]["actual_planning_predict"][field] = value
    assert not p0_predict_samples_matched(panel)


@pytest.mark.parametrize("field", ["completed", "failed_predict_calls", "predict_samples", "horizon_tokens"])
def test_missing_actual_fields(field):
    panel = rows()
    del panel[0]["methods"]["cem"]["receipt"]["actual_planning_predict"][field]
    assert not p0_predict_samples_matched(panel)


@pytest.mark.parametrize("mutation", ["seed", "root", "index", "execution", "goal", "scope", "legacy", "split"])
def test_identity_and_receipt_fail_closed(mutation):
    panel = rows()
    if mutation in ("seed", "index"):
        key = "seed" if mutation == "seed" else "root_index"
        panel[1][key] = panel[0][key]
    elif mutation == "root":
        panel[1]["executed_context"]["root_rgb_sha256"] = panel[0]["executed_context"]["root_rgb_sha256"]
    elif mutation in ("execution", "goal"):
        key = "reset_rgb_sha256" if mutation == "execution" else "goal_rgb_sha256"
        panel[0]["methods"]["cem"]["execution_identity"][key] = "b"*64
    elif mutation == "scope":
        panel[0]["methods"]["cem"]["receipt"]["predict_accounting_scope"] = "totalcompute"
    elif mutation == "legacy":
        panel[0]["methods"]["cem"]["receipt"] = {"forward_rollout_evaluations": 32}
    else:
        panel[0]["split"] = "unknown"
    assert not p0_predict_samples_matched(panel)


@pytest.mark.parametrize("value", [None, [], {}, [None]])
def test_no_outcomes_or_malformed(value):
    assert not p0_predict_samples_matched(value)


def test_calibration_excluded_but_duplicate_identity_rejected():
    panel = rows()
    panel[0]["split"] = "calibration"
    panel[0]["methods"] = {}
    assert p0_predict_samples_matched(panel)
    panel[1]["split"] = "calibration"
    assert not p0_predict_samples_matched(panel)


def test_forged_gate_cannot_bypass_receipts_and_gate_required():
    value = report()
    value["rows"][0]["methods"]["cem"]["receipt"]["actual_planning_predict"]["predict_samples"] = 33
    with pytest.raises(ValueError, match="predict samples"):
        validate_p0_admission_report(value)
    value = report()
    del value["gates"][GATE]
    with pytest.raises(ValueError):
        validate_p0_admission_report(value)


def test_efficacy_and_existing_integrity_not_weakened():
    original = report()
    for key in P0_INTEGRITY_KEYS:
        value = copy.deepcopy(original)
        value["gates"][key] = False
        with pytest.raises(ValueError):
            validate_p0_admission_report(value)


@pytest.mark.parametrize("histogram", [None, {}, {"3": 3}, {"3": True}, {"0": 4},
                                     {"03": 4}, {3: 2, "3": 2}, {False: 4}, {"3": -4}])
def test_inconsistent_actual_call_histogram_rejected(histogram):
    panel = rows()
    panel[0]["methods"]["cem"]["receipt"]["actual_planning_predict"]["context_lengths"] = histogram
    assert not p0_predict_samples_matched(panel)


@pytest.mark.parametrize("level", ["row", "method", "receipt"])
def test_outer_error_cannot_be_hidden_by_completed_actual_receipt(level):
    panel = rows()
    target = panel[0]
    if level != "row":
        target = target["methods"]["cem"]
    if level == "receipt":
        target = target["receipt"]
    target["error"] = "planning failed"
    assert not p0_predict_samples_matched(panel)


@pytest.mark.parametrize("value", [None, [], {"protocol": None}, {"stage_gate": None}])
def test_malformed_report_rejected_as_value_error(value):
    with pytest.raises(ValueError):
        validate_p0_admission_report(value)


def test_equal_panel_totals_do_not_hide_per_root_mismatch():
    panel = rows()
    for row, samples in zip(panel, (31, 33)):
        row["methods"]["cem"]["receipt"]["actual_planning_predict"]["predict_samples"] = samples
    assert not p0_predict_samples_matched(panel)


def test_horizon_cannot_change_between_matched_pairs():
    panel = rows()
    for method in panel[1]["methods"].values():
        method["receipt"]["actual_planning_predict"]["horizon_tokens"] = 3
    assert not p0_predict_samples_matched(panel)


def test_resource_match_does_not_override_efficacy_gate():
    for ci in ([0, 0], [-1, 0], [-1, 1]):
        value = report()
        value["comparisons_outcome_only"]["cem"]["ci95"] = ci
        with pytest.raises(ValueError):
            validate_p0_admission_report(value)
