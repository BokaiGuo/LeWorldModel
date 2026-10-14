from experiments.empirical_spectral_audit.decision_sufficiency import assess_panel, assess_record


def _record(**overrides):
    record = {
        "predictive": {"baseline_error": 1.0, "method_error": 0.8},
        "action": {
            "baseline_ranking_regret": 0.4,
            "method_ranking_regret": 0.2,
            "baseline_mode_accuracy": 0.7,
            "method_mode_accuracy": 0.9,
        },
        "decision": {
            "baseline_planning_risk": 0.5,
            "method_planning_risk": 0.3,
            "baseline_pose_success": 0.2,
            "method_pose_success": 0.4,
        },
    }
    for section, values in overrides.items():
        record[section].update(values)
    return record


def test_all_three_layers_are_required_for_positive_claim():
    report = assess_record(_record())
    assert report["status"] == "three_layer_positive"
    assert report["positive_claim_permitted"] is True


def test_prediction_gain_without_ranking_gain_is_hard_boundary():
    report = assess_record(_record(action={"method_ranking_regret": 0.5}))
    assert report["status"] == "boundary_or_negative"
    assert "prediction_improvement_without_action_ranking_improvement" in report["diagnostic_boundary_flags"]
    assert report["positive_claim_permitted"] is False


def test_mode_gain_without_pose_closure_is_not_pose_success():
    report = assess_record(_record(decision={"method_pose_success": 0.1}))
    assert "mode_accuracy_improvement_without_pose_closure" in report["diagnostic_boundary_flags"]
    assert report["positive_claim_permitted"] is False


def test_missing_independent_consequence_fails_closed():
    record = _record()
    del record["decision"]["method_planning_risk"]
    report = assess_record(record)
    assert report["status"] == "incomplete_fail_closed"
    assert "decision.method_planning_risk" in report["missing_fields"]


def test_panel_does_not_average_a_failed_record_away():
    report = assess_panel([_record(), _record(action={"method_ranking_regret": 0.5})])
    assert report["status"] == "gated_panel"
    assert report["three_layer_positive_count"] == 1
    assert report["positive_claim_permitted"] is False
