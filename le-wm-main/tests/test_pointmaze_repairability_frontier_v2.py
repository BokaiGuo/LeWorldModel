import numpy as np

from experiments.gdd_wm import run_pointmaze_repairability_frontier_v2 as v2


def test_regularized_router_standardizes_and_returns_probabilities():
    predict, detail = v2._fit_logistic(np.asarray([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]), np.asarray([0.0, 0.0, 1.0, 1.0]), l2=1.0)
    probability = predict(np.asarray([[0.5, 5.0], [2.5, 25.0]]))
    assert detail["l2"] == 1.0
    assert probability.shape == (2,)
    assert 0.0 <= probability[0] < probability[1] <= 1.0


def test_router_decision_is_pre_outcome_selection_and_can_retain_psd():
    groups = {"r": [{"candidate_id": "a"}, {"candidate_id": "b"}]}
    scores = {"r": {"psd": np.asarray([3.0, 1.0]), "mlp": np.asarray([1.0, 3.0]), "naive": np.asarray([3.0, 1.0])}}
    result = v2._router_decision(groups, scores, {"r": np.asarray([0.0, 2.0])}, {"r": 0.1}, threshold=0.5)
    row = result["rows"]["r"]
    assert row["router_selected_mlp_pre_outcome"] is False
    assert row["routed_regret"] == row["psd_regret"] == 0.0
    assert result["outcome_cost_used_for_selection"] is False


def test_summary_declares_no_gain_when_router_is_worse_than_psd():
    decision = {"outcome_cost_used_for_selection": False, "rows": {"r": {"psd_regret": 0.0, "mlp_regret": 0.0, "naive_regret": 0.0, "routed_regret": 0.1, "repair_outcome_mlp_beats_psd": True, "router_selected_mlp_pre_outcome": True}}}
    summary = v2._summarize(decision)
    assert summary["preselection_prediction_verdict"]["status"] == "not_confirmed_in_this_pilot"


def test_diagnosis_separates_rare_repairs_support_failure_and_calibrated_abstention():
    result = v2._diagnose(
        {"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0},
        {"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0},
        {"action_support": {"gdd_auc": 0.5}},
        {"gdd_auc": 0.5},
        {"router_coverage": 0.0},
        gdd_root_count=20,
    )
    assert "repair_events_are_rare_in_cross_fitted_train_roots" not in result["diagnostic_signals"]  # 25%, not rare
    assert "action_bank_support_alone_has_no_GDD_repair_discrimination" in result["diagnostic_signals"]
    assert "calibration_supports_abstention_only_not_MLP_release" in result["diagnostic_signals"]


def test_selected_calibration_threshold_is_recomputed_not_left_at_last_grid_value():
    groups = {"root": [{"candidate_id": 0}, {"candidate_id": 1}]}
    scores = {"root": {"psd": np.asarray([1.0, 3.0]), "mlp": np.asarray([3.0, 1.0]), "naive": np.asarray([1.0, 3.0])}}
    costs = {"root": np.asarray([0.0, 2.0])}
    probability = {"root": 0.75}
    threshold, regret, selected = v2._select_calibration_threshold(groups, scores, costs, probability)
    assert threshold == 0.75
    assert regret == 0.0
    assert v2._summarize(selected)["router_coverage"] == 1.0


def test_calibration_frontier_distinguishes_no_release_from_no_mlp_psd_effect():
    groups = {"root": [{"candidate_id": 0}, {"candidate_id": 1}]}
    scores = {"root": {
        "psd": np.asarray([1.0, 3.0]),
        "mlp": np.asarray([3.0, 1.0]),
        "naive": np.asarray([1.0, 3.0]),
    }}
    costs = {"root": np.asarray([0.0, 2.0])}
    frontier = v2._calibration_threshold_frontier(
        groups, scores, costs, {"root": 0.75},
    )
    released = next(row for row in frontier if row["threshold"] == 0.75)
    abstained = next(row for row in frontier if row["threshold"] == 1.0)
    assert released["router_coverage"] == 1.0
    assert released["routed_minus_psd_regret"] < 0.0
    assert abstained["router_coverage"] == 0.0
    assert v2._public_calibration_frontier(frontier)[0].get("decision") is None


def test_v2_audit_rejects_missing_deferred_outcome_contract(tmp_path):
    from experiments.gdd_wm.audit_pointmaze_repairability_frontier_v2 import audit

    report = tmp_path / "bad.json"
    report.write_text("{}", encoding="utf-8")
    assert audit(report)["status"] == "rejected_contract_audit"
