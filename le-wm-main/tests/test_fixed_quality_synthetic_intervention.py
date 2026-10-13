from experiments.empirical_spectral_audit.run_fixed_quality_synthetic_intervention import make_panel


def test_fixed_quality_intervention_keeps_nuisance_factors_fixed_and_exposes_adversary():
    report = make_panel(repeats=4, seed=7)
    assert report["support"]["fixed_spectrum"]
    assert report["support"]["fixed_prediction_mse"]
    assert report["support"]["fixed_atm_action_inverse_risk"]
    assert len(report["panel"]) == len(report["adversarial_panel"])
    assert {row["task_metric"] for row in report["panel"]} == {"task_xy", "task_angle", "task_balanced"}
    assert report["selector_regret"]["status"] == "pass"
    assert report["adversarial_selector_regret"]["status"] == "pass"
    assert report["selector_regret"]["selection_outcome_leakage"] is False
