from experiments.empirical_spectral_audit.oral_evidence_protocol import (
    nested_incremental_risk_audit,
    pusht_mechanism_protocol,
    two_stage_selector,
    validate_learned_checkpoint_panel,
)


def _row(architecture, environment, seed, stage, mse, risk, alignment):
    return {
        "architecture": architecture, "environment": environment, "seed": seed,
        "training_stage": stage, "latent_dimension": 8, "prediction_mse": mse,
        "atm_action_inverse_risk": 0.1, "predictor_spectrum": [0.8, 0.5],
        "task_metric": "task_xy", "planning_risk": risk, "alignment": alignment,
        "commutator": alignment, "schur": alignment, "transient": alignment,
        "horizons": [1, 2, 4, 8, 16],
    }


def test_panel_gate_blocks_small_panel():
    report = validate_learned_checkpoint_panel([_row("a", "e", 1, 1, 1.0, 0.2, 0.1)])
    assert report["status"] == "blocked"


def test_two_stage_selector_is_mse_filtered_and_not_test_tuned():
    records = [_row("a", "e", 1, 1, 1.00, 0.5, 0.8), _row("b", "e", 2, 1, 1.02, 0.2, 0.1)]
    result = two_stage_selector(records, mse_tolerance=0.05)
    assert result["status"] == "pass"
    assert result["selector"] == "MSE-filtered-geometry"
    assert result["test_outcome_used_for_selection"] is False


def test_nested_risk_audit_has_leave_one_environment_out_folds():
    records = []
    for env in ("e1", "e2"):
        for index in range(4):
            records.append(_row("a" if index % 2 else "b", env, index, 1, 1.0, float(index), float(index)))
    result = nested_incremental_risk_audit(records)
    assert result["n_folds"] == 2
    assert "leave-one-environment-out" in result["model_specification"]


def test_pusht_protocol_contains_all_mechanism_stages():
    protocol = pusht_mechanism_protocol()
    assert protocol["stages"][0] == "history"
    assert "angle_error" in protocol["required_metrics"]
