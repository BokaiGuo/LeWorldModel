import json

from experiments.gdd_wm.audit_repairability_gate import audit, run


def _report(*, crossfit: bool, selected: str, shuffled_mean: float = 1.0) -> dict:
    return {
        "calibration_selection": {"safe_gate": {"passed": True}},
        "nonlinear_calibration_selection": {
            "bilinear_train_crossfit_gate": {
                "full_dynamics_gate_passed": crossfit,
                "selected": selected,
            }
        },
        "metrics": {
            "calibration_safe_gate": {
                "vs_action_only": {"regret_delta": {"mean": -1.0, "ci95": [-2.0, -0.1]}}
            },
            "listwise_shuffled_prediction": {
                "vs_action_only": {"regret_delta": {"mean": shuffled_mean, "ci95": [0.0, 2.0]}}
            },
        },
    }


def test_gate_requires_predicted_dynamics_crossfit(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_report(crossfit=False, selected="action_only")))
    result = audit(path)
    assert result["status"] == "fallback_action_only"
    assert result["authorization"]["calibration_non_degradation_gate"] is True


def test_gate_permits_only_full_predicted_dynamics(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_report(crossfit=True, selected="bilinear_full_predicted_dynamics")))
    result = audit(path)
    assert result["status"] == "permitted_predicted_dynamics"


def test_held_out_shuffle_is_diagnostic_only(tmp_path) -> None:
    path = tmp_path / "audit.json"
    output = tmp_path / "panel.json"
    path.write_text(json.dumps(_report(crossfit=True, selected="bilinear_full_predicted_dynamics", shuffled_mean=-2.0)))
    result = run([path], output)
    assert result["audits"][0]["status"] == "permitted_predicted_dynamics"
    assert result["audits"][0]["held_out_diagnostic_only"]["shuffled_control_better_than_selected"] is True
    assert output.exists()
