from pathlib import Path

from experiments.empirical_spectral_audit.oral_experiment_coverage_audit import (
    _design,
    _metric_matrix,
)
from experiments.empirical_spectral_audit.run_oral_level_unified_diagnostic_audit import (
    _normalise_panel_row,
)


def _payload(environment="PointMaze", architecture="a", seed="1", stage=1):
    return {
        "protocol": {
            "environment": environment,
            "prediction_quality": {
                "action_distribution": "shared",
                "fit_horizon": 1,
            },
        },
        "record": {
            "architecture": architecture,
            "checkpoint": f"/tmp/{architecture}/seed_{seed}/epoch_{stage}.pt",
            "group": f"{environment}/test/{architecture}/seed_{seed}",
            "model_factors": {
                "action_distribution": "shared",
                "latent_dimension": 4,
                "training_stage": stage,
            },
            "diagnostics": {
                "prediction_mse": 1.0,
                "latent_variance": 2.0,
                "action_inverse_risk": 0.1,
                "task_metric_alignment": 0.2,
                "task_metric_commutator": 0.3,
                "task_metric_schur_offdiagonal_energy": 0.4,
                "task_metric_transient_amplification": 1.2,
                "predictor_spectrum": [{"real": 0.5, "imag": 0.0}],
            },
            "heldout": {"planning_loss": 0.7},
        },
    }


def test_design_keeps_missing_environment_and_horizon_gates_explicit():
    rows = [_normalise_panel_row(_payload(), Path("record.json"))]
    report = _design(rows)
    assert report["checks"]["architectures"] is False
    assert report["checks"]["environments"] is False
    assert report["checks"]["horizons"] is False
    assert report["status"] == "blocked"


def test_metric_matrix_recognises_action_inverse_risk_and_latent_variance():
    payload = _payload()
    payload["_audit_record_path"] = "record.json"
    row = _normalise_panel_row(payload, Path("record.json"))
    matrix = _metric_matrix([payload], [row])
    fields = matrix["all_records"]["required_diagnostics"]
    assert fields["atm_action_inverse_risk"]["present"] == 1
    assert fields["latent_variance"]["present"] == 1
    assert fields["planning_risk"]["present"] == 1
