import json
from pathlib import Path

from experiments.empirical_spectral_audit.run_oral_level_unified_diagnostic_audit import (
    _coverage,
    _normalise_panel_row,
    run_audit,
)


def _payload(arch: str, seed: int, stage: int, env: str = "PushT") -> dict:
    return {
        "record_schema": "real_checkpoint_selection.v1",
        "protocol": {
            "environment": env,
            "prediction_quality": {
                "action_distribution": "a",
                "fit_horizon": 1,
            },
        },
        "record": {
            "architecture": arch,
            "checkpoint": f"/tmp/{arch}/seed_{seed}/epoch_{stage}.pt",
            "model_factors": {
                "action_distribution": "a",
                "latent_dimension": 4,
                "training_stage": stage,
            },
            "diagnostics": {
                "prediction_mse": 1.0,
                "predictor_spectrum": [{"real": 0.5, "imag": 0.0}],
                "task_metric_alignment": 0.1,
                "task_metric_commutator": 0.2,
                "task_metric_schur_offdiagonal_energy": 0.3,
                "task_metric_transient_amplification": 0.4,
            },
            "heldout": {"planning_loss": 1.0},
        },
    }


def test_coverage_keeps_missing_external_fields_visible():
    payload = _payload("a", 1, 1)
    result = _coverage([payload])
    assert result["required_diagnostics"]["prediction_mse"]["present"] == 1
    assert result["required_diagnostics"]["atm_action_inverse_risk"]["missing"] == 1
    assert result["required_diagnostics"]["planning_risk"]["present"] == 1


def test_operator_proxy_metrics_are_not_confused_with_atm():
    payload = _payload("a", 1, 1)
    payload["record"]["diagnostics"]["transfer_operator"] = {
        "A": [[0.5, 0.0], [0.0, 0.25]],
        "B": [[1.0], [0.0]],
    }
    result = _coverage([payload])
    assert result["required_diagnostics"]["resolvent_norm"]["sources"] == {"derived_operator_proxy": 1}
    assert result["required_diagnostics"]["reachability_score"]["sources"] == {"derived_operator_proxy": 1}
    assert result["required_diagnostics"]["atm_action_inverse_risk"]["present"] == 0


def test_normalise_panel_row_declares_horizon_and_factors():
    row = _normalise_panel_row(_payload("a", 1, 1), Path("record.json"))
    assert row["architecture"] == "a"
    assert row["environment"] == "PushT"
    assert row["horizons"] == [1]
    assert row["training_stage"] == 1


def test_run_audit_reports_non_identifiable_current_style_panel(tmp_path):
    root = tmp_path / "records"
    root.mkdir()
    for index in range(2):
        candidate = root / f"candidate_{index}"
        candidate.mkdir()
        (candidate / "checkpoint_record.json").write_text(json.dumps(_payload("a", index, index + 1)))
    results = tmp_path / "results"
    output = tmp_path / "out"
    report = run_audit(root, results, output)
    assert report["strict_matched_audit"]["claim_status"] == "non_identifiable"
    assert report["strict_matched_audit"]["paired_statistics"]["status"] == "blocked_no_identifiable_matched_strata"
    assert (output / "oral_level_unified_diagnostic_audit.json").exists()
