import json
from pathlib import Path

from experiments.empirical_spectral_audit.run_best_paper_readiness_audit import audit


def _record(root: Path) -> None:
    path = root / "candidate_000"
    path.mkdir(parents=True)
    payload = {
        "record_schema": "real_checkpoint_selection.v1",
        "protocol": {
            "environment": "PushT",
            "coordinate_gauge": "decoded_observable_pose_raw",
            "diagnostic_split": "frozen before checkpoint selection",
            "heldout_planning": "disjoint from diagnostic split",
            "runtime_successor_verification": False,
            "prediction_quality": {"action_distribution": "a"},
        },
        "record": {
            "architecture": "a",
            "checkpoint": "/tmp/a/seed_1/epoch_1.pt",
            "group": "PushT/a/seed_1",
            "model_factors": {"latent_dimension": 2, "training_stage": 1, "action_distribution": "a"},
            "diagnostics": {
                "prediction_mse": 1.0,
                "predictor_spectrum": [{"real": 0.5, "imag": 0.0}],
                "task_metric_alignment": 0.1,
                "task_metric_commutator": 0.2,
                "task_metric_schur_offdiagonal_energy": 0.3,
                "task_metric_transient_amplification": 0.4,
                "transfer_operator": {
                    "A": [[0.5, 0.0], [0.0, 0.25]],
                    "B": [[1.0], [0.0]],
                    "D": [[1.0, 0.0]],
                    "risk_envelope": {
                        "status": "blocked_missing_independent_components",
                        "action_effect": "unidentified",
                        "off_manifold": "unidentified",
                        "readout": "independent task labels required",
                        "bound_allowed_for_control": False,
                    },
                },
            },
            "heldout": {"planning_loss": 1.0},
        },
    }
    (path / "checkpoint_record.json").write_text(json.dumps(payload))


def test_best_paper_audit_computes_g_h_and_rejects_missing_gates(tmp_path):
    root = tmp_path / "records"
    _record(root)
    report = audit(root, tmp_path / "results")
    item = report["transfer_tool"]["record_reports"][0]
    assert item["finite_horizon_transfer"]["status"] == "available"
    assert item["finite_horizon_transfer"]["horizons"]["4"]["G_h_norm"] > 0
    assert item["permission"]["status"] == "rejected_fail_closed"
    assert report["theory_requirement"]["counterexample"]["same_declared_prediction_mse"] is True
    assert report["independent_real_gain"]["status"] == "blocked_missing_independent_gain_schema"


def test_best_paper_audit_rejects_incomplete_external_metric_object(tmp_path):
    root = tmp_path / "records"
    _record(root)
    results = tmp_path / "results"
    results.mkdir()
    artifact = {
        "external_independent_gain": {
            "schema": "external_independent_gain.v1",
            "protocol": {
                "privileged_state_used": False,
                "test_planning_used_for_selection": False,
                "train_calibration_test_root_disjoint": True,
                "independent_heldout_consequence": True,
            },
            "metrics": {
                "action_ranking_inversion": {"status": "available", "estimate": 0.1, "ci95": [0.0, 0.2]},
                "multistep_task_error": {"status": "available", "estimate": 0.2, "ci95": [0.1, 0.3]},
                "contact_retention": {"status": "not_applicable"},
                "angle_closure": {"status": "available", "estimate": 0.3, "ci95": [0.2, 0.4]},
                "planning_regret": {"status": "available", "estimate": 0.4, "ci95": [0.3, 0.5]},
            },
        }
    }
    (results / "incomplete.json").write_text(json.dumps(artifact))
    report = audit(root, results)
    gate = report["independent_real_gain"]
    assert gate["candidate_artifact_count"] == 1
    assert gate["valid_artifact_count"] == 0
    assert any("contact_retention:status=not_applicable" in reason for item in gate["invalid_artifacts"] for reason in item["reasons"])
