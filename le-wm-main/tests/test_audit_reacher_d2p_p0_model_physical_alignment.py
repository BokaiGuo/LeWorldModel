import pytest

from experiments.spectral_reachability.audit_reacher_d2p_p0_model_physical_alignment import audit


def _report(model_deltas, physical_deltas):
    return {
        "schema": "d2p_reacher_p0_goal_progress.v1",
        "protocol": {"outcome_only": True},
        "_sha256": "report-hash",
        "rows": [
            {
                "seed": index,
                "methods": {
                    "d2p_goal_progress": {
                        "predicted_terminal_cost": 1.0 + model_delta,
                        "physical": {"final_qpos_distance": 1.0 + physical_delta},
                    },
                    "cem": {"predicted_terminal_cost": 1.0, "physical": {"final_qpos_distance": 1.0}},
                },
            }
            for index, (model_delta, physical_delta) in enumerate(zip(model_deltas, physical_deltas))
        ],
    }


def _witness():
    return {
        "schema": "d2p_reacher_p0_witness_replay_audit.v1",
        "status": "pass",
        "source_report_sha256": "report-hash",
        "gates": {"cuda_only": True, "replay": True},
    }


def test_alignment_audit_reports_disagreement_without_tuning_signal():
    result = audit(_report([-0.2, -0.1], [-0.3, 0.4]), _witness())
    assert result["status"] == "model_physical_alignment_boundary"
    assert result["strict_order_agreement_count"] == 1
    assert result["strict_order_disagreement_count"] == 1
    assert result["model_d2p_lower_cost_root_count"] == 2
    assert result["physical_d2p_lower_distance_root_count"] == 1
    assert result["gates"]["witness_source_matches_report"] is True


def test_alignment_audit_rejects_nonfinite_or_unverified_inputs():
    with pytest.raises(ValueError, match="passing witness"):
        audit(_report([-0.1], [-0.1]), {"schema": "wrong"})
    with pytest.raises(ValueError, match="finite"):
        audit(_report([float("nan")], [-0.1]), _witness())
