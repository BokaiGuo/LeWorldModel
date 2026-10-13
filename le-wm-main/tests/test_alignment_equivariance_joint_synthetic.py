from experiments.empirical_spectral_audit.run_alignment_equivariance_joint_synthetic import (
    _summary,
)


def test_joint_summary_contains_both_distinct_diagnostics():
    rows = [
        {
            "mode": "both",
            "predictor_weight": 1.0,
            "alignment_weight": 0.01,
            "seed": 0,
            "test_one_step_mse": 1.0,
            "rollout_mse": {"8": 2.0},
            "composition_error": 3.0,
            "equivariance": {"predictor_equivariance_error": 4.0},
            "operator_alignment": {"operator_alignment": 5.0},
        },
    ]
    summary = next(iter(_summary(rows).values()))
    assert summary["predictor_equivariance_error"]["mean"] == 4.0
    assert summary["operator_alignment"]["mean"] == 5.0
