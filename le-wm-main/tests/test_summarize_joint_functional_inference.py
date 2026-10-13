import json

import pytest

from experiments.empirical_spectral_audit.summarize_joint_functional_inference import (
    run_report,
)


def _row(mode, seed, offset):
    return {
        "mode": mode,
        "seed": seed,
        "test_one_step_mse": 1.0 + offset,
        "id_rollout_mse": {"8": 2.0 + offset},
        "ood_rollout_mse": {"action_2x": {"8": 3.0 + offset}, "combined": {"8": 4.0 + offset}},
        "planning": {"ood_combined": {"planning_regret": 5.0 + offset, "goal_success_rate": 0.5 - offset}},
        "diagnostics": {"predictor_equivariance_error": 6.0 + offset, "operator_alignment": 7.0 + offset},
    }


def test_factorial_inference_reports_main_effect_and_interaction(tmp_path):
    rows = []
    for seed in [0, 1, 2]:
        rows.extend(
            [
                _row("none", seed, 0.0),
                _row("predictor", seed, -0.1),
                _row("alignment", seed, -0.2),
                _row("both", seed, -0.4),
            ]
        )
    source = tmp_path / "joint.json"
    source.write_text(json.dumps({"records": rows}))
    report = run_report(source, repeats=100, seed=2)
    comparisons = report["comparisons"]
    assert comparisons["both_vs_none"]["metrics"]["ood_planning_regret"]["mean_oriented_difference"] == pytest.approx(-0.4)
    assert comparisons["interaction_both_minus_predictor_minus_alignment_plus_none"]["metrics"]["ood_planning_regret"]["mean_oriented_difference"] == pytest.approx(-0.1)
    assert comparisons["both_vs_none"]["metrics"]["ood_planning_regret"]["n_paired_seeds"] == 3
