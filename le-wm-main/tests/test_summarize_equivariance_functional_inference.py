import json

import pytest

from experiments.empirical_spectral_audit.summarize_equivariance_functional_inference import (
    run_report,
)


def _row(mode, weight, seed, offset):
    return {
        "mode": mode,
        "regularization_weight": weight,
        "seed": seed,
        "train_one_step_mse": 1.0 + offset,
        "ood_one_step_mse": {"action_2x": 2.0 + offset, "combined": 3.0 + offset},
        "ood_rollout_mse": {"action_2x": {"8": 4.0 + offset}, "combined": {"8": 5.0 + offset}},
        "planning": {
            "in_distribution": {"planning_regret": 6.0 + offset, "goal_success_rate": 0.4 - offset},
            "ood_combined": {"planning_regret": 7.0 + offset, "goal_success_rate": 0.3 - offset},
        },
    }


def test_inference_pairs_seeds_and_orients_success(tmp_path):
    rows = []
    for seed in [0, 1, 2]:
        rows.extend([_row("none", 0.0, seed, 0.0), _row("predictor", 1.0, seed, -0.1)])
    source = tmp_path / "equivariance.json"
    source.write_text(json.dumps({"records": rows}))
    report = run_report(source, repeats=100, seed=2)
    metrics = report["comparisons"]["predictor_1_vs_none"]["metrics"]
    assert metrics["ood_planning_regret"]["mean_oriented_difference_target_minus_baseline"] == pytest.approx(-0.1)
    assert metrics["ood_planning_success"]["mean_oriented_difference_target_minus_baseline"] == pytest.approx(-0.1)
    assert metrics["ood_planning_regret"]["n_paired_seeds"] == 3
