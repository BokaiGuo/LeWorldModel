import json

import pytest

from experiments.empirical_spectral_audit.summarize_joint_ablation_inference import run_report


def _row(mode, predictor_weight, alignment_weight, seed, offset):
    return {
        "mode": mode,
        "predictor_weight": predictor_weight,
        "alignment_weight": alignment_weight,
        "seed": seed,
        "test_one_step_mse": 1.0 + offset,
        "rollout_mse": {"8": 2.0 + offset},
        "composition_error": 3.0 + offset,
        "equivariance": {"predictor_equivariance_error": 4.0 + offset},
        "operator_alignment": {"operator_alignment": 5.0 + offset},
    }


def test_run_report_pairs_seed_conditions(tmp_path):
    rows = []
    for seed in [0, 1, 2]:
        rows.extend(
            [
                _row("none", 0.0, 0.0, seed, 0.0),
                _row("predictor", 1.0, 0.0, seed, -0.2),
                _row("alignment", 0.0, 0.01, seed, -0.1),
                _row("both", 1.0, 0.01, seed, -0.3),
            ]
        )
    source = tmp_path / "joint.json"
    source.write_text(json.dumps({"records": rows}))
    report = run_report(source, bootstrap_repeats=100, bootstrap_seed=5)
    summary = report["comparisons"]["joint_vs_predictor_weight_1"]["metrics"]["rollout_h8_mse"]
    assert summary["n_paired_seeds"] == 3
    assert summary["mean_difference_target_minus_baseline"] == pytest.approx(-0.1)
    assert summary["fraction_target_better"] == 1.0
