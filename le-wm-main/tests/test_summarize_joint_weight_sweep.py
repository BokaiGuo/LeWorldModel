import json

import pytest

from experiments.empirical_spectral_audit.summarize_joint_weight_sweep import run_report


def _row(mode, predictor, alignment, seed, offset):
    return {
        "mode": mode,
        "predictor_weight": predictor,
        "alignment_weight": alignment,
        "seed": seed,
        "test_one_step_mse": 1.0 + offset,
        "rollout_mse": {"8": 2.0 + offset},
        "composition_error": 3.0 + offset,
        "equivariance": {"predictor_equivariance_error": 4.0 + offset},
        "operator_alignment": {"operator_alignment": 5.0 + offset},
    }


def test_run_report_pairs_each_joint_weight(tmp_path):
    rows = []
    for seed in [0, 1, 2]:
        rows.extend(
            [
                _row("predictor", 1.0, 0.0, seed, 0.0),
                _row("both", 1.0, 0.001, seed, -0.2),
            ]
        )
    source = tmp_path / "sweep.json"
    source.write_text(json.dumps({"records": rows}))
    report = run_report(source, repeats=100, seed=5)
    summary = report["comparisons"]["joint_alignment_0.001"]["metrics"]["rollout_h8_mse"]
    assert summary["n_paired_seeds"] == 3
    assert summary["mean_difference_joint_minus_predictor"] == pytest.approx(-0.2)
    assert summary["fraction_joint_lower"] == 1.0
