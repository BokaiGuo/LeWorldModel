import json

import pytest

from experiments.empirical_spectral_audit.summarize_data_efficiency_inference import run_report


def _row(n, mode, seed, offset):
    return {
        "sample_count": n,
        "mode": mode,
        "seed": seed,
        "test_one_step_mse": 1.0 + offset,
        "rollout_h8_mse": 2.0 + offset,
        "composition_error": 3.0 + offset,
        "predictor_equivariance_error": 4.0 + offset,
        "operator_alignment": 5.0 + offset,
    }


def test_run_report_pairs_same_sample_count(tmp_path):
    rows = []
    for seed in [0, 1, 2]:
        rows.extend(
            [
                _row(64, "none", seed, 0.0),
                _row(64, "predictor", seed, -0.1),
                _row(64, "both", seed, -0.2),
            ]
        )
    source = tmp_path / "data.json"
    source.write_text(json.dumps({"records": rows}))
    report = run_report(source, repeats=100, seed=2)
    summary = report["comparisons"]["n_64_joint_vs_none"]["metrics"]["rollout_h8_mse"]
    assert summary["n_paired_seeds"] == 3
    assert summary["mean_difference_target_minus_baseline"] == pytest.approx(-0.2)
    assert summary["fraction_target_lower"] == 1.0
