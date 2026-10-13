import json

import pytest

from experiments.empirical_spectral_audit.summarize_pixel_world_inference import run_report


def _row(mode, weight, seed, offset):
    return {
        "mode": mode,
        "alignment_weight": weight,
        "seed": seed,
        "test_one_step_mse": 1.0 + offset,
        "rollout_mse": {"8": 2.0 + offset},
        "action_composition_error_mean": 3.0 + offset,
        "diagnostics": {"global_commutator": 4.0 + offset, "conditional_commutator": 5.0 + offset},
    }


def test_run_report_pairs_pixel_conditions(tmp_path):
    rows = []
    for seed in [0, 1, 2]:
        rows.extend(
            [
                _row("none", 0.0, seed, 0.0),
                _row("global", 0.01, seed, -0.1),
                _row("conditional", 0.01, seed, -0.2),
                _row("global", 0.1, seed, -0.3),
                _row("conditional", 0.1, seed, -0.4),
            ]
        )
    source = tmp_path / "pixel.json"
    source.write_text(json.dumps({"records": rows}))
    report = run_report(source, repeats=100, seed=7)
    summary = report["comparisons"]["conditional_0.01_vs_none"]["metrics"]["rollout_h8_mse"]
    assert summary["n_paired_seeds"] == 3
    assert summary["mean_difference_target_minus_baseline"] == pytest.approx(-0.2)
    assert summary["fraction_target_lower"] == 1.0
