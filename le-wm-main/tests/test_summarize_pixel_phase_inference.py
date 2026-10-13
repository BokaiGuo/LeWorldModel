import json

import pytest

from experiments.empirical_spectral_audit.summarize_pixel_phase_inference import run_report


def _row(sigreg, mode, weight, seed, offset):
    return {
        "sigreg_weight": sigreg,
        "mode": mode,
        "alignment_weight": weight,
        "seed": seed,
        "test_one_step_mse": 1.0 + offset,
        "rollout_mse": {"8": 2.0 + offset},
        "action_composition_error_mean": 3.0 + offset,
        "diagnostics": {"global_commutator": 4.0 + offset, "conditional_commutator": 5.0 + offset},
    }


def test_run_report_matches_same_sigreg_baseline(tmp_path):
    rows = []
    for seed in [0, 1, 2]:
        rows.extend(
            [
                _row(0.01, "none", 0.0, seed, 0.0),
                _row(0.01, "conditional", 0.1, seed, -0.3),
            ]
        )
    source = tmp_path / "phase.json"
    source.write_text(json.dumps({"records": rows}))
    report = run_report(source, repeats=100, seed=3)
    metric = report["comparisons"]["sigreg_0.01_conditional_0.1_vs_none"]["metrics"]["rollout_h8_mse"]
    assert metric["n_paired_seeds"] == 3
    assert metric["mean_difference_target_minus_same_sigreg_baseline"] == pytest.approx(-0.3)
    assert metric["fraction_target_lower"] == 1.0
