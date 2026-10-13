import json

import pytest

from experiments.empirical_spectral_audit.summarize_pixel_nuisance_inference import (
    run_report,
)


def _row(mode, weight, seed, offset):
    return {
        "mode": mode,
        "alignment_weight": weight,
        "seed": seed,
        "id_one_step_mse": 1.0 + offset,
        "ood_one_step_mse": 2.0 + offset,
        "id_rollout_mse": {"8": 3.0 + offset},
        "ood_rollout_mse": {"8": 4.0 + offset},
        "ood_tangent_diagnostics": {
            "nuisance_to_action_norm_ratio": 5.0 + offset,
            "predicted_action_tangent_cosine": 0.5 - offset,
        },
        "diagnostics": {"global_commutator": 6.0 + offset, "conditional_commutator": 7.0 + offset},
    }


def test_pixel_nuisance_inference_orients_cosine_and_pairs(tmp_path):
    rows = []
    for seed in [0, 1, 2]:
        rows.extend([_row("none", 0.0, seed, 0.0), _row("conditional", 0.1, seed, -0.1)])
    source = tmp_path / "pixel_nuisance.json"
    source.write_text(json.dumps({"records": rows}))
    report = run_report(source, repeats=100, seed=2)
    metrics = report["comparisons"]["conditional_0.1_vs_none"]["metrics"]
    assert metrics["ood_one_step_mse"]["mean_oriented_difference_target_minus_baseline"] == pytest.approx(-0.1)
    assert metrics["ood_predicted_action_tangent_cosine"]["mean_oriented_difference_target_minus_baseline"] == pytest.approx(-0.1)
    assert metrics["ood_one_step_mse"]["n_paired_seeds"] == 3
