import json

import pytest

from experiments.empirical_spectral_audit.summarize_alignment_functional_inference import (
    run_report,
)


def _row(mode, weight, seed, offset):
    return {
        "mode": mode,
        "alignment_weight": weight,
        "seed": seed,
        "test_one_step_mse": 1.0 + offset,
        "random_rollout_mse": {"id": {"8": 2.0 + offset}, "ood": {"8": 3.0 + offset}},
        "action_composition_error": {"0->0": 4.0 + offset},
        "planning": {
            "id": {"planning_regret": 5.0 + offset, "goal_success_rate": 0.5 - offset},
            "ood": {"planning_regret": 6.0 + offset, "goal_success_rate": 0.4 - offset},
        },
        "diagnostics": {
            "global_commutator": 7.0 + offset,
            "conditional_commutator_mean": 8.0 + offset,
        },
    }


def test_functional_inference_orients_success_and_pairs_seeds(tmp_path):
    rows = []
    for seed in [0, 1, 2]:
        rows.extend(
            [
                _row("none", 0.0, seed, 0.0),
                _row("conditional", 0.1, seed, -0.1),
            ]
        )
    source = tmp_path / "functional.json"
    source.write_text(json.dumps({"records": rows}))
    report = run_report(source, repeats=100, seed=2)
    metrics = report["comparisons"]["conditional_0.1_vs_none"]["metrics"]
    assert metrics["id_planning_regret"]["mean_oriented_difference_target_minus_baseline"] == pytest.approx(-0.1)
    assert metrics["id_planning_success"]["mean_oriented_difference_target_minus_baseline"] == pytest.approx(-0.1)
    assert metrics["id_planning_regret"]["n_paired_seeds"] == 3
