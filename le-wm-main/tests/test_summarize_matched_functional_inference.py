import json

import numpy as np

from experiments.empirical_spectral_audit.summarize_matched_functional_inference import (
    _nearest_same_seed_pairs,
    _sign_flip_pvalue,
    run_report,
)


def _row(seed, mode, error, rollout, success=0.5):
    return {
        "seed": seed,
        "mode": mode,
        "alignment_weight": 0.0,
        "test_one_step_mse": error,
        "random_rollout_mse": {"id": {"8": rollout}, "ood": {"8": rollout + 0.1}},
        "action_composition_error": {"a": rollout},
        "planning": {
            "id": {"planning_regret": rollout, "goal_success_rate": success},
            "ood": {"planning_regret": rollout + 0.1, "goal_success_rate": success},
        },
        "diagnostics": {"global_commutator": rollout, "conditional_commutator_mean": rollout},
    }


def test_matching_is_same_seed_and_filters_relative_gap():
    rows = [_row(0, "none", 1.0, 1.0), _row(0, "global", 1.05, 0.5), _row(1, "none", 1.0, 1.0), _row(1, "global", 2.0, 0.2)]
    pairs = _nearest_same_seed_pairs(rows, "alignment", "global@0", "none@0")
    assert [pair.seed for pair in pairs] == [0, 1]
    assert np.isclose(pairs[0].relative_one_step_gap, 0.05)


def test_exact_sign_flip_is_reproducible():
    assert _sign_flip_pvalue(np.asarray([1.0, 1.0, 1.0])) == 0.25


def test_report_marks_empty_match_as_insufficient(tmp_path):
    payload = {"records": [_row(0, "none", 1.0, 1.0), _row(0, "global", 2.0, 0.2)]}
    path = tmp_path / "alignment_functional_synthetic.json"
    path.write_text(json.dumps(payload))
    report = run_report(path, tolerances=(0.1,), repeats=100, seed=1)
    result = report["comparisons"]["global@0_vs_none@0"]["matched_by_tolerance"]["0.1"]
    assert result["n_matched"] == 0
    assert result["metrics"]["test_one_step_mse"]["evidence_status"] == "insufficient_matched_pairs"
