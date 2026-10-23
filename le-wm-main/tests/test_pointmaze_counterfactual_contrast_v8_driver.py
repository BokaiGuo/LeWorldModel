from __future__ import annotations

import inspect

from experiments.decision_benchmark import run_pointmaze_counterfactual_contrast_v8_robust_cuda as driver


def test_v8_driver_binds_fresh_roots_larger_calibration_and_robust_bank() -> None:
    source = inspect.getsource(driver.main)
    assert driver.OUTPUT.name == "pointmaze_counterfactual_contrast_v8_robust_bank_20261022_cuda"
    assert driver.ROOT_RANGE == (10_808, 11_320)
    assert driver.SPLIT_COUNTS == {"train": 128, "calibration": 192, "outcome": 64}
    assert len(driver.CONTEXTS) >= len(driver.V7_CONTEXTS)
    assert all(path.is_absolute() for path in driver.CONTEXTS)
    assert "torch.backends.cudnn.enabled = False" in source
    assert 'require_cuda_recurrent_cell("cuda")' in source
    assert 'candidate_design="robust_goal_balanced_interaction_pairs"' in source
    assert "split_counts=SPLIT_COUNTS" in source
    assert "observability_suite=True, history_gate=True" in source
    assert "unbound calibration checkpoint" in source
    assert "partial score directory" in source


def test_v8_extension_contract_binds_the_unseen_root_range() -> None:
    source = inspect.getsource(driver._ensure_fresh_extension)
    assert "new_episode_count" in source
    assert "new_root_id_range" in source
    assert "V8 extension contract mismatch" in source


def test_v8_rejects_a_candidate_bank_without_two_fold_decision_headroom() -> None:
    valid_fold = {
        "winner_entropy_normalized": 0.75,
        "distinct_winner_count": 4,
        "best_second_gap_ge_0p003_fraction": 0.75,
        "candidate_cost_spread_q10": 0.10,
    }
    report = {
        "interaction_pair_design": {
            "candidate_design": "robust_goal_balanced_interaction_pairs",
            "design_fold_headroom": [valid_fold, dict(valid_fold)],
        }
    }
    assert driver._validate_robust_bank_contract(report)["status"] == "passed_before_model_fitting"
    report["interaction_pair_design"]["design_fold_headroom"][1]["distinct_winner_count"] = 1
    try:
        driver._validate_robust_bank_contract(report)
    except RuntimeError as error:
        assert "headroom contract" in str(error)
    else:
        raise AssertionError("weak design fold unexpectedly passed V8 pre-fit contract")
