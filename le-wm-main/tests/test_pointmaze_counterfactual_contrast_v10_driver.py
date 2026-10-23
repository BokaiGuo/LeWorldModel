from __future__ import annotations

import inspect

from experiments.decision_benchmark import run_pointmaze_counterfactual_contrast_v10_fixed_confirmation_cuda as driver


def test_v10_is_a_large_fresh_fixed_preoutcome_confirmation() -> None:
    source = inspect.getsource(driver.main)
    assert driver.ROOT_RANGE == (11_832, 12_920)
    assert driver.SPLIT_COUNTS == {"train": 256, "calibration": 512, "outcome": 256}
    assert driver.FIXED_POLICY == {
        "history_source": "source_history_model", "hidden": 16,
        "contrast_weight": 1.0, "temperature": 0.75, "policy": {"kind": "recurrent"},
    }
    assert "_v9_preoutcome_reference()" in source
    assert "calibrate_fixed_policy(" in source
    assert "robust_goal_balanced_interaction_pairs" in source
    assert "require_cuda_recurrent_cell(\"cuda\")" in source


def test_v10_binds_v9_preoutcome_selection_and_excludes_v9_context() -> None:
    source = inspect.getsource(driver._v9_preoutcome_reference)
    assert driver.V9_CONTEXT in driver.CONTEXTS
    assert "outcome_labels_opened" in source
    assert "outcome_labels_generated" in source


def test_v10_declares_enough_fresh_roots_for_evaluation_and_robust_design() -> None:
    from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as builder

    assert driver.ROOT_RANGE[1] - driver.ROOT_RANGE[0] == sum(driver.SPLIT_COUNTS.values()) + builder.INTERACTION_DESIGN_ROOTS
