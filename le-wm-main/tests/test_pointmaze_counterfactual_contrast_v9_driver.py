from __future__ import annotations

import inspect

from experiments.decision_benchmark import run_pointmaze_counterfactual_contrast_v9_rgb_geometry_cuda as driver


def test_v9_binds_fresh_roots_and_declared_geometry_feature_family() -> None:
    source = inspect.getsource(driver.main)
    assert driver.OUTPUT.name == "pointmaze_counterfactual_contrast_v9_rgb_geometry_20261022_cuda"
    assert driver.ROOT_RANGE == (11_320, 11_832)
    assert driver.SPLIT_COUNTS == {"train": 128, "calibration": 192, "outcome": 64}
    assert driver.HISTORY_SOURCES == (
        "source_history_model", "source_history_visual_state", "hybrid_latent_visual_state",
    )
    assert "history_sources=HISTORY_SOURCES" in source
    assert "robust_goal_balanced_interaction_pairs" in source
    assert "require_cuda_recurrent_cell(\"cuda\")" in source
    assert "outcome.exists()" in source


def test_v9_excludes_v8_context_and_refuses_partial_model_state() -> None:
    source = inspect.getsource(driver.main)
    assert driver.V8_CONTEXT in driver.CONTEXTS
    assert "unbound calibration checkpoint" in source
    assert "partial score directory" in source
