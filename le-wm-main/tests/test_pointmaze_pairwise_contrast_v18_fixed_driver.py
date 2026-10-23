from __future__ import annotations

import inspect

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v18_fixed_cuda as v18
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed as preflight
from experiments.gdd_wm import run_pointmaze_pairwise_contrast_v18_preflight as launcher


def test_v18_uses_a_native_preflight_schema_and_fresh_root_range() -> None:
    assert preflight.SCHEMA == panel.PAIRWISE_CONTRAST_V18_PREFLIGHT_SCHEMA
    assert v18.ROOT_RANGE == (25_544, 26_376)
    assert v18.SPLIT_COUNTS == {"train": 256, "calibration": 256, "outcome": 256}
    assert v18.CANDIDATE_DESIGN == "velocity_matched_braking_pairs"
    assert launcher.ROOT_RANGE == v18.ROOT_RANGE


def test_v18_freezes_v17_policy_before_fresh_calibration_and_outcome() -> None:
    source = inspect.getsource(v18._fixed_calibration)
    assert "calibrate_fixed_policy" in source
    assert "hidden=32" in source
    assert "contrast_weight=1.0" in source
    assert "temperature=1.0" in source
    assert "source_history_model" in source
