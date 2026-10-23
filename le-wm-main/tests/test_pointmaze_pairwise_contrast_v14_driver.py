from __future__ import annotations

import inspect

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v14_cuda as v14


def test_v14_binds_the_velocity_matched_pool_and_grouped_candidate_contract() -> None:
    assert v14.ROOT_RANGE == (22_216, 23_048)
    assert v14.SPLIT_COUNTS == {"train": 256, "calibration": 256, "outcome": 256}
    assert v14.CANDIDATE_DESIGN == "velocity_matched_acceleration_pairs"
    assert v14.RUN_SCHEMA == "pointmaze_pairwise_contrast_v14_velocity_alias_runtime.v1"
    source = inspect.getsource(v14._ensure_velocity_pool)
    assert "velocity_matched_manifest.json" in source
    assert '"pair_count": 416' in source
    configured = inspect.getsource(v14._configure_runner)
    assert "runner._ensure_extension = _ensure_velocity_pool" in configured
