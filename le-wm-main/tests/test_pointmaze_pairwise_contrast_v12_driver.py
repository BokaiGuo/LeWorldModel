from __future__ import annotations

import inspect

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v12_cuda as v12


def test_v12_binds_fresh_balanced_preflight_to_cuda_runner() -> None:
    assert v12.ROOT_RANGE == (20_552, 21_384)
    assert v12.SPLIT_COUNTS == {"train": 256, "calibration": 256, "outcome": 256}
    assert v12.CANDIDATE_DESIGN == "pairwise_contrast_balanced_reversal_interaction_pairs"
    assert v12.RUN_SCHEMA == "pointmaze_pairwise_contrast_v12_runtime.v1"
    assert v12.RUNTIME_FILENAME == "v12_cuda_runtime.json"
    assert "_configure_runner()" in inspect.getsource(v12.main)


def test_pairwise_runner_ignores_only_audit_receipt_annotations_when_revalidating() -> None:
    source = inspect.getsource(v12.runner._ensure_preflight)
    assert '"extension", "prior_context_count"' in source
    assert "canonical preflight payload" in source
    assert "runner.main()" in inspect.getsource(v12.main)
