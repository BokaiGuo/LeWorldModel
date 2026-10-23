from __future__ import annotations

import inspect

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v11_cuda as driver


def test_v11_binds_fresh_pairwise_protocol_before_cuda_fit() -> None:
    source = inspect.getsource(driver.main)
    assert driver.ROOT_RANGE == (19_720, 20_552)
    assert driver.SPLIT_COUNTS == {"train": 256, "calibration": 256, "outcome": 256}
    assert "require_cuda_recurrent_cell(\"cuda\")" in source
    assert "_ensure_preflight(contexts)" in source
    assert driver.CANDIDATE_DESIGN == "pairwise_contrast_robust_interaction_pairs"
    assert "CANDIDATE_DESIGN" in source
    assert "frozen_preflight_ledger=PREFLIGHT" in source
    assert "full-bank planner regret only" in source
    assert "calibrate_pairwise_contrast(" in source


def test_v11_root_budget_includes_disjoint_pair_design_roots() -> None:
    from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as builder

    assert driver.ROOT_RANGE[1] - driver.ROOT_RANGE[0] == sum(driver.SPLIT_COUNTS.values()) + builder.INTERACTION_DESIGN_ROOTS


def test_v11_resume_reuses_frozen_context_ledger_not_its_own_panel() -> None:
    source = inspect.getsource(driver._prior_contexts)
    assert "if PREFLIGHT.is_file()" in source
    assert "prior_context_sha256" in source
    assert "frozen preflight context bytes changed" in source
