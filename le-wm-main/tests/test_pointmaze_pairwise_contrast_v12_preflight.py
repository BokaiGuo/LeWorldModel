from __future__ import annotations

import inspect

import numpy as np

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v12 as v12


def test_v12_registers_balanced_pairwise_static_contract() -> None:
    assert v12.SCHEMA in panel.SUPPORTED_FROZEN_PREFLIGHT_SCHEMAS
    assert v12.SCHEMA in panel.EXACT_SHARED_SCORER_PREFLIGHT_SCHEMAS
    assert v12.MIN_FOLD_SIGN_BALANCE == 0.25
    source = inspect.getsource(v12.preflight)
    assert "pairwise_contrast_balanced_reversal_interaction_candidate_banks" in source
    assert "pairwise_two_fold_sign_balance" in source


def test_v12_host_audit_uses_fresh_roots_and_never_opens_gpu_fit_or_outcomes() -> None:
    from experiments.gdd_wm import run_pointmaze_pairwise_contrast_v12_balance_preflight as audit

    assert audit.ROOT_RANGE == (20_552, 21_384)
    assert audit.SPLIT_COUNTS == {"train": 256, "calibration": 256, "outcome": 256}
    source = inspect.getsource(audit.main)
    assert "_ensure_extension()" in source
    assert "v12.preflight(" in source
    assert "cuda" not in source.lower()


def test_position_aliased_selector_hard_rejects_a_position_only_shortcut() -> None:
    states = np.asarray([
        [-1.0, 0.0, -1.0, 0.0], [-0.8, 0.0, 1.0, 0.0], [-0.6, 0.0, -1.0, 0.0], [-0.4, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0], [-0.8, 0.0, -1.0, 0.0], [-0.6, 0.0, 1.0, 0.0], [-0.4, 0.0, -1.0, 0.0],
    ])
    proposals = np.zeros((2, 2, panel.TOTAL_STEPS, 2), dtype=np.float64)
    proposals[:, 0, 0, 0], proposals[:, 1, 0, 0] = 0.6, -0.6

    def simulator(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray([state[0] + action[0, 0] * state[2], 0.0, 0.0, 0.0])

    held, report = panel.screen_pairwise_contrast_position_aliased_interaction_pairs(
        states, proposals, np.asarray([0.0, 0.0]), simulator=simulator,
        pair_count=2, minimum_fold_sign_balance=0.40,
        maximum_crossfold_position_only_sign_accuracy=0.625,
    )
    assert held.shape == (4, panel.TOTAL_STEPS, 2)
    assert report["feasible_pair_count"] >= 2
    assert all(item["maximum_crossfold_position_only_sign_accuracy"] <= 0.625 for item in report["selected_pair_contract"])


def test_position_aliased_selector_reports_which_hard_gate_exhausted_the_bank() -> None:
    states = np.asarray([[float(index), 0.0, 0.0, 0.0] for index in range(8)])
    proposals = np.zeros((2, 2, panel.TOTAL_STEPS, 2), dtype=np.float64)
    proposals[:, 0, 0, 0], proposals[:, 1, 0, 0] = 0.4, -0.4

    def simulator(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray([state[0] + action[0, 0], 0.0, 0.0, 0.0])

    import pytest
    with pytest.raises(ValueError, match="contrast_coverage_balance_count=.*proxy_admissible_count=.*intersection_count="):
        panel.screen_pairwise_contrast_position_aliased_interaction_pairs(
            states, proposals, np.asarray([0.0, 0.0]), simulator=simulator,
            pair_count=2, minimum_fold_sign_balance=0.40,
            maximum_crossfold_position_only_sign_accuracy=0.625,
        )
