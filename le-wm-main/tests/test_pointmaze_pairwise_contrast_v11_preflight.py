from __future__ import annotations

import inspect

import numpy as np

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v11 as v11


def test_v11_declares_pairwise_not_full_bank_headroom_contract() -> None:
    assert v11.SCHEMA in panel.SUPPORTED_FROZEN_PREFLIGHT_SCHEMAS
    assert v11.SCHEMA in panel.EXACT_SHARED_SCORER_PREFLIGHT_SCHEMAS
    source = inspect.getsource(v11.preflight)
    assert "pairwise_contrast_metrics" in source
    assert "pairwise_absolute_cost_gap_q10" in source
    assert "pairwise_state_conditional_signs" in source
    assert "outcome" in v11.preflight.__doc__.lower() if v11.preflight.__doc__ else True


def test_pairwise_selector_optimizes_pair_contrast_on_each_design_fold() -> None:
    states = np.asarray([
        [-1.0, 0.0, 0.0, 0.0], [-0.8, 0.0, 0.0, 0.0], [-0.6, 0.0, 0.0, 0.0], [-0.4, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    ])
    proposals = np.zeros((10, 2, panel.TOTAL_STEPS, 2), dtype=np.float64)
    for index in range(len(proposals)):
        proposals[index, 0, 0, 0] = 0.2 + 0.02 * index
        proposals[index, 1, 0, 0] = -0.2 - 0.02 * index

    def simulator(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray([state[0] + action[0, 0], 0.0, 0.0, 0.0])

    held, report = panel.screen_pairwise_contrast_robust_interaction_pairs(
        states, proposals, np.asarray([0.0, 0.0]), simulator=simulator, pair_count=2,
    )
    assert held.shape == (4, panel.TOTAL_STEPS, 2)
    assert report["selected_pooled_absolute_gap_q10"] >= 0.003
    assert report["selected_gap_ge_margin_fraction"] >= 0.80
    assert report["selected_sign_mixed_pair_count"] == 2


def test_pairwise_selector_rejects_imbalanced_winner_shortcut_before_fit() -> None:
    states = np.asarray([
        [-1.0, 0.0, 0.0, 0.0], [-0.8, 0.0, 0.0, 0.0], [-0.6, 0.0, 0.0, 0.0], [-0.4, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    ])
    proposals = np.zeros((2, 2, panel.TOTAL_STEPS, 2), dtype=np.float64)
    proposals[:, 0, 0, 0], proposals[:, 1, 0, 0] = 0.25, -0.25

    def simulator(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray([state[0] + action[0, 0], 0.0, 0.0, 0.0])

    import pytest
    with pytest.raises(ValueError, match="hard two-fold sign-balance gate"):
        panel.screen_pairwise_contrast_robust_interaction_pairs(
            states, proposals, np.asarray([0.0, 0.0]), simulator=simulator,
            pair_count=2, minimum_fold_sign_balance=0.25,
        )
