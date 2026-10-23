from __future__ import annotations

import inspect

import numpy as np

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v14_velocity_alias as v14


def test_velocity_alias_proposals_are_opposed_and_in_control_support() -> None:
    pairs = panel.velocity_opposed_acceleration_pair_proposals(seed=91, count=8)
    assert pairs.shape == (8, 2, panel.TOTAL_STEPS, 2)
    assert np.allclose(pairs[:, 0], -pairs[:, 1])
    assert float(np.max(np.abs(pairs))) <= 1.0


def test_velocity_braking_proposals_vary_magnitude_without_revealing_velocity() -> None:
    pairs = panel.velocity_braking_acceleration_pair_proposals(seed=92, count=8)
    assert pairs.shape == (8, 2, panel.TOTAL_STEPS, 2)
    weak_norm = np.linalg.norm(pairs[:, 0, 0], axis=1)
    strong_norm = np.linalg.norm(pairs[:, 1, 0], axis=1)
    assert np.all(strong_norm > weak_norm)
    cross = np.sum(pairs[:, 0, 0] * pairs[:, 1, 0], axis=1)
    assert np.all(cross > 0.0)


def test_velocity_alias_folds_keep_complete_velocity_pairs_together() -> None:
    first, second = panel._velocity_matched_design_folds(64)
    assert len(first) == len(second) == 32
    assert all((int(index) ^ 1) in set(first) for index in first)
    assert all((int(index) ^ 1) in set(second) for index in second)


def test_v14_preflight_declares_pair_group_isolation_and_alias_gates() -> None:
    assert v14.SCHEMA == panel.PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA
    assert v14.MIN_FOLD_REVERSAL == 0.75
    assert v14.MAX_PROXY_ACCURACY == 0.625
    source = inspect.getsource(v14.preflight)
    assert "_grouped_split" in source
    assert "complete_velocity_pairs_stay_within_split_or_design" in source
    assert "opposed_velocity_winner_reversal" in source
    assert "position_only_proxy_rejected" in source


def test_velocity_alias_ledgers_return_the_frozen_grouped_split_to_the_builder() -> None:
    source = inspect.getsource(panel.frozen_preflight_provenance)
    assert "PAIRWISE_CONTRAST_V14_PREFLIGHT_SCHEMA" in source
    assert "PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA" in source
    assert "exact_split_schemas" in source
