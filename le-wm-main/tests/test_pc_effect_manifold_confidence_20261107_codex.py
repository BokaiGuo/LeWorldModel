import numpy as np
import pytest

from experiments.pc_effect_manifold_confidence_20261107_codex import indices, paired_stats


def test_selection_deterministic_and_tie_stable():
    residual = np.zeros((8, 9))
    cost = np.tile(np.arange(9), (8, 1))
    selected = indices(residual, cost, 10)
    assert selected["manifold"] == list(range(32))
    assert len(set(selected["random"])) == 32
    assert selected == indices(residual, cost, 10)


def test_pair_stats_and_invalid():
    cost = np.tile(np.arange(9.), (8, 1))
    stats = paired_stats(cost, cost, list(range(32)))
    assert stats["valid"] == stats["matches"] == 32
    with pytest.raises(ValueError):
        paired_stats(cost, cost, [1, 1])
    with pytest.raises(ValueError):
        indices(np.full((8, 9), np.nan), cost, 10)
