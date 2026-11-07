import numpy as np

from experiments import pc_exact_waypoint_effect_retention_20261107_codex as experiment


def test_rank_banks_are_disjoint_and_ignore_cost():
    offsets = np.random.default_rng(1).uniform(size=384)
    pool, a, b = experiment.rank_indices(offsets)
    assert all(len(x) == 8 for x in (pool, a, b))
    assert len(set(np.r_[pool, a, b])) == 24
    ordered = np.argsort(offsets, kind="stable")
    assert np.array_equal(pool, ordered[np.arange(24, 384, 48)])


def test_selection_independent_of_unqueried_bank():
    scores = np.random.default_rng(4).uniform(size=(8, 8))
    parent = np.arange(8.)
    immediate = parent[::-1]
    selected = experiment.core.select(scores, parent, immediate, 27)
    assert set(selected) == set(experiment.ARMS)
    assert all(len(v) == len(set(v)) == 2 for v in selected.values())
    assert selected == experiment.core.select(scores.copy(), parent, immediate, 27)
