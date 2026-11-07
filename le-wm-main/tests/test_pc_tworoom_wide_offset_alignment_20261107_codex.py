import numpy as np
import pytest

from experiments import pc_tworoom_wide_offset_alignment_20261107_codex as wide


def test_pool_matches_repaired_shared_offsets(monkeypatch):
    geometry = (np.array([80., 80.]), np.array([145., 80.]), 112., 50.)
    monkeypatch.setattr(wide.repair.v2.v1, "geometry", lambda current, goal: geometry)
    pool = wide.candidate_pool(None, None)
    old, _ = wide.repair.candidate_pool(None, None)
    assert pool.shape == (13, 25, 2)
    assert np.max(np.abs(pool)) <= 1.
    np.testing.assert_allclose(pool[:5], old[[1, 4, 7, 10, 13]], atol=0, rtol=0)
    assert not np.array_equal(pool[5], pool[6])


def test_non_cross_wall_is_not_silent(monkeypatch):
    geometry = (np.array([80., 80.]), np.array([90., 80.]), 112., 50.)
    monkeypatch.setattr(wide.repair.v2.v1, "geometry", lambda current, goal: geometry)
    with pytest.raises(ValueError):
        wide.candidate_pool(None, None)
