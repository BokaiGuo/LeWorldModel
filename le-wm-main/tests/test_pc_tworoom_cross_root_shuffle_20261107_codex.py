import numpy as np
import pytest

from experiments.pc_tworoom_cross_root_shuffle_20261107_codex import (
    choose, rotate_sources,
)


def test_rotation_is_a_bijection_without_fixed_points():
    assert rotate_sources(4) == [1, 2, 3, 0]
    with pytest.raises(ValueError):
        rotate_sources(1)


def test_cross_root_changes_only_diversity_not_opportunity():
    costs = np.tile(np.arange(8, dtype=float)[:, None], (1, 8))
    vectors = np.zeros((8, 8))
    vectors[4, 0] = 20
    assert choose(costs, vectors) == [0, 4]
    vectors[7, 0] = 100
    assert choose(costs, vectors) == [0, 4]
    with pytest.raises(ValueError):
        choose(costs, np.ones((7, 8)))
