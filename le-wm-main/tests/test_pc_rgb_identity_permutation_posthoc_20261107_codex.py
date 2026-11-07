import itertools

import numpy as np
import pytest

from experiments.pc_rgb_identity_permutation_posthoc_20261107_codex import retain
from experiments.pc_rgb_hold_retention_search_20261107_codex import choose_retained


def test_identity_and_shared_relabel_invariance():
    costs = np.array([[4, 1, 2], [6, 5, 0], [7, 2, 3]], dtype=float)
    assert retain(costs, (0, 0, 0)) == retain(costs, (1, 1, 1))
    assert retain(costs, (0, 0, 0))[0] == 1
    for swaps in itertools.product((0, 1), repeat=3):
        assert retain(costs, swaps)[0] == 1


def test_stable_ties_and_validation():
    assert retain(np.zeros((3, 3)), (0, 0, 0)) == [0, 1]
    with pytest.raises(ValueError):
        retain(np.zeros((3, 2)), (0, 0, 0))
    with pytest.raises(ValueError):
        retain(np.zeros((3, 3)), (0, 2, 0))


def test_all_permutations_match_production_retention():
    rng = np.random.default_rng(121)
    for costs in [rng.normal(size=(3, 3)).astype(np.float32) for _ in range(50)]:
        for swaps in itertools.product((0, 1), repeat=3):
            changed = costs.copy()
            for i, swap in enumerate(swaps):
                if swap:
                    changed[i, 1:] = costs[i, 1:][::-1]
            assert retain(costs, swaps) == choose_retained(changed, "effect", 0)
