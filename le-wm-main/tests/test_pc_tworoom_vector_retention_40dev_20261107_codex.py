import numpy as np
import pytest

from experiments.pc_tworoom_vector_retention_40dev_20261107_codex import select_from_union
from experiments.pc_tworoom_vector_retention_posthoc_20261107_codex import select_chain


def test_union_equivalent_to_full_scoring_and_ties():
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(8, 8)), rng.normal(size=(8, 8))
    keep = [5, 2]
    ids = np.sort((np.array(keep)[:, None] * 8 + np.arange(8)).ravel())
    assert select_from_union(a, ids, b.ravel()[ids], keep) == select_chain(a, b, keep)
    assert select_from_union(np.zeros((8, 8)), ids, np.zeros(16), keep) == ("a", 0)


def test_missing_and_duplicate_union_fail_closed():
    a = np.zeros((8, 8))
    with pytest.raises(ValueError):
        select_from_union(a, np.arange(15), np.zeros(15), [0, 1])
    with pytest.raises(ValueError):
        select_from_union(a, np.zeros(16, dtype=int), np.zeros(16), [0, 1])
