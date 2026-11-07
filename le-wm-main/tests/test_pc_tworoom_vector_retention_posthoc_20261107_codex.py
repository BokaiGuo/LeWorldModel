import numpy as np
import pytest

from experiments.pc_tworoom_vector_retention_posthoc_20261107_codex import retention, select_chain


def test_retention_preserves_lookahead_first_and_unique_slot():
    a = np.ones((8, 8))
    a[3, 0] = -1
    a[4, 1] = 5
    result = retention(a, np.arange(8), 7)
    assert result["parent"] == [0, 1]
    assert result["lookahead"][0] == 3
    assert result["vector"][0] == result["shuffled"][0] == 3
    assert len(set(result["vector"])) == len(set(result["shuffled"])) == 2


def test_selection_retains_a_best_and_uses_only_retained_b():
    a = np.full((8, 8), 5.)
    b = np.full((8, 8), 10.)
    b[7, 0] = 0
    assert select_chain(a, b, [1, 3]) == ("a", 0)
    b[3, 2] = 4
    assert select_chain(a, b, [3, 1]) == ("b", 26)


def test_nonfinite_and_duplicate_rejected():
    a = np.zeros((8, 8))
    with pytest.raises(ValueError):
        select_chain(a, a, [1, 1])
    a[0, 0] = np.nan
    with pytest.raises(ValueError):
        retention(a, np.zeros(8), 0)
