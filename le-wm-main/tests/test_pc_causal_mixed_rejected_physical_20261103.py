import numpy as np
import pytest

from experiments.pc_causal_mixed_rejected_physical_20261103 import rank_counts


def test_eight_outcome_rank_counts_and_ties():
    result = rank_counts(np.arange(8), np.arange(8)[::-1])
    assert result == {"pairs": 28, "reversed": 28, "left_ties": 0, "right_ties": 0}
    ties = rank_counts(np.zeros(8), np.arange(8))
    assert ties["left_ties"] == 28 and ties["reversed"] == 0
    with pytest.raises(ValueError):
        rank_counts(np.arange(9), np.arange(9))
    with pytest.raises(ValueError):
        rank_counts([float("nan")] * 8, np.arange(8))
