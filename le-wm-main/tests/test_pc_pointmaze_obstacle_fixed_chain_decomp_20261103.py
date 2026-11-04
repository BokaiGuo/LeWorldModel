import numpy as np
import pytest

from experiments.pc_pointmaze_obstacle_fixed_chain_decomp_20261103 import rank_counts


def test_rank_counts_orders_and_ties():
    x = np.arange(9.)
    assert rank_counts(x, x)["reversed"] == 0
    assert rank_counts(x, -x)["reversed"] == 36
    result = rank_counts(np.zeros(9), x)
    assert result["left_ties"] == 36
    assert result["right_ties"] == 0


@pytest.mark.parametrize("bad", [np.zeros(8), np.full(9, np.nan)])
def test_rank_counts_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        rank_counts(bad, np.arange(9.))
