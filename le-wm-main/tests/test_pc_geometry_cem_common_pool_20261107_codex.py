import numpy as np
import pytest

from experiments.pc_geometry_cem_common_pool_20261107_codex import select


def test_selection_and_tie_policy():
    assert select([3., 1., 2.]) == 1
    assert select([1., 1., 2.]) == 0


@pytest.mark.parametrize("scores", [[1, 2], [1, np.nan, 3], [1, 2, np.inf]])
def test_invalid_scores_fail_closed(scores):
    with pytest.raises(ValueError):
        select(scores)
