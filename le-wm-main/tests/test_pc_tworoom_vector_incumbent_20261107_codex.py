import numpy as np
import pytest

from experiments.pc_tworoom_vector_incumbent_20261107_codex import choose


def test_choose_compares_only_own_choice_and_shared_incumbents():
    assert choose([1, -50, -20, -10, 2, 3], 0) == 0
    assert choose([8, 4, 5, 9, 3, 7], 2) == 4
    assert choose([8, 4, 5, 9, 3, 2], 1) == 5


def test_ties_keep_original_then_cem():
    assert choose([2, 4, 5, 9, 2, 2], 0) == 0
    assert choose([3, 4, 5, 9, 2, 2], 0) == 4


@pytest.mark.parametrize("scores,index", [([0] * 5, 0), ([0] * 6, 4),
                                           ([0, 0, np.nan, 0, 0, 0], 0)])
def test_invalid(scores, index):
    with pytest.raises(ValueError):
        choose(scores, index)
