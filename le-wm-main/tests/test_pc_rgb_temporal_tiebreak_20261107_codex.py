import numpy as np
import pytest
from experiments.pc_rgb_temporal_tiebreak_20261107_codex import choose_index


def test_near_tie_uses_endpoint():
    assert choose_index([3., 3.0005, 3.1], [5., 1., 0.]) == 1


def test_outside_temporal_tolerance_never_wins():
    assert choose_index([3., 3.002], [5., 0.]) == 0


def test_stable_order_for_both_near_ties():
    assert choose_index([3., 3.0002], [1.0001, 1.]) == 0


@pytest.mark.parametrize("a,b", [([], []), ([1], [1, 2]), ([np.nan], [1])])
def test_invalid(a, b):
    with pytest.raises(ValueError):
        choose_index(a, b)
