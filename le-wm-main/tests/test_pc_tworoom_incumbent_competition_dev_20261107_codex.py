import numpy as np
import pytest

from experiments.pc_tworoom_incumbent_competition_dev_20261107_codex import select


def test_competition_respects_arm_and_stable_tie():
    assert select([2, 0, 0, 0, 1, 3], 0) == 4
    assert select([1, 0, 0, 0, 1, 1], 0) == 0
    assert select([4, 0, 0, 0, 1, 1], 0) == 4


def test_invalid_or_nonfinite_rejected():
    with pytest.raises(ValueError):
        select([1, 2], 0)
    with pytest.raises(ValueError):
        select([1, 2, 3, 4, np.nan, 6], 0)
    with pytest.raises(ValueError):
        select([1, 2, 3, 4, 5, 6], 4)
