import numpy as np
import pytest

from experiments.pc_tworoom_retention_numerical_stress_20261107_codex import select


def test_selection_ties_and_shift_invariance():
    scores = np.tile(np.arange(8, dtype=float), (8, 1))
    assert select(scores)[:2] == (0, 1)
    shifted = scores + np.arange(8, dtype=float)[:, None]
    assert select(shifted)[:2] == (0, 1)
    scores[3, 0] = -5
    assert select(scores)[0] == 3


def test_nonfinite_and_incorrect_shape_fail():
    with pytest.raises(ValueError):
        select(np.zeros((8, 7)))
    x = np.zeros((8, 8))
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        select(x)
