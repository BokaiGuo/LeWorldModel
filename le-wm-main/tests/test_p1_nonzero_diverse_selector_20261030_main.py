import numpy as np
import pytest

from experiments.spectral_reachability.p1_nonzero_diverse_selector_20261030_main import select


def test_zero_is_rejected_even_when_it_has_best_score():
    zero = np.zeros((5, 2), np.float32)
    zero[0, 0] = -0.0
    nonzero = zero.copy()
    nonzero[1, 0] = 0.2
    assert select([0, 1], [zero, nonzero], [0.0, 1.0]) == 1


def test_all_zero_abstains():
    assert select([0], [np.zeros((5, 2), np.float32)], [0.0]) is None


def test_tie_is_stable_and_nonfinite_rejected():
    one, two = np.ones((5, 2)), np.full((5, 2), 0.5)
    assert select([1, 0], [one, two], [2.0, 2.0]) == 0
    with pytest.raises(ValueError):
        select([0], [one], [float("nan")])
