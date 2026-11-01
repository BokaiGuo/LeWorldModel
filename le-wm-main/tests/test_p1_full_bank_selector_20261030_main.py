import numpy as np
import pytest

from experiments.spectral_reachability.p1_full_bank_selector_20261030_main import choose


def test_ties_keep_incumbent_and_zero_can_win():
    prefixes = np.ones((20, 5, 2), dtype=np.float32)
    scores = np.ones(20)
    assert choose(prefixes, scores) == 0
    prefixes[5] = 0
    scores[5] = 0
    assert choose(prefixes, scores) == 5


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_candidate_rejected(bad):
    scores = np.zeros(20)
    scores[9] = bad
    with pytest.raises(ValueError):
        choose(np.zeros((20, 5, 2)), scores)


def test_missing_candidate_rejected():
    with pytest.raises(ValueError):
        choose(np.zeros((19, 5, 2)), np.zeros(19))
