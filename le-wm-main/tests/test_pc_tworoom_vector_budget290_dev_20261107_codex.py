import numpy as np
import pytest

from experiments.pc_tworoom_vector_budget290_dev_20261107_codex import choose


def test_incumbent_fallback_and_unretained_b_exclusion():
    a = np.full((8, 8), 5.)
    ids = np.arange(24)
    b = np.full(24, 10.)
    b[20] = -10.
    assert choose(a, [3., 4.], ids, b, [0, 1]) == ("inc", 0)
    b[10] = 1.
    assert choose(a, [3., 4.], ids, b, [0, 1]) == ("b", 10)


def test_ties_and_invalid_query_fail_closed():
    a = np.zeros((8, 8))
    assert choose(a, [0., 0.], np.arange(16), np.zeros(16), [0, 1]) == ("a", 0)
    with pytest.raises(ValueError):
        choose(a, [0., 0.], np.arange(15), np.zeros(15), [0, 1])
    with pytest.raises(ValueError):
        choose(a, [0., 0.], np.arange(16), np.zeros(16), [0, 0])
