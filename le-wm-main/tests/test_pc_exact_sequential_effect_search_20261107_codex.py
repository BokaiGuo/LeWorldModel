import numpy as np
import pytest

from experiments.pc_exact_sequential_effect_search_20261107_codex import next_pair


def test_history_changes_prediction_and_prevents_requery():
    a = np.ones((8, 10, 2))
    b = np.zeros((64, 10, 2))
    b[1, 0, 0] = 1.
    costs = np.ones((8, 8))
    costs[0] = 5.
    assert next_pair(a, b, costs, [], 0, "descriptor") == 64
    history = [(0, -100.), (1, 2.)]
    assert next_pair(a, b, costs, history, 2, "descriptor") == 2
    assert next_pair(a, b, costs, history[:1], 1, "descriptor") not in {0}


def test_invalid_history_rejected():
    a = np.zeros((8, 10, 2))
    b = np.zeros((64, 10, 2))
    c = np.zeros((8, 8))
    with pytest.raises(ValueError):
        next_pair(a, b, c, [(2, 0.), (2, 1.)], 2, "descriptor")
