import numpy as np
import pytest

from experiments.pc_effect_margin_gate_20261107_codex import choose, outcome


def test_gate_strict_ties_and_outcomes():
    cost = np.ones((8, 9))
    cost[:, 1] = 0.5
    cost[:, 2] = 0.25
    selected = choose(cost, 0.5)
    assert selected["direct"].sum() == 16
    assert selected["gated"].sum() == 8
    distance = np.tile(np.arange(9.), (8, 1))
    term = np.zeros((8, 9, 25), dtype=bool)
    term[:, 2, -1] = True
    value = outcome(distance, term, selected["gated"])
    assert value["distance"].sum() == 16
    assert value["ever_success"].sum() == 8


def test_fail_closed():
    with pytest.raises(ValueError):
        choose(np.ones((8, 9)), -1)
    with pytest.raises(ValueError):
        outcome(np.ones((8, 9)), np.zeros((8, 9, 25)), np.zeros((8, 8), dtype=bool))
