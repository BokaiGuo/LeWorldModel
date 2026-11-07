import numpy as np
import pytest

from experiments.pc_exact_descriptor_combination_proposal_20261107_codex import choose, proposal_predictions, tasks


def test_ties_break_by_flat_prefix_then_tail():
    values = np.ones((8, 64))
    values[1, 2] = values[0, 3] = 0
    assert choose(values, 2) == [3, 66]
    with pytest.raises(ValueError):
        choose(np.full((8, 64), np.nan))


def test_predictor_uses_only_a_and_new_b_actions():
    a = np.zeros((8, 10, 2))
    b = np.ones((64, 10, 2))
    a_cost = np.arange(64).reshape(8, 8).astype(float)
    pred = proposal_predictions(a, b, a_cost, np.zeros(8), 42)
    assert all(pred[k].shape == (8, 64) for k in ("row_mean", "additive", "descriptor", "shuffled"))
    assert len(tasks()) == 24
    assert len({tuple(t["start"] + t["goal"]) for t in tasks()}) == 24
