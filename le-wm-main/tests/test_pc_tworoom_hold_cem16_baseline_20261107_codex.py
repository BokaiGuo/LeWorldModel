import numpy as np
import pytest
import torch

from experiments import pc_tworoom_hold_cem16_baseline_20261107_codex as run


class Scaler:
    mean_ = np.zeros(2)
    scale_ = np.ones(2)

    def transform(self, x):
        return np.asarray(x)

    def inverse_transform(self, x):
        return np.asarray(x)


class Boundary:
    def __init__(self):
        self.sizes = []

    def score(self, ctx, actions):
        self.sizes.append(tuple(actions.shape))
        return actions.square().sum(dim=(-1, -2)), {}


def test_cem16_budget_clock_selection_and_determinism():
    boundary = Boundary()
    context = {"pixels": torch.zeros(1)}
    actions, pool, costs, selected, _ = run.solve(boundary, context, Scaler(), None, 15, 7)
    assert boundary.sizes == [(1, 1, 3, 10), (1, 5, 3, 10), (1, 5, 3, 10), (1, 5, 3, 10)]
    assert selected == 0
    np.testing.assert_array_equal(actions, np.zeros((15, 2)))
    assert pool.shape == (16, 25, 2)
    assert costs.shape == (16,)
    _, pool_again, costs_again, _, _ = run.solve(Boundary(), context, Scaler(), None, 15, 7)
    np.testing.assert_array_equal(pool, pool_again)
    np.testing.assert_array_equal(costs, costs_again)


def test_rejects_invalid_remaining_deadline():
    with pytest.raises(ValueError):
        run.solve(Boundary(), {"pixels": torch.zeros(1)}, Scaler(), None, 14, 7)
