import math

import pytest
import torch

from experiments.spectral_reachability import codex_p1_bounded_joint_20261101_exclusive as subject


def test_hard_constraint_recovery_and_tie():
    rows = [
        {"candidate": 0, "iteration": 0, "q1": 0.9, "q3": 0.1},
        {"candidate": 1, "iteration": 2, "q1": 1.0, "q3": 0.8},
        {"candidate": 1, "iteration": 1, "q1": 1.1, "q3": 0.8},
        {"candidate": 2, "iteration": 0, "q1": 1.0, "q3": 1.0},
    ]
    assert subject.choose(1.0, rows) == rows[2]
    assert subject.choose(0.05, rows) is None
    with pytest.raises(ValueError):
        subject.choose(1.0, [{"q1": math.nan, "q3": 0}])


def test_fixed_steps_and_projection():
    calls, rows, reverse_calls = [], [], []
    initial = torch.ones((3, 5, 2))
    def evaluate(action, iteration):
        calls.append(iteration)
        return torch.tensor(1.0), action[0].mean(), -action.sum()
    def reverse(loss, action, iteration):
        reverse_calls.append(iteration)
        return torch.autograd.grad(loss, action)[0]
    def record(iteration, action, q1, q3):
        rows.append((iteration, action.detach().clone()))
    subject.optimize(initial, evaluate, reverse, record)
    assert calls == list(range(9))
    assert reverse_calls == list(range(8))
    assert len(rows) == 9
    assert all(bool((x.abs() <= 1).all()) for _, x in rows)
    with pytest.raises(ValueError):
        subject.optimize(initial * 2, evaluate, reverse, record)
