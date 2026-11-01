import numpy as np
import torch

from experiments.spectral_reachability import codex_p1_bounded_joint_diag32_20261101_exclusive as diag


def test_broadcast_matches_flat_tiled_normalization():
    action = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    mean = torch.tensor([-0.0087, 0.0068])
    std = torch.tensor([0.2019, 0.2002])
    assert torch.equal(((action - mean) / std).reshape(-1),
                       (action.reshape(-1) - mean.repeat(5)) / std.repeat(5))


def test_fixed_budget_gradient_and_projection():
    records, reverse = [], []
    initial = torch.full((3, 5, 2), 0.9)
    def evaluate(action, iteration):
        return torch.tensor(0.0), action[0].square().mean(), -action.sum()
    def backward(loss, action, iteration):
        reverse.append(iteration)
        return torch.autograd.grad(loss, action)[0]
    def record(iteration, action, q1, q3, loss, gradient):
        records.append((iteration, action.detach().clone(), None if gradient is None else gradient.clone()))
    diag.optimize(initial, evaluate, backward, record)
    assert len(records) == 33 and reverse == list(range(32))
    assert records[-1][2] is None
    assert torch.allclose(records[0][2], -torch.ones_like(initial))
    assert all(bool((row[1].abs() <= 1).all()) for row in records)
    assert np.isclose(records[-1][1].max().item(), 1.0)
