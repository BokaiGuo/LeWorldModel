from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from experiments.pc_bounded_cem_20261105_codex import BoundedCEM, normalized_bounds
from experiments.pc_bounded_branch_cem_20261105_codex import BoundedBranchCEM


class CheckPhysicalActions(torch.nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.mean = torch.tensor(mean)
        self.scale = torch.tensor(scale)
        self.calls = 0

    def get_cost(self, info, actions):
        self.calls += 1
        physical = actions.reshape(*actions.shape[:-1], -1, 2) * self.scale + self.mean
        assert torch.all(physical <= 1) and torch.all(physical >= -1)
        return -actions.sum((-1, -2))


def test_bounds_preserve_action_block_order_and_fail_closed():
    low, high = normalized_bounds([0.25, -0.4], [0.5, 0.2], 4)
    assert torch.equal(low[:2], low[2:])
    assert torch.equal(high[:2], high[2:])
    with pytest.raises(ValueError):
        normalized_bounds([0.0], [0.0], 2)
    with pytest.raises(ValueError):
        normalized_bounds([0.0, 0.0], [1.0, 1.0], 3)


def test_all_scored_and_returned_actions_are_physically_legal(tmp_path):
    mean, scale = [0.25, -0.4], [0.5, 0.2]
    model = CheckPhysicalActions(mean, scale)
    solver = BoundedCEM(model=model, action_mean=mean, action_scale=scale,
                        batch_size=2, num_samples=20, n_steps=3, topk=3,
                        device="cpu", seed=5, audit_path=tmp_path / "audit")
    solver.configure(action_space=Box(-1, 1, shape=(3, 2), dtype=np.float32),
                     n_envs=3, config=SimpleNamespace(action_block=2, horizon=4))
    actions = solver.solve({"x": torch.zeros(3, 1)})["actions"]
    physical = actions.reshape(3, 4, 2, 2) * torch.tensor(scale) + torch.tensor(mean)
    assert torch.all(physical <= 1) and torch.all(physical >= -1)
    assert model.calls == 6


def test_branch_fixed_budget_and_legality(tmp_path):
    class CountActions(CheckPhysicalActions):
        def __init__(self):
            super().__init__([0.25, -0.4], [0.5, 0.2])
            self.scored = 0

        def get_cost(self, info, actions):
            self.scored += actions.shape[0] * actions.shape[1]
            return super().get_cost(info, actions)

    model = CountActions()
    solver = BoundedBranchCEM(model=model, action_mean=[0.25, -0.4], action_scale=[0.5, 0.2],
                              batch_size=1, num_samples=300, n_steps=10, topk=30,
                              device="cpu", seed=5, audit_path=tmp_path / "audit")
    solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                     n_envs=1, config=SimpleNamespace(action_block=2, horizon=4))
    result = solver.solve({"x": torch.zeros(1, 1)})
    assert model.scored == 3000
    physical = result["actions"].reshape(1, 4, 2, 2) * torch.tensor([0.5, 0.2]) + torch.tensor([0.25, -0.4])
    assert torch.all(physical <= 1) and torch.all(physical >= -1)
