import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from experiments.pc_branch_effect_lookahead_20261105_codex import BranchEffectLookahead


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.batches = []

    def get_cost(self, info, actions):
        self.batches.append(actions.detach().clone())
        physical = actions.reshape(*actions.shape[:-1], -1, 2) * torch.tensor([0.5, 0.2]) + torch.tensor([0.25, -0.4])
        assert torch.all(physical <= 1) and torch.all(physical >= -1)
        return (actions ** 2).sum((-1, -2))


@pytest.mark.parametrize("comparison", ["lookahead", "effect"])
def test_budget_legality_and_trace(tmp_path, comparison):
    model = RecordingModel()
    path = tmp_path / comparison
    solver = BranchEffectLookahead(model=model, action_mean=[0.25, -0.4], action_scale=[0.5, 0.2],
                                   comparison=comparison, batch_size=1, num_samples=300, n_steps=10,
                                   topk=30, device="cpu", seed=5, audit_path=path)
    solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                     n_envs=1, config=SimpleNamespace(action_block=2, horizon=4))
    result = solver.solve({"x": torch.zeros(1, 1)})
    assert sum(x.shape[1] for x in model.batches) == 3000
    assert len(model.batches) == 20
    record = json.loads(path.read_text())
    assert record["environments"][0]["scored"] == 3000
    assert len(record["environments"][0]["rounds"]) == 9
    assert torch.equal(torch.tensor(record["environments"][0]["selected_normalized_chain"]),
                       result["actions"][0])


def test_first_round_and_probes_match(tmp_path):
    models = []
    for mode in ("lookahead", "effect"):
        model = RecordingModel()
        solver = BranchEffectLookahead(model=model, action_mean=[0.25, -0.4], action_scale=[0.5, 0.2],
                                       comparison=mode, batch_size=1, num_samples=300, n_steps=10,
                                       topk=30, device="cpu", seed=5)
        solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                         n_envs=1, config=SimpleNamespace(action_block=2, horizon=4))
        solver.solve({"x": torch.zeros(1, 1)})
        models.append(model)
    assert torch.equal(models[0].batches[0], models[1].batches[0])
    assert torch.equal(models[0].batches[1], models[1].batches[1])
    assert not torch.equal(models[0].batches[2], models[1].batches[2])
