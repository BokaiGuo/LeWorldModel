import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from experiments.pc_explicit_prefix_program_20261106_codex import ExplicitPrefixProgram


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def get_cost(self, info, actions):
        self.calls.append(actions.clone())
        return (actions ** 2).sum((-1, -2))


@pytest.mark.parametrize("mode", ["uniform", "lookahead"])
def test_fixed_prefix_and_budget(tmp_path, mode):
    model = RecordingModel()
    path = tmp_path / "audit"
    solver = ExplicitPrefixProgram(model=model, action_mean=[0, 0], action_scale=[1, 1],
                                   prefix_mode=mode, batch_size=1, num_samples=100,
                                   n_steps=3, topk=10, seed=5, device="cpu", audit_path=path)
    solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                     n_envs=1, config=SimpleNamespace(action_block=2, horizon=5))
    result = solver.solve({"x": torch.zeros(1, 1)})
    row = json.loads(path.read_text())["environments"][0]
    assert sum(call.shape[1] for call in model.calls) == row["scored_model_candidates"] == 632
    assert len(model.calls) == 5
    prefixes = torch.tensor(row["prefixes"])
    for call in model.calls[1:4]:
        branches = call.reshape(2, 100, 5, 4)
        assert torch.equal(branches[:, :, :2], prefixes[:, None].expand(-1, 100, -1, -1))
    assert torch.equal(result["actions"][0, :2], prefixes[row["winner"]])
    assert np.isclose(result["costs"][0], float(result["actions"].square().sum()))


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        ExplicitPrefixProgram(model=RecordingModel(), action_mean=[0, 0], action_scale=[1, 1],
                              num_samples=100, n_steps=2, topk=10)
