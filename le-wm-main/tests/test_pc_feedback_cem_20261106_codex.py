import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from experiments.pc_feedback_cem_20261106_codex import FeedbackCEM


class CostModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def get_cost(self, info, actions):
        self.calls.append(actions.detach().clone())
        return (actions ** 2).sum((-1, -2))


@pytest.mark.parametrize("selection", ["mean", "scored_pool"])
@pytest.mark.parametrize("steps,samples,topk", [(1, 20, 4), (3, 30, 6)])
def test_budget_scoring_and_warm_tail(tmp_path, selection, steps, samples, topk):
    model = CostModel()
    path = tmp_path / "audit.jsonl"
    solver = FeedbackCEM(model=model, action_mean=[0, 0], action_scale=[1, 1],
                         selection=selection, batch_size=1, num_samples=samples,
                         n_steps=steps, topk=topk, device="cpu", seed=5, audit_path=path)
    solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                     n_envs=1, config=SimpleNamespace(action_block=2, horizon=4))
    warm = torch.full((1, 2, 4), 0.1)
    result = solver.solve({"x": torch.zeros(1, 1)}, init_action=warm)
    audit = json.loads(path.read_text())["environments"][0]
    assert sum(x.shape[1] for x in model.calls) == samples * steps + 2
    assert audit["scored_model_candidates"] == samples * steps + 2
    assert torch.allclose(torch.tensor(audit["completed_warm_actions"])[:2], warm[0])
    assert np.isclose(result["costs"][0], float((result["actions"] ** 2).sum()))
    if selection == "scored_pool":
        assert np.isclose(result["costs"][0], min(audit["option_costs"]))
    else:
        assert audit["selected"] == 0


def test_no_warm_tail_is_not_claimed(tmp_path):
    model = CostModel()
    path = tmp_path / "audit.jsonl"
    solver = FeedbackCEM(model=model, action_mean=[0, 0], action_scale=[1, 1],
                         batch_size=1, num_samples=20, n_steps=1, topk=4,
                         device="cpu", seed=5, audit_path=path)
    solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                     n_envs=1, config=SimpleNamespace(action_block=2, horizon=4))
    solver.solve({"x": torch.zeros(1, 1)})
    audit = json.loads(path.read_text())["environments"][0]
    assert audit["warm_tail_length"] == 0
    assert audit["completed_warm_actions"] is None
