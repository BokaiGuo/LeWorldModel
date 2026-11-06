import json
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from experiments.pc_bounded_prefix_full_pool_novelty_20261105_codex import (
    BoundedPrefixFullPoolNovelty,
    candidate_set_difference,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.counts = []

    def get_cost(self, info, actions):
        self.counts.append(actions.shape[1])
        physical = actions * 0.5 + 0.2
        assert torch.all((physical >= -1) & (physical <= 1))
        return (actions ** 2).sum((-1, -2))


def test_full_pool_matches_ignore_order_and_duplicates():
    proposals = torch.tensor([[[[1.0]], [[2.0]], [[1.0]]]])
    pool = torch.tensor([[[[3.0]], [[1.0]], [[1.0]]]])
    novel, _ = candidate_set_difference(proposals, pool)
    assert novel.tolist() == [[False, True, False]]


def test_extra_audit_does_not_change_solver_and_all_queries_are_legal(tmp_path):
    results, models, logs = [], [], []
    for enabled in (False, True):
        model = Model()
        path = tmp_path / str(enabled)
        solver = BoundedPrefixFullPoolNovelty(
            model=model, action_mean=[0.2, 0.2], action_scale=[0.5, 0.5],
            counterfactual_audit=enabled, batch_size=1, num_samples=20, n_steps=3,
            topk=3, device="cpu", seed=7, audit_path=path)
        solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                         n_envs=1, config=SimpleNamespace(action_block=1, horizon=4))
        results.append(solver.solve({"x": torch.zeros(1, 1)})["actions"])
        logs.append(json.loads(path.read_text()))
        models.append(model)
    assert torch.equal(*results)
    assert models[0].counts == [20, 16, 4, 16, 4]
    assert models[1].counts == [20, 16, 4, 4, 16, 4, 4]
    for original, audited in zip(logs[0]["iterations"], logs[1]["iterations"]):
        assert original["proposal_elites"] == audited["proposal_elites"]
        assert original["probe_elites"] == audited["probe_elites"]
        assert original["sign_disagreements"] == audited["sign_disagreements"]
        if audited["step"]:
            record = audited["full_pool_audit"]["records"][0]
            assert len(record["unique_mask"]) == 4
            assert len(record["proposal_elite_mask"]) == 4
