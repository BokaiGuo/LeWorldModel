import json
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from experiments.pc_bounded_prefix_full_pool_novelty_20261105_codex import BoundedPrefixFullPoolNovelty
from experiments.pc_prefix_novel_same_refinement_20261105_codex import PrefixNovelSameRefinement


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.counts = []

    def get_cost(self, info, actions):
        self.counts.append(actions.shape[1])
        physical = actions * 0.5 + 0.2
        assert torch.all((physical >= -1) & (physical <= 1))
        prefix = actions[:, :, :2].sum((-1, -2))
        tail = actions[:, :, 2:].sum((-1, -2))
        return (tail - 2 * prefix).square()


def test_paired_refinement_does_not_change_main_actions(tmp_path):
    outputs, models = [], []
    for cls, name in ((BoundedPrefixFullPoolNovelty, "pool"),
                      (PrefixNovelSameRefinement, "refine")):
        model = Model()
        solver = cls(model=model, action_mean=[0.2, 0.2], action_scale=[0.5, 0.5],
                     batch_size=1, num_samples=20, n_steps=10, topk=3,
                     device="cpu", seed=5, audit_path=tmp_path / name)
        solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                             n_envs=10, config=SimpleNamespace(action_block=1, horizon=4))
        outputs.append(solver.solve({"x": torch.zeros(10, 1)})["actions"])
        models.append(model)
    assert torch.equal(*outputs)
    logs = json.loads((tmp_path / "refine").read_text())
    valid = [row for row in logs["refinement"] if not row["no_novel_candidate"]]
    assert valid
    assert all(row["additional_full_chain_scores"] == 192 for row in valid)
    assert models[1].counts.count(64) == 3 * len(valid)
    for row in valid:
        assert all(after <= before for before, after in zip(row["original_costs"], row["refined_costs"]))
