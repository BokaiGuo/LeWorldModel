import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from experiments.pc_explicit_prefix_program_20261106_codex import ExplicitPrefixProgram
from experiments.pc_explicit_prefix_scored_incumbent_20261107_codex import ExplicitPrefixScoredIncumbent


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def get_cost(self, info, actions):
        self.calls.append(actions.clone())
        return ((actions - .37) ** 2).sum((-1, -2))


@pytest.mark.parametrize("mode", ["uniform", "lookahead"])
def test_unchanged_pool_and_best_scored_selection(mode, tmp_path):
    def run(cls, path):
        model = RecordingModel()
        solver = cls(model=model, action_mean=[0, 0], action_scale=[1, 1], prefix_mode=mode,
                     batch_size=1, num_samples=100, n_steps=3, topk=10, seed=5,
                     device="cpu", audit_path=path)
        solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                         n_envs=1, config=SimpleNamespace(action_block=2, horizon=5))
        result = solver.solve({"x": torch.zeros(1, 1)})
        return model, result
    old_model, old = run(ExplicitPrefixProgram, tmp_path / "old")
    new_model, new = run(ExplicitPrefixScoredIncumbent, tmp_path / "new")
    assert len(old_model.calls) == len(new_model.calls) == 5
    for first, second in zip(old_model.calls, new_model.calls, strict=True):
        assert torch.equal(first, second)
    candidates = torch.cat(new_model.calls, dim=1)
    costs = ((candidates - .37) ** 2).sum((-1, -2))[0]
    assert candidates.shape[1] == 632
    assert torch.equal(new["actions"][0], candidates[0, costs.argmin()])
    assert np.isclose(new["costs"][0], float(costs.min()))
    assert new["costs"][0] <= old["costs"][0] + 1e-6
    row = json.loads((tmp_path / "new").read_text())["environments"][0]
    assert row["scored_model_candidates"] == 632
    assert row["best_source"]["kind"] in ("initial", "sample", "mean")
