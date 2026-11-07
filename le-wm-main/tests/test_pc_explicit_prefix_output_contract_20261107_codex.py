import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from experiments.pc_explicit_prefix_output_contract_20261107_codex import (
    ExplicitPrefixSelectedActionOnly,
)
from experiments.pc_explicit_prefix_scored_incumbent_20261107_codex import (
    ExplicitPrefixScoredIncumbent,
)


class RecordingModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), device=device))
        self.calls = []

    def get_cost(self, info, actions):
        self.calls.append(actions.clone())
        return ((actions - 0.37) ** 2).sum((-1, -2))


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("mode", ["uniform", "lookahead"])
@pytest.mark.parametrize("seed", [5, 19, 83])
def test_output_contract_preserves_search(device, mode, seed, tmp_path):
    def run(cls, label):
        model = RecordingModel(device)
        path = tmp_path / label
        solver = cls(model=model, action_mean=[0, 0], action_scale=[1, 1],
                     prefix_mode=mode, batch_size=1, num_samples=100, n_steps=3,
                     topk=10, seed=seed, device=device, audit_path=path)
        solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                         n_envs=1, config=SimpleNamespace(action_block=2, horizon=5))
        result = solver.solve({"x": torch.zeros(1, 1, device=device)})
        return model, result, json.loads(path.read_text())["environments"][0]

    old_model, old, old_audit = run(ExplicitPrefixScoredIncumbent, "old")
    new_model, new, new_audit = run(ExplicitPrefixSelectedActionOnly, "new")
    assert len(old_model.calls) == len(new_model.calls) == 5
    assert all(torch.equal(a, b) for a, b in zip(old_model.calls, new_model.calls, strict=True))
    assert torch.equal(old["actions"], new["actions"])
    assert old["costs"] == new["costs"]
    assert old_audit == new_audit
    assert set(new) == {"actions", "costs", "selected_action_distribution"}
    assert new["selected_action_distribution"] == "unavailable"
    with pytest.raises(KeyError):
        _ = new["var"]
    with pytest.raises(KeyError):
        _ = new["mean"]
