import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from experiments.pc_native_prefix_conditional_20261105_codex import NativePrefixConditional


class PrefixTask(torch.nn.Module):
    def __init__(self, invalid=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self.invalid = invalid

    def get_cost(self, info, actions):
        self.calls.append(actions.clone())
        prefix = actions[:, :, :2].sum((-1, -2))
        tail = actions[:, :, 2:].sum((-1, -2))
        result = (tail - 2 * prefix).square()
        return result * float("nan") if self.invalid else result


def build(model, mode="conditional", **kwargs):
    instance = NativePrefixConditional(model=model, mode=mode, batch_size=2,
                                       num_samples=20, n_steps=3, topk=3,
                                       device="cpu", seed=5, **kwargs)
    instance.configure(action_space=Box(-1, 1, shape=(3, 2), dtype=np.float32),
                       n_envs=3, config=SimpleNamespace(action_block=1, horizon=4))
    return instance


def test_equal_budget_common_probes_and_conditional_tail(tmp_path):
    models = [PrefixTask(), PrefixTask()]
    paths = [tmp_path / "conditional.jsonl", tmp_path / "shared.jsonl"]
    for model, mode, path in zip(models, ("conditional", "shared"), paths):
        result = build(model, mode=mode, audit_path=path).solve({"x": torch.zeros(3, 1)})
        assert result["actions"].shape == (3, 4, 2)
        record = json.loads(path.read_text())
        assert all(it["scored_per_environment"] == 20 for it in record["iterations"])
        assert [call.shape[1] for call in model.calls] == [20, 16, 4, 16, 4] * 2
    assert torch.equal(models[0].calls[0], models[1].calls[0])
    assert torch.equal(models[0].calls[1], models[1].calls[1])
    assert not torch.equal(models[0].calls[2], models[1].calls[2])
    # Proposals preserve one of the two previously probed prefixes.
    partial, proposal = models[0].calls[1], models[0].calls[2]
    assert torch.equal(proposal[:, :2, :2], partial[:, -8:-7, :2].expand(-1, 2, -1, -1))
    assert torch.equal(proposal[:, 2:, :2], partial[:, -4:-3, :2].expand(-1, 2, -1, -1))


def test_fail_closed_on_nonfinite():
    with pytest.raises(ValueError, match="invalid candidate costs"):
        build(PrefixTask(invalid=True)).solve({"x": torch.zeros(3, 1)})


def test_rejects_prefix_as_long_as_horizon():
    with pytest.raises(ValueError, match="prefix horizon"):
        build(PrefixTask(), prefix_steps=4).solve({"x": torch.zeros(3, 1)})
