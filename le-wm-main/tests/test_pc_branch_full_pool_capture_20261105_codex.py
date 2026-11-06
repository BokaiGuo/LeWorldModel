from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from experiments.pc_branch_full_pool_capture_20261105_codex import FullPoolCapture
from experiments.pc_branch_effect_lookahead_20261105_codex import BranchEffectLookahead


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_cost(self, info, actions):
        info["predicted_emb"] = actions.clone()
        info["goal_emb"] = torch.zeros(1, 1, 1, actions.shape[-1])
        return actions.square().sum((-1, -2))


def make(cls, **kwargs):
    solver = cls(model=Model(), action_mean=[0, 0], action_scale=[1, 1], comparison="effect",
                 batch_size=1, num_samples=300, n_steps=10, topk=30, device="cpu", seed=5, **kwargs)
    solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                     n_envs=1, config=SimpleNamespace(action_block=2, horizon=4))
    return solver


def test_capture_preserves_actions_and_budget(tmp_path):
    capture = make(FullPoolCapture, capture_dir=tmp_path / "capture")
    control = make(BranchEffectLookahead)
    info = {"x": torch.zeros(1, 1)}
    assert torch.equal(capture.solve(info)["actions"], control.solve(info)["actions"])
    files = sorted((tmp_path / "capture").glob("*.pt"))
    assert len(files) == 20
    data = [torch.load(p, weights_only=False) for p in files]
    assert sum(item["actions"].shape[1] for item in data) == 3000
    assert all(torch.equal(item["prediction"], item["actions"]) for item in data)
    with pytest.raises(FileExistsError):
        make(FullPoolCapture, capture_dir=tmp_path / "capture")
