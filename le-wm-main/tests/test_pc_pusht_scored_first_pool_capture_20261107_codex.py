from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from experiments.pc_explicit_prefix_output_contract_20261107_codex import ExplicitPrefixSelectedActionOnly
from experiments.pc_pusht_scored_first_pool_capture_20261107_codex import ScoredFirstPoolCapture


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_cost(self, info, action):
        pred = action.sum((-1, -2))
        info["predicted_emb"] = pred[..., None, None]
        info["goal_emb"] = torch.zeros((len(action), 1, 1))
        return pred


def test_capture_preserves_solver_and_records_every_query(tmp_path):
    def run(cls, **kwargs):
        solver = cls(model=Model(), action_mean=[0, 0], action_scale=[1, 1], prefix_mode="lookahead",
                     batch_size=1, num_samples=100, n_steps=3, topk=10, seed=5, device="cpu", **kwargs)
        solver.configure(action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                         n_envs=1, config=SimpleNamespace(action_block=2, horizon=5))
        return solver.solve({"x": torch.zeros(1, 1)})

    original = run(ExplicitPrefixSelectedActionOnly)
    captured = run(ScoredFirstPoolCapture, capture_dir=tmp_path / "pool")
    assert torch.equal(original["actions"], captured["actions"])
    assert original["costs"] == captured["costs"]
    batches = [torch.load(path, weights_only=True) for path in sorted((tmp_path / "pool").glob("*.pt"))]
    assert [x["actions"].shape[1] for x in batches] == [30, 200, 200, 200, 2]
    assert sum(x["actions"].shape[1] for x in batches) == 632
