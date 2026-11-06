from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from experiments.pc_branch_second_replan_capture_20261106_codex import SecondReplanCapture
from test_pc_branch_full_pool_capture_20261105_codex import Model


def test_replan_identity_and_calls(tmp_path):
    solver = SecondReplanCapture(model=Model(), action_mean=[0, 0], action_scale=[1, 1],
                                 comparison="effect", batch_size=1, num_samples=300, n_steps=10,
                                 topk=30, device="cpu", seed=5, capture_dir=tmp_path / "capture")
    solver.configure(action_space=Box(-1, 1, shape=(50, 2), dtype=np.float32),
                     n_envs=50, config=SimpleNamespace(action_block=2, horizon=4))
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        solver.solve({"x": torch.zeros(50, 1), "id": torch.arange(50)[:, None]})
        solver.solve({"x": torch.zeros(1, 1), "id": torch.tensor([[25]])})
    finally:
        torch.set_num_threads(old_threads)
    files = sorted((tmp_path / "capture").glob("task_25_replan_call_*.pt"))
    assert len(files) == 20
    assert sum(torch.load(f, weights_only=False)["actions"].shape[1] for f in files) == 3000
    assert not list((tmp_path / "capture").glob("task_15_replan_call_*.pt"))
