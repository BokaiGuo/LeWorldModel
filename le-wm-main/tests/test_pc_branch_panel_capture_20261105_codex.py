import json

import torch

from experiments.pc_branch_panel_capture_20261105_codex import PanelCapture
from tests.test_pc_branch_full_pool_capture_20261105_codex import make
from experiments.pc_branch_effect_lookahead_20261105_codex import BranchEffectLookahead


def test_panel_capture_does_not_change_actions_and_keeps_warm_start(tmp_path):
    recorded = make(PanelCapture, capture_dir=tmp_path / "capture")
    baseline = make(BranchEffectLookahead)
    info = {"x": torch.zeros(1, 1), "id": torch.tensor([[7]]), "step_idx": torch.tensor([[0]])}
    a, b = recorded.solve(info)["actions"], baseline.solve(info)["actions"]
    assert torch.equal(a, b)
    assert len(list((tmp_path / "capture").glob("task_00_call_*.pt"))) == 20
    record = json.loads((tmp_path / "capture/replanning.jsonl").read_text())
    assert record["identity"]["id"] == [[7]]
    assert record["selected_actions"] == a.tolist()
    assert record["warm_start"] is None
