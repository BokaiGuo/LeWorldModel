import numpy as np
import pytest
import torch

from experiments.pc_input_permissions import tensor_receipt
from experiments.pc_rgb_feedback_frame_clock_replay_20261107_codex import check_inputs


def test_replayed_frame_receipt_rejects_wrong_clock():
    frames = [np.full((2, 2, 3), i, np.uint8) for i in (1, 2, 3)]
    goal = np.full((2, 2, 3), 4, np.uint8)
    transform = lambda x: torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
    receipt = {"track_id": "primary_rgb", "operation": "score", "inputs": {
        "pixels": tensor_receipt(torch.stack([transform(x) for x in frames])[None, None]),
        "goal": tensor_receipt(transform(goal)[None, None, None])}}
    check_inputs([receipt], frames, goal, "history", transform)
    with pytest.raises(ValueError, match="rendered image"):
        check_inputs([receipt], frames[::-1], goal, "history", transform)
