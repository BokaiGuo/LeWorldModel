import pytest
import torch
from torch import nn
from unittest.mock import patch

from experiments.pc_pusht_conditional_rollout_parity_20261107_codex import native_from_embedding
from experiments.pc_pusht_shared_prefix_rescore_20261107_codex import conditional_rollout
from jepa import JEPA


class WindowPredictor(nn.Module):
    def forward(self, state, action):
        return state + action.cumsum(dim=1)


def test_native_and_conditional_match_and_restore_encoder():
    model = JEPA(nn.Identity(), WindowPredictor(), nn.Identity())
    original_encode = model.encode.__func__
    initial = torch.tensor([[0.2, -0.1]])
    actions = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2) / 10
    expected = conditional_rollout(model, initial, actions, 0)
    actual = native_from_embedding(model, initial, actions)
    torch.testing.assert_close(actual, expected)
    assert model.encode.__func__ is original_encode


def test_native_rejects_nonshared_initial_batch():
    model = JEPA(nn.Identity(), WindowPredictor(), nn.Identity())
    with pytest.raises(ValueError, match="shared initial"):
        native_from_embedding(model, torch.zeros(2, 2), torch.zeros(2, 3, 2))


def test_native_three_frame_history_matches_conditional_rollout():
    model = JEPA(nn.Identity(), WindowPredictor(), nn.Identity())
    history = torch.tensor([[0.2, -0.1], [0.4, 0.3], [0.5, -0.2]])
    actions = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2) / 10
    manual = conditional_rollout(model, history, actions, 2)
    with patch.object(model, "encode", side_effect=lambda info: {**info, "emb": history[None]}):
        result = model.rollout({"pixels": torch.zeros(1, 3, 3, 1)}, actions[None])
    torch.testing.assert_close(manual, result["predicted_emb"][0, :, 3:])
