import pytest
import torch

from experiments.pc_teacher_forced_rollout_dev_20261106_codex import teacher_predictions


class RecordingModel:
    def __init__(self):
        self.calls = []

    def action_encoder(self, action):
        return action

    def predict(self, observations, actions):
        self.calls.append((observations.clone(), actions.clone()))
        return observations + 10 * actions


def test_teacher_uses_current_action_and_only_past_observations():
    model = RecordingModel()
    observations = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    actions = torch.arange(10, 15, dtype=torch.float32).reshape(5, 1)
    result = teacher_predictions(model, observations, actions, history_size=3)
    torch.testing.assert_close(result[:, 0], observations[:-1, 0] + 10 * actions[:, 0])
    assert [x.shape[1] for x, _ in model.calls] == [1, 2, 3, 3, 3]
    torch.testing.assert_close(model.calls[-1][0][0, :, 0], observations[2:5, 0])
    torch.testing.assert_close(model.calls[-1][1][0, :, 0], actions[2:5, 0])


def test_teacher_rejects_inconsistent_clock():
    with pytest.raises(ValueError):
        teacher_predictions(RecordingModel(), torch.zeros(5, 1), torch.zeros(5, 1))
