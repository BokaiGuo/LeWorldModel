import pytest
import torch

from experiments.pc_teacher_forced_rollout_dev_20261106_codex import teacher_predictions


class RecordingModel:
    def __init__(self):
        self.histories = []

    def action_encoder(self, actions):
        return actions

    def predict(self, observations, actions):
        self.histories.append((observations.clone(), actions.clone()))
        return observations + actions


def test_teacher_forcing_uses_true_past_and_current_action():
    model = RecordingModel()
    observations = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    actions = (10 + torch.arange(5, dtype=torch.float32)).reshape(5, 1)
    result = teacher_predictions(model, observations, actions)
    assert torch.equal(result[:, 0], torch.tensor([10., 12., 14., 16., 18.]))
    assert [h[0].shape[1] for h in model.histories] == [1, 2, 3, 3, 3]
    assert torch.equal(model.histories[-1][0][0, :, 0], torch.tensor([2., 3., 4.]))
    assert torch.equal(model.histories[-1][1][0, :, 0], torch.tensor([12., 13., 14.]))


def test_rejects_misaligned_clock():
    with pytest.raises(ValueError):
        teacher_predictions(RecordingModel(), torch.zeros(2, 1), torch.zeros(2, 1))
