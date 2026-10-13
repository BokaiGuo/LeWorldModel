import numpy as np
import torch

from experiments.run_pusht_contact_predictor_teacher_forced_repair import (
    normalized_context_action_sequence,
    predict_latent_sequence,
)


def test_teacher_forced_action_sequence_has_context_plus_future_rows():
    case = {"context_actions": np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)}
    actions = normalized_context_action_sequence(case, np.asarray([0.55, -0.2]), 4)
    assert actions.shape == (6, 10)
    assert np.isfinite(actions).all()
    np.testing.assert_allclose(actions[2], actions[3])
    np.testing.assert_allclose(actions[3], actions[5])


class _IdentityAction:
    def __call__(self, actions):
        return actions[..., :1]


class _AdditivePredictor:
    def __init__(self):
        self.action_encoder = _IdentityAction()

    def predict(self, emb, act_emb):
        return emb + act_emb


def test_teacher_forced_rollout_returns_one_prediction_per_future_step():
    model = _AdditivePredictor()
    context = torch.zeros(2, 3, 1)
    actions = torch.zeros(2, 6, 1)
    actions[:, 2:, 0] = 1.0
    predicted = predict_latent_sequence(model, context, actions)
    assert predicted.shape == (2, 4, 1)
    assert torch.isfinite(predicted).all()
    # The first prediction uses the final context action, then each future row.
    torch.testing.assert_close(predicted[:, :, 0], torch.tensor([[1.0, 2.0, 3.0, 4.0]]).expand(2, -1))
    teacher_forced = predict_latent_sequence(
        model, context, actions, teacher_forcing_targets=torch.zeros(2, 4, 1)
    )
    torch.testing.assert_close(teacher_forced[:, :, 0], torch.ones(2, 4))


def test_teacher_forced_rollout_rejects_short_action_context():
    model = _AdditivePredictor()
    context = torch.zeros(1, 3, 1)
    actions = torch.zeros(1, 3, 1)
    try:
        predict_latent_sequence(model, context, actions)
    except ValueError as error:
        assert "at least one future row" in str(error)
    else:
        raise AssertionError("short action context must be rejected")
