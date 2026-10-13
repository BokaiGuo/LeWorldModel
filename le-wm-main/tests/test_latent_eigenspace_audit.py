import torch

from experiments.latent_eigenspace_audit import autoregressive_horizon_predictions


class AdditivePredictor:
    def __init__(self, history_size: int = 2):
        self.predictor = type("Predictor", (), {})()
        self.predictor.pos_embedding = torch.zeros(1, history_size, 1)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return emb + act_emb


def test_autoregressive_horizon_predictions_align_all_start_points():
    model = AdditivePredictor(history_size=2)
    emb = torch.zeros(1, 6, 1)
    act_emb = torch.ones(1, 6, 1)

    predictions = autoregressive_horizon_predictions(model, emb, act_emb, horizons=[1, 3])

    assert predictions[1].shape == (1, 3, 1)
    assert predictions[3].shape == (1, 3, 1)
    assert torch.allclose(predictions[1], torch.ones(1, 3, 1))
    assert torch.allclose(predictions[3], torch.full((1, 3, 1), 3.0))


def test_autoregressive_horizon_predictions_use_future_actions():
    model = AdditivePredictor(history_size=3)
    emb = torch.zeros(1, 5, 1)
    act_emb = torch.tensor([[[1.0], [2.0], [4.0], [8.0], [16.0]]])

    predictions = autoregressive_horizon_predictions(model, emb, act_emb, horizons=[2])

    expected = torch.tensor([[[3.0], [6.0], [12.0]]])
    assert torch.allclose(predictions[2], expected)
