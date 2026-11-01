import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_causal_innovation import corrected_prediction, innovation_from_executed_history


class _CausalWindowModel:
    def encode(self, info):
        return {"emb": info["pixels"].mean(dim=(-1, -2, -3)).unsqueeze(-1)}

    def action_encoder(self, actions):
        return actions

    def predict(self, embeddings, actions):
        # Distinguish every temporal slot so a dropped/doubled action fails.
        weights = torch.tensor([1., 10., 100.], device=actions.device)
        return embeddings[:, -1:] + (actions[..., 0] * weights).sum(dim=1).reshape(-1, 1, 1)


def test_innovation_uses_exact_predecessor_window_and_last_executed_action():
    images = torch.tensor([1., 2., 3.]).reshape(1, 3, 1, 1, 1)
    history = torch.tensor([4., 5., 6.]).reshape(1, 3, 1)
    innovation, receipt = innovation_from_executed_history(_CausalWindowModel(), images, history, torch.tensor([700.]))
    assert innovation.item() == 700 - (3 + 4 + 50 + 600)
    assert not receipt["future_labels_used"]
    assert receipt["coefficient"] == 1.


def test_same_causal_innovation_applies_to_all_candidates_without_fitting():
    innovation = torch.tensor([.1, -.2])
    original = innovation.clone()
    for candidate in (torch.tensor([.3, .4]), torch.tensor([10., -10.])):
        assert torch.equal(corrected_prediction(candidate, innovation), candidate + innovation)
    assert torch.equal(innovation, original)


@pytest.mark.parametrize("kind", ["batch", "short", "shape", "finite"])
def test_innovation_rejects_invalid_causal_context(kind):
    images, history, observed = torch.zeros(1, 3, 1, 1, 1), torch.zeros(1, 3, 1), torch.zeros(1)
    if kind == "batch":
        images = images.expand(2, -1, -1, -1, -1)
    elif kind == "short":
        history = history[:, :2]
    elif kind == "shape":
        observed = observed.reshape(1, 1)
    else:
        observed[:] = float("nan")
    with pytest.raises(ValueError):
        innovation_from_executed_history(_CausalWindowModel(), images, history, observed)
