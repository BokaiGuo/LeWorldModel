import pytest
import torch

from experiments.pc_input_permissions import RGB, freeze_audit
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession


class CachedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.encodes = 0
        self.eval()

    def get_cost(self, context, actions):
        b, s = actions.shape[:2]
        if "emb" not in context:
            self.encodes += 1
            context["emb"] = context["pixels"].reshape(b, 1, 1, 1).expand(b, s, 1, 1)
        if "goal_emb" not in context:
            self.encodes += 1
            context["goal_emb"] = context["goal"].reshape(b, 1, 1)
        return actions.sum(dim=(2, 3)) + context["emb"][:, :, 0, 0] + context["goal_emb"][:, 0, 0, None]


@pytest.fixture
def session(tmp_path):
    source = tmp_path / "checkpoint"
    source.write_text("test fixture")
    audit = freeze_audit(RGB, "lewm", [source], preprocessing="fixture", clock="fixture")
    return RGBObservationScoringSession(CachedModel(), audit)


def test_reuses_encodings_across_varying_batches_without_mutation(session):
    inputs = {"pixels": torch.ones(1), "goal": torch.ones(1) * 2, "action": torch.empty(1, 0)}
    with torch.inference_mode():
        first, _ = session.score(inputs, torch.zeros(1, 5, 1, 2))
        second, _ = session.score(inputs, torch.ones(1, 2, 1, 2))
    assert first.tolist() == [[3] * 5]
    assert second.tolist() == [[5] * 2]
    assert session._model.encodes == 2
    assert (session.cache_hits, session.cache_misses) == (1, 1)
    assert set(inputs) == {"pixels", "goal", "action"}


@pytest.mark.parametrize("changed", ["pixels", "goal", "action"])
def test_reject_changed_context(session, changed):
    inputs = {"pixels": torch.ones(1), "goal": torch.ones(1), "action": torch.ones(1)}
    with torch.inference_mode():
        session.score(inputs, torch.zeros(1, 2, 1, 2))
        inputs[changed].add_(1)
        with pytest.raises(ValueError, match="changed"):
            session.score(inputs, torch.zeros(1, 2, 1, 2))


def test_rejects_gradient_path_and_extra_state(session):
    inputs = {"pixels": torch.ones(1), "goal": torch.ones(1), "action": torch.ones(1)}
    with pytest.raises(ValueError, match="gradients"):
        session.score(inputs, torch.zeros(1, 2, 1, 2))
    with torch.inference_mode():
        with pytest.raises(ValueError):
            session.score({**inputs, "state": torch.ones(1)}, torch.zeros(1, 2, 1, 2))
