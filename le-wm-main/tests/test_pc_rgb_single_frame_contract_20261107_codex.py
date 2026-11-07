import numpy as np
import pytest
import torch
from types import SimpleNamespace

from experiments.pc_input_permissions import RGB, RGBLeWM, freeze_audit
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest
from jepa import JEPA


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.eval()

    def get_cost(self, context, actions):
        context["predicted_emb"] = actions
        return actions.square().sum((-1, -2)) * self.weight


@pytest.fixture
def guard(tmp_path):
    path = tmp_path / "dependency"
    path.write_text("synthetic fixture")
    audit = freeze_audit(RGB, "lewm", [path], preprocessing="fixture", clock="five-step packets")
    model = ToyModel()
    return SingleFrameRGB(model, audit, expected_model_digest=model_digest(model),
                          action_mean=np.array([0., 0.]), action_scale=np.array([1., 1.]),
                          packet_steps=5, physical_low=-1, physical_high=1)


def inputs():
    return {"pixels": torch.zeros(1, 1, 1, 3, 8, 8),
            "goal": torch.zeros(1, 1, 1, 3, 8, 8),
            "action": torch.empty(1, 1, 0, 10)}


def test_single_frame_score_keeps_action_gradients(guard):
    actions = torch.full((1, 2, 3, 10), .25, requires_grad=True)
    value, context = guard.score(inputs(), actions)
    assert value.shape == (1, 2)
    value.sum().backward()
    torch.testing.assert_close(actions.grad, torch.ones_like(actions))
    assert "predicted_emb" in context
    guard.verify_unchanged()


def test_nonempty_history_rejected_not_silently_ignored(guard):
    data = inputs()
    data["action"] = torch.zeros(1, 1, 1, 10)
    with pytest.raises(ValueError, match="history"):
        guard.score(data, torch.zeros(1, 1, 2, 10))


@pytest.mark.parametrize("change", ["privileged", "multi_frame", "packet", "bounds", "nan"])
def test_invalid_input_rejected(guard, change):
    data, actions = inputs(), torch.zeros(1, 1, 2, 10)
    if change == "privileged":
        data["proprio"] = torch.zeros(1, 4)
    elif change == "multi_frame":
        data["pixels"] = torch.zeros(1, 1, 2, 3, 8, 8)
    elif change == "packet":
        actions = torch.zeros(1, 1, 2, 9)
    elif change == "bounds":
        actions[0, 0, 0, 0] = 1.01
    else:
        actions[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        guard.score(data, actions)


def test_model_mutation_detected(guard):
    with torch.no_grad():
        guard.model.weight.add_(1)
    with pytest.raises(ValueError, match="model"):
        guard.verify_unchanged()


def test_wrong_model_binding_rejected(tmp_path):
    path = tmp_path / "dependency"
    path.write_text("synthetic fixture")
    audit = freeze_audit(RGB, "lewm", [path], preprocessing="fixture", clock="fixture")
    with pytest.raises(ValueError, match="model"):
        SingleFrameRGB(ToyModel(), audit, expected_model_digest="wrong",
                       action_mean=[0, 0], action_scale=[1, 1], packet_steps=5,
                       physical_low=-1, physical_high=1)


def test_existing_lewm_actually_replaces_supplied_action_history(tmp_path):
    class Encoder(torch.nn.Module):
        def forward(self, pixels, interpolate_pos_encoding):
            return SimpleNamespace(last_hidden_state=pixels.mean((1, 2, 3))[:, None, None])

    class Predictor(torch.nn.Module):
        def forward(self, emb, act_emb):
            return emb + act_emb[..., :1]

    model = JEPA(Encoder(), Predictor(), torch.nn.Identity())
    model.register_parameter("device_anchor", torch.nn.Parameter(torch.zeros(()), requires_grad=False))
    model.eval().requires_grad_(False)
    path = tmp_path / "dependency"
    path.write_text("synthetic fixture")
    audit = freeze_audit(RGB, "lewm", [path], preprocessing="fixture", clock="fixture")
    wrapper = RGBLeWM(model, audit)
    candidates = torch.full((1, 1, 2, 10), .2)
    first = inputs()
    second = inputs()
    first["action"] = torch.zeros(1, 1, 1, 10)
    second["action"] = torch.full((1, 1, 1, 10), 100.)
    cost_a, out_a = wrapper.score(first, candidates)
    cost_b, out_b = wrapper.score(second, candidates)
    torch.testing.assert_close(out_a["action"], candidates[:, :, :1])
    torch.testing.assert_close(out_b["action"], candidates[:, :, :1])
    torch.testing.assert_close(cost_a, cost_b)
