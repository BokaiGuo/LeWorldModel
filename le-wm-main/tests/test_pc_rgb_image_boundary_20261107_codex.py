import pytest
import torch

from experiments.pc_input_permissions import RGB, freeze_audit
from experiments.pc_rgb_image_boundary_20261107_codex import RGBImageLeWM


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_cost(self, context, actions):
        self.calls += 1
        return actions.sum(dim=(-1, -2))

    def encode(self, inputs):
        self.calls += 1
        return inputs


@pytest.fixture
def bound(tmp_path):
    path = tmp_path / "weights"
    path.write_text("fixture")
    audit = freeze_audit(RGB, "lewm", [path], preprocessing="fixture", clock="fixture")
    model = Model().eval()
    return RGBImageLeWM(model, audit, image_size=8), model


def valid():
    context = {"pixels": torch.zeros(1, 1, 1, 3, 8, 8),
               "goal": torch.zeros(1, 1, 1, 3, 8, 8),
               "action": torch.empty(1, 1, 0, 10)}
    return context, torch.zeros(1, 2, 3, 10)


def test_accepts_rgb_and_records_exact_existing_inputs(bound):
    boundary, model = bound
    context, actions = valid()
    cost, _ = boundary.score(context, actions)
    assert tuple(cost.shape) == (1, 2)
    assert model.calls == 1
    assert set(boundary.receipts[0]["inputs"]) == {"pixels", "goal", "action", "candidate_actions"}
    boundary.encode({"pixels": torch.zeros(1, 1, 3, 8, 8)})
    assert model.calls == 2


@pytest.mark.parametrize("kind", ["state_vector", "hidden_state", "gray", "wrong_batch", "bad_packet",
                                  "future_actions", "nonfloat", "expanded_images", "nan"])
def test_rejects_invalid_inputs_before_model(bound, kind):
    boundary, model = bound
    context, actions = valid()
    if kind == "state_vector":
        context["pixels"] = torch.zeros(1, 1, 4)
    elif kind == "hidden_state":
        context["state"] = torch.zeros(1, 4)
    elif kind == "gray":
        context["pixels"] = torch.zeros(1, 1, 1, 1, 8, 8)
    elif kind == "wrong_batch":
        context["goal"] = torch.zeros(2, 1, 1, 3, 8, 8)
    elif kind == "bad_packet":
        actions = torch.zeros(1, 2, 3, 4)
    elif kind == "future_actions":
        context["action"] = torch.zeros(1, 1, 5, 10)
    elif kind == "nonfloat":
        context["pixels"] = context["pixels"].to(torch.uint8)
    elif kind == "expanded_images":
        context["pixels"] = torch.zeros(1, 2, 1, 3, 8, 8)
    else:
        actions[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        boundary.score(context, actions)
    assert model.calls == 0
    assert not boundary.receipts
