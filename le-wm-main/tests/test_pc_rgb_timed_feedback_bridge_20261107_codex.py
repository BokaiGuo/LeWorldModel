import pytest
import torch

from experiments.pc_input_permissions import RGB, freeze_audit
from experiments.pc_rgb_history_contract_20261107_codex import score_future
from experiments.pc_rgb_single_frame_contract_20261107_codex import model_digest
from experiments.pc_rgb_timed_history_contract_20261107_codex import TimedHistoryRGB
from experiments.pc_rgb_timed_feedback_bridge_20261107_codex import TimedFeedbackBridge


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.eval()

    def get_cost(self, context, actions):
        context["predicted_emb"] = actions
        return actions.square().sum((-1, -2)) * self.weight


@pytest.fixture
def bridge(tmp_path):
    dependency = tmp_path / "source"
    dependency.write_text("fixture")
    audit = freeze_audit(RGB, "lewm", [dependency], preprocessing="fixture", clock="fixture")
    model = Toy()
    timed = TimedHistoryRGB(model, audit, expected_model_digest=model_digest(model),
                            action_mean=[0., 0.], action_scale=[1., 1.], packet_steps=5,
                            physical_low=-1., physical_high=1., max_frames=3)
    return TimedFeedbackBridge(timed)


def sample():
    return dict(pixels=torch.zeros(1, 1, 2, 3, 4, 4),
                goal=torch.zeros(1, 1, 1, 3, 4, 4),
                history=torch.full((1, 1, 1, 10), .1),
                future=torch.full((1, 2, 2, 10), .25, requires_grad=True))


def test_bridge_preserves_history_and_future_gradient(bridge):
    data = sample()
    bridge.bind(frame_steps=(0, 5), current_step=5, history=data["history"])
    values, ctx = score_future(bridge, **data)
    torch.testing.assert_close(ctx["predicted_emb"][:, :, :1], data["history"].expand(-1, 2, -1, -1))
    values.sum().backward()
    torch.testing.assert_close(data["future"].grad, torch.ones_like(data["future"]))


def test_missing_binding_rejected(bridge):
    with pytest.raises(ValueError, match="bind"):
        score_future(bridge, **sample())


def test_changed_declared_history_rejected(bridge):
    data = sample()
    bridge.bind(frame_steps=(0, 5), current_step=5, history=data["history"])
    data["history"] = data["history"] + .1
    with pytest.raises(ValueError, match="history"):
        score_future(bridge, **data)


def test_mismatched_candidate_prefix_rejected(bridge):
    data = sample()
    bridge.bind(frame_steps=(0, 5), current_step=5, history=data["history"])
    full = torch.cat((data["history"].expand(-1, 2, -1, -1), data["future"]), 2)
    full = full.clone()
    full[0, 1, 0, 0] = .4
    with pytest.raises(ValueError, match="history"):
        bridge.score({"pixels": data["pixels"], "goal": data["goal"], "action": data["history"]}, full)


def test_bad_clock_and_empty_future_rejected(bridge):
    data = sample()
    bridge.bind(frame_steps=(0, 6), current_step=6, history=data["history"])
    with pytest.raises(ValueError, match="clock"):
        score_future(bridge, **data)
    bridge.bind(frame_steps=(0, 5), current_step=5, history=data["history"])
    with pytest.raises(ValueError, match="future"):
        bridge.score({"pixels": data["pixels"], "goal": data["goal"], "action": data["history"]},
                     data["history"])
