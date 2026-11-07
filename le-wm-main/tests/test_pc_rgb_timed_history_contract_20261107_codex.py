import numpy as np
import pytest
import torch

from experiments.pc_input_permissions import RGB, freeze_audit
from experiments.pc_rgb_single_frame_contract_20261107_codex import model_digest
from experiments.pc_rgb_timed_history_contract_20261107_codex import TimedHistoryRGB


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.eval()

    def get_cost(self, context, actions):
        context["predicted_emb"] = actions
        return actions.square().sum((-1, -2)) * self.weight


@pytest.fixture
def wrapper(tmp_path):
    dependency = tmp_path / "source"
    dependency.write_text("fixture")
    audit = freeze_audit(RGB, "lewm", [dependency], preprocessing="fixture", clock="fixture")
    model = Toy()
    return TimedHistoryRGB(model, audit, expected_model_digest=model_digest(model),
                           action_mean=[0., 0.], action_scale=[1., 1.], packet_steps=5,
                           physical_low=-1., physical_high=1., max_frames=3)


def inputs():
    return {"pixels": torch.zeros(1, 1, 3, 3, 4, 4), "goal": torch.zeros(1, 1, 1, 3, 4, 4),
            "history": torch.zeros(1, 1, 2, 10),
            "future": torch.full((1, 2, 1, 10), .25, requires_grad=True),
            "frame_steps": (0, 5, 10), "current_step": 10}


def test_shared_history_and_future_gradient(wrapper):
    args = inputs()
    values, context = wrapper.score_timed(**args)
    assert values.shape == (1, 2)
    assert context["predicted_emb"].shape == (1, 2, 3, 10)
    values.sum().backward()
    torch.testing.assert_close(args["future"].grad, torch.ones_like(args["future"]))
    assert wrapper.verify_unchanged()


@pytest.mark.parametrize("steps,current", [((0, 5, 11), 11), ((0, 5, 10), 15),
                                          ((0, 5), 10), ((0, 5, 10.0), 10)])
def test_wrong_or_ambiguous_frame_clock_rejected(wrapper, steps, current):
    args = inputs()
    args["frame_steps"], args["current_step"] = steps, current
    with pytest.raises(ValueError, match="clock"):
        wrapper.score_timed(**args)


@pytest.mark.parametrize("field", ["history", "future"])
def test_out_of_bounds_physical_packets_rejected(wrapper, field):
    args = inputs()
    actions = args[field].detach().clone()
    actions.reshape(-1)[0] = 1.01
    args[field] = actions
    with pytest.raises(ValueError, match="bounds"):
        wrapper.score_timed(**args)


def test_single_frame_and_explicit_empty_history(wrapper):
    args = inputs()
    args.update(pixels=args["pixels"][:, :, -1:], history=torch.empty(1, 1, 0, 10),
                frame_steps=(10,), current_step=10)
    values, _ = wrapper.score_timed(**args)
    assert values.shape == (1, 2)


def test_rejects_future_frame_count_and_nonfinite_goal(wrapper):
    args = inputs()
    args["goal"] = torch.full_like(args["goal"], float("nan"))
    with pytest.raises(ValueError):
        wrapper.score_timed(**args)
