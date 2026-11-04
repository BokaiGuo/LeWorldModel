import pytest
import torch

from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history


class Inner:
    def __init__(self):
        self.calls = []

    def encode_act(self, action):
        return action

    def forward_pred(self, visual, action, proprio):
        self.calls.append((visual.clone(), action.clone(), proprio.clone()))
        return visual + action[..., :1, None, None, None], None, proprio + action[..., :1, None]


class Model:
    proprio_mode = "predict_proprio"

    def __init__(self):
        self.model = Inner()


def test_multistep_history_sliding_and_gradient():
    model = Model()
    context = {"visual": torch.zeros(1, 2, 1, 1, 1, 1),
               "proprio": torch.zeros(1, 2, 1, 1)}
    history = torch.tensor([[[2.]]])
    future = torch.tensor([[[3.]], [[5.]], [[7.]]], requires_grad=True)
    output = unroll_with_action_history(model, context, history, future, window=2)
    assert output["visual"].shape == (5, 1, 1, 1, 1, 1)
    assert output["proprio"].shape == (5, 1, 1, 1)
    assert [call[1].flatten().tolist() for call in model.model.calls] == [
        [2., 3.], [3., 5.], [5., 7.]]
    assert output["visual"][-1].item() == 15.
    output["visual"][-1].sum().backward()
    torch.testing.assert_close(future.grad, torch.ones_like(future))


def test_invalid_history_and_mode_fail_closed():
    context = {"visual": torch.zeros(1, 2, 1, 1, 1, 1),
               "proprio": torch.zeros(1, 2, 1, 1)}
    model = Model()
    future = torch.ones(1, 1, 1)
    with pytest.raises(ValueError, match="history must contain"):
        unroll_with_action_history(model, context, torch.zeros(0, 1, 1), future, 2)
    with pytest.raises(ValueError, match="nonfinite"):
        unroll_with_action_history(model, context, torch.full((1, 1, 1), float("nan")), future, 2)
    model.proprio_mode = "compute_new_pose"
    with pytest.raises(ValueError, match="predictor-generated"):
        unroll_with_action_history(model, context, torch.zeros(1, 1, 1), future, 2)
