import numpy as np
import pytest
import torch

from experiments.pc_tworoom_rgb_stop_feedback_20261107_codex import select_tail_rgb


class Scaler:
    def transform(self, x):
        return x


class Boundary:
    def score(self, context, actions):
        assert set(context) == {"pixels", "goal", "action"}
        assert actions.shape == (1, 2, 3, 10)
        return torch.tensor([[3.0, 1.0]]), {}


def test_decision_interface_has_only_rgb_goal_and_legal_actions():
    image = np.zeros((4, 4, 3), np.uint8)
    tails = np.zeros((2, 15, 2), np.float32)
    transform = lambda x: torch.from_numpy(x.copy())
    selected, scores = select_tail_rgb(image, image, tails, transform=transform,
                                       scaler=Scaler(), boundary=Boundary(), device="cpu")
    assert selected == 1 and scores == [3.0, 1.0]
    with pytest.raises(TypeError):
        select_tail_rgb(image, image, tails, transform=transform, scaler=Scaler(),
                        boundary=Boundary(), device="cpu", state=np.zeros(2))
    tails[0, 0, 0] = 1.1
    with pytest.raises(ValueError):
        select_tail_rgb(image, image, tails, transform=transform,
                        scaler=Scaler(), boundary=Boundary(), device="cpu")
