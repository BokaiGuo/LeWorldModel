import pytest
import torch

from experiments.spectral_reachability.run_reacher_pc_feasible_goal_cometric import score


class Identity:
    device = torch.device("cpu")
    def apply(self, x):
        return x


def test_goal_score_identity_and_shape():
    assert score(Identity(), [1, 2], [0, 0]) == 2.5
    with pytest.raises(ValueError):
        score(Identity(), [1, 2], [0])
