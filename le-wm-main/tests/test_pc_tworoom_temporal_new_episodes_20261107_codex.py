import numpy as np
import pytest
import torch

from experiments.pc_tworoom_temporal_new_episodes_20261107_codex import cem_temporal


class IdentityScaler:
    mean_ = np.zeros(2)
    scale_ = np.ones(2)

    def transform(self, value):
        return np.asarray(value)

    inverse_transform = transform


class Boundary:
    def score(self, current, actions):
        # Initial state equals goal; including it would make every temporal cost zero.
        n, packets = actions.shape[1:3]
        predicted = torch.ones((1, n, packets + 1, 2))
        predicted[:, :, 0] = 0
        predicted[:, :, 1:, 0] = actions[..., 0] + 2
        goal = torch.zeros((1, 1, 2))
        endpoint = (predicted[:, :, -1] ** 2).sum(-1)
        return endpoint, {"predicted_emb": predicted, "goal_emb": goal}


@pytest.mark.parametrize("budget", [16, 300])
@pytest.mark.parametrize("remaining", [5, 10, 15, 20, 25])
def test_cem_budget_clock_and_temporal_excludes_initial(budget, remaining):
    current = {"pixels": torch.empty(1)}
    with torch.inference_mode():
        result = cem_temporal(Boundary(), current, IdentityScaler(), None, remaining, 42, budget)
    assert result["queries"] == budget
    assert result["packet_steps"] == budget * remaining // 5
    assert 1 + sum(row["count"] for row in result["rounds"]) == budget
    assert result["chosen_cost"] >= 1
    assert result["actions"].shape == (remaining, 2)
    assert np.abs(result["actions"]).max() <= 1


def test_cem_rejects_invalid_clock_and_budget():
    with pytest.raises(ValueError):
        cem_temporal(Boundary(), {}, IdentityScaler(), None, 7, 42, 16)
    with pytest.raises(ValueError):
        cem_temporal(Boundary(), {}, IdentityScaler(), None, 5, 42, 17)
