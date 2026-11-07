import numpy as np
import pytest
import torch

from experiments.pc_tworoom_cem16_variance_floor_20261107_codex import base, solve, update_elites


def test_floor_and_stable_elite_ties():
    samples = np.broadcast_to(np.arange(5)[:, None, None], (5, 5, 2)).copy()
    mean, std = update_elites(samples, [0, 0, 3, 4, 5], 0.7)
    np.testing.assert_allclose(mean, 0.5)
    np.testing.assert_allclose(std, 0.7)
    _, unmodified = update_elites(samples, [0, 0, 3, 4, 5], 1e-5)
    np.testing.assert_allclose(unmodified, 0.5)


def test_invalid_scores_and_floor_rejected():
    samples = np.zeros((5, 5, 2))
    with pytest.raises(ValueError):
        update_elites(samples, [0, 0, 0, 0, np.nan], 0.2)
    with pytest.raises(ValueError):
        update_elites(samples, [0] * 5, 0)


class IdentityScaler:
    mean_ = np.zeros(2)
    scale_ = np.ones(2)

    def transform(self, x):
        return np.asarray(x)

    def inverse_transform(self, x):
        return np.asarray(x)


class FakeBoundary:
    def score(self, current, action):
        del current
        batch, candidates, steps, _ = action.shape
        cumulative = action.sum(-1).cumsum(-1)
        pred = torch.cat((torch.zeros((batch, candidates, 1)), cumulative), dim=-1)[..., None]
        goal = torch.tensor([[[1.0]]])
        cost = (pred[:, :, -1] - goal[:, -1, None]).square().sum(-1)
        return cost, {"predicted_emb": pred, "goal_emb": goal}


@pytest.mark.parametrize("remaining", [5, 10, 15, 20, 25])
def test_original_floor_reproduces_reference_solver(remaining):
    current = {"pixels": torch.zeros(1)}
    old = np.zeros((remaining, 2), np.float32)
    reference = base.cem_temporal(FakeBoundary(), current, IdentityScaler(), old, remaining, 44, 16)
    actual = solve(FakeBoundary(), current, IdentityScaler(), old, remaining, 44, 1e-5)
    np.testing.assert_array_equal(actual["actions"], reference["actions"])
    assert actual["chosen"] == reference["chosen"]
    assert actual["chosen_cost"] == reference["chosen_cost"]
    for observed, recorded in zip(actual["rounds"], reference["rounds"], strict=True):
        assert recorded["count"] == 5
        for field in ("mean", "std", "costs"):
            np.testing.assert_array_equal(observed[field], recorded[field])
