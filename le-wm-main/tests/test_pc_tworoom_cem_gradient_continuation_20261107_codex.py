import numpy as np
import pytest
import torch

from experiments.pc_tworoom_cem_gradient_continuation_20261107_codex import (
    cem_candidates, gradient_step, select_scored,
)


def test_cem_reconstructs_last_elite_and_clips():
    actions = np.arange(300 * 2, dtype=float).reshape(300, 1, 2) / 1000
    costs = np.arange(300, dtype=float)
    sampled = cem_candidates(actions, costs, np.array([[-.2, -.2]]),
                             np.array([[.2, .2]]), 2)
    assert sampled.shape == (20, 1, 2)
    assert np.all(sampled >= -.2) and np.all(sampled <= .2)
    np.testing.assert_array_equal(sampled, cem_candidates(actions, costs, np.array([[-.2, -.2]]),
                                                          np.array([[.2, .2]]), 2))


def test_gradient_step_uses_max_norm_and_physical_bounds():
    x = torch.tensor([.9, 0.])
    out = gradient_step(x, torch.tensor([-2., 1.]), torch.tensor([-1., -1.]),
                        torch.tensor([.91, 1.]))
    torch.testing.assert_close(out, torch.tensor([.91, -.025]))
    assert not out.requires_grad


def test_gradient_zero_and_nonfinite():
    x = torch.tensor([.2])
    torch.testing.assert_close(gradient_step(x, torch.zeros(1), -torch.ones(1), torch.ones(1)), x)
    with pytest.raises(ValueError):
        gradient_step(x, torch.tensor([float("nan")]), -torch.ones(1), torch.ones(1))


def test_selection_retains_earliest_incumbent_on_tie():
    actions = np.array([[1.], [2.], [3.]])
    selected, cost, index = select_scored(actions, [1., 1., 3.])
    assert index == 0 and cost == 1.
    np.testing.assert_array_equal(selected, [1.])
