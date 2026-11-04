import numpy as np
import torch

from experiments.pc_causal_joint_prefix_gradient_20261103 import decide, optimize


def test_joint_optimizer_respects_fixed_controls_and_bounds():
    initial = torch.zeros(10, 15, 2)
    initial[1, :5] = .4
    initial[2:, :5] = -.2

    def score(action):
        return action[:, :5].mean(dim=(1, 2)) + 1, (action[:, 5:] - .3).square().mean(dim=(1, 2))

    plans, first, terminal, gradient = optimize(initial, score, 1., steps=3)
    assert plans.shape == (4, 10, 15, 2)
    assert gradient.shape == (3, 10, 15, 2)
    np.testing.assert_array_equal(plans[:, 0, :5], np.zeros((4, 5, 2)))
    np.testing.assert_allclose(plans[:, 1, :5], .4)
    assert np.max(np.abs(plans)) <= 1
    assert np.all(terminal[-1] < terminal[0])
    assert first.shape == terminal.shape


def test_decide_strictly_requires_nonprogress_and_both_control_wins():
    first = np.ones((2, 10))
    first[:, 2:] = 1.1
    terminal = np.full((2, 10), 2.)
    terminal[:, 0], terminal[:, 1] = 1., .8
    terminal[1, 2] = .7
    controls, decisions = decide(first, terminal, 1.)
    assert controls == [0, 0]
    assert decisions[0]["qualified"] == [2, 1]
    assert decisions[1]["qualified"] is None
    first[:, 2] = .9
    assert decide(first, terminal, 1.)[1][0]["qualified"] is None
