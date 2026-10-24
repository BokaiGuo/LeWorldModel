import numpy as np
import torch

from experiments.spectral_reachability.run_sra_p2_action_order_cuda import average_rank, goal_projected_effect, spearman


def test_average_rank_assigns_equal_values_equal_ranks():
    values = np.asarray([3.0, 1.0, 1.0, 2.0, 3.0])
    ranks = average_rank(values)
    assert ranks[1] == ranks[2]
    assert ranks[0] == ranks[4]
    assert ranks[1] < ranks[3] < ranks[0]


def test_spearman_with_ties_uses_average_rank_not_input_order():
    x = np.asarray([1.0, 1.0, 2.0, 2.0, 3.0])
    y = np.asarray([5.0, 5.0, 7.0, 7.0, 9.0])
    assert np.isclose(spearman(x, y), 1.0)


def test_goal_projected_effect_uses_the_effect_component_toward_the_goal():
    assert np.isclose(goal_projected_effect(np.asarray([2.0, 3.0]), np.asarray([0.0, 3.0]), np.asarray([3.0, 3.0])), 2.0)
    assert np.isclose(goal_projected_effect(torch.tensor([2.0, 3.0]), torch.tensor([0.0, 3.0]), torch.tensor([3.0, 3.0])), 2.0)
