import numpy as np

from experiments.pc_multi_direction_goal_combination_dev_20261105_codex import combine_directions
from experiments.pc_good_chain_response_edit_v2_dev_20261105_codex import candidates


def unordered(chains):
    return {tuple(x.reshape(-1).tolist()) for x in chains}


def test_two_independent_target_components_change_candidate_set():
    actual = np.zeros((2, 5, 2))
    actual[0, 0, 0] = 1
    actual[1, 0, 1] = 1
    feature = np.eye(2)
    first, _ = combine_directions(actual, feature, [1, 0])
    second, _ = combine_directions(actual, feature, [0, 1])
    parent = np.zeros((15, 2), dtype=np.float32)
    zero = np.zeros((5, 2))
    assert unordered(candidates(parent, [first, zero, zero])[0]) != unordered(candidates(parent, [second, zero, zero])[0])


def test_single_direction_opposite_target_only_reorders_symmetric_edits():
    actual = np.ones((1, 5, 2))
    feature = np.ones((1, 1))
    first, _ = combine_directions(actual, feature, [1])
    second, _ = combine_directions(actual, feature, [-1])
    parent = np.zeros((15, 2), dtype=np.float32)
    zero = np.zeros((5, 2))
    assert unordered(candidates(parent, [first, zero, zero])[0]) == unordered(candidates(parent, [second, zero, zero])[0])


def test_random_combination_is_independent_of_target():
    actual = np.arange(40).reshape(4, 5, 2)
    feature = np.arange(8).reshape(4, 2)
    coeffs = [1., -2., 3., -4.]
    a, _ = combine_directions(actual, feature, [1, 0], coeffs)
    b, _ = combine_directions(actual, feature, [0, 1], coeffs)
    np.testing.assert_array_equal(a, b)
