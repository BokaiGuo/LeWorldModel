import numpy as np

from experiments.pc_action_opportunity_stats import average_rank, partial_rank_correlation, spearman


def test_average_rank_assigns_mean_rank_to_ties():
    np.testing.assert_allclose(average_rank(np.array([2.0, 1.0, 1.0, 3.0])), [3.0, 1.5, 1.5, 4.0])


def test_spearman_is_invariant_to_joint_row_order():
    left = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
    right_a = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    permutation = [1, 0, 2, 4, 3]
    right_b = right_a[permutation]
    left_b = left[permutation]
    assert spearman(left, right_a) == spearman(left_b, right_b)


def test_partial_rank_correlation_accepts_explicit_controls():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = x * 2.0 + np.array([0.0, 1.0, -1.0, 0.5, -0.5, 0.0])
    controls = np.column_stack((x, np.ones_like(x)))
    value = partial_rank_correlation(x, y, controls)
    assert value is not None
    assert np.isfinite(value)
