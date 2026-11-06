import numpy as np

from experiments.audit_pc_action_opportunity_identity_20261104_codex import features, targets


def test_sorted_and_shuffled_preserve_row_values_not_identity():
    vectors = np.array([[2.0, 1.0, 3.0], [5.0, 9.0, 7.0]])
    controls = np.array([[10.0], [11.0]])
    for mode in ("sorted", "shuffled"):
        result = features(vectors, controls, mode, 17)
        np.testing.assert_array_equal(np.sort(result[:, :3], axis=1), np.sort(vectors, axis=1))
        np.testing.assert_array_equal(result[:, 3], controls[:, 0])
    assert not np.array_equal(features(vectors, controls, "sorted", 17)[:, :3], vectors)


def test_absolute_target_uses_candidate_specific_baseline():
    result = targets(np.array([[2.0, 1.0], [3.0, 0.0]]), np.array([5.0, 9.0]))
    np.testing.assert_array_equal(result["relative_improvement"], [2.0, 3.0])
    np.testing.assert_array_equal(result["absolute_cost"], [3.0, 6.0])
