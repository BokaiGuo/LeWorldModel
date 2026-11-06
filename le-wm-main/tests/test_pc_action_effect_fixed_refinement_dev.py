import numpy as np

from experiments.pc_action_effect_fixed_refinement_dev_20261105_codex import starts_from_result


def test_fixed_starts_keep_shared_global_and_both_retained_slots():
    zero = np.zeros((15, 2), np.float32)
    one = np.ones((15, 2), np.float32)
    other = np.full((15, 2), 0.5, np.float32)
    result = {
        "global_proposals": {"actions": [zero.tolist(), zero.tolist(), one.tolist()], "scores": [0.1, 0.2, 0.3]},
        "child": {"layers": [{"depth": 2, "retained": [1, 0],
                              "candidates": [{"chain": zero.tolist()}, {"chain": other.tolist()}]}]},
    }
    starts, indices = starts_from_result(result)
    assert indices == [0, 2]
    np.testing.assert_array_equal(starts[0], zero)
    np.testing.assert_array_equal(starts[1], one)
    np.testing.assert_array_equal(starts[2], other)
    np.testing.assert_array_equal(starts[3], zero)
