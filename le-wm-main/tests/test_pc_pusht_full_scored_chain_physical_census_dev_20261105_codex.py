import numpy as np
import pytest

from experiments.pc_pusht_full_scored_chain_physical_census_dev_20261105_codex import extract


def test_extraction_retains_first_origin_and_duplicate_score_span():
    first = np.zeros((15, 2), np.float32)
    second = np.ones((15, 2), np.float32)
    d = {"evaluations": [
        {"actions": [first.tolist(), second.tolist()], "scores": [3., 2.]},
        {"actions": [first.tolist()], "scores": [1.]},
        {"actions": [np.zeros((5, 2)).tolist()], "scores": [0.]},
    ]}
    actions, low, high, count, origin, total, short = extract(d)
    np.testing.assert_array_equal(actions, np.stack((first, second)))
    np.testing.assert_array_equal(low, [1., 2.])
    np.testing.assert_array_equal(high, [3., 2.])
    np.testing.assert_array_equal(count, [2, 1])
    np.testing.assert_array_equal(origin, [[0, 0], [0, 1]])
    assert (total, short) == (3, 1)


def test_extraction_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="invalid trace"):
        extract({"evaluations": [{"actions": [np.zeros((15, 2)).tolist()], "scores": [float("nan")]}]})
