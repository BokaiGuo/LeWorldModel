import numpy as np
import pytest

from experiments.pc_rgb_temporal_final_search_20261107_codex import choose_with_temporal


def test_temporal_selects_bound_tail_without_changing_retained_prefix():
    parents = np.zeros((3, 25, 2), np.float32)
    parents[2, :10] = 0.5
    probes = np.zeros((2, 15, 2), np.float32)
    tails = np.ones((2, 15, 2), np.float32) * 0.2
    result = {"retained_prefixes": [2, 0]}
    scores = [np.ones(3), np.ones(9), np.array([1., 0., 1., 1.])]
    choice, chain = choose_with_temporal(result, scores, parents, probes, tails, 10, 25)
    assert (choice.source, choice.index) == ("tail", 1)
    np.testing.assert_array_equal(chain[:10], parents[2, :10])
    np.testing.assert_array_equal(chain[10:], tails[1])
    assert result["retained_prefixes"] == [2, 0]


def test_invalid_temporal_score_shape_rejected():
    with pytest.raises(ValueError):
        choose_with_temporal({}, [np.zeros(3)], None, None, None, 10, 25)
