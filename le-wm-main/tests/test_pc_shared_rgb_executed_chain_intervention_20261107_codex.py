import numpy as np
import pytest

from experiments.pc_shared_rgb_executed_chain_intervention_20261107_codex import candidate_prefixes


def test_candidates_keep_exact_original_zero_and_seed():
    original = np.arange(10, dtype=np.float32).reshape(5, 2) / 10
    x = candidate_prefixes(original, 7)
    assert x.shape == (32, 5, 2)
    np.testing.assert_array_equal(x[0], original)
    np.testing.assert_array_equal(x[1], 0)
    np.testing.assert_array_equal(x, candidate_prefixes(original, 7))
    assert np.all(x[2:] == x[2:, :1])


def test_invalid_prefix_rejected():
    with pytest.raises(ValueError):
        candidate_prefixes(np.ones((5, 2)) * 2, 1)
