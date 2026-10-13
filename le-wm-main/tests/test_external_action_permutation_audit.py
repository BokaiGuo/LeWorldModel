import numpy as np

from experiments.empirical_spectral_audit.run_external_action_permutation_audit import (
    _random_derangement,
)


def test_random_derangement_has_no_fixed_points_and_is_reproducible():
    first = _random_derangement(42, seed=20261003)
    second = _random_derangement(42, seed=20261003)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(42))
    assert not np.any(first == np.arange(42))
