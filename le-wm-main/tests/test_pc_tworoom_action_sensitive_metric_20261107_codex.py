import numpy as np
import pytest

from experiments.pc_tworoom_action_sensitive_metric_20261107_codex import action_subspace


def test_subspace_ignores_prefixwise_constant_offsets():
    rng = np.random.default_rng(23)
    data = rng.normal(size=(8, 8, 16))
    basis, singular = action_subspace(data)
    offset = rng.normal(size=(8, 1, 16))
    shifted, singular_shifted = action_subspace(data + offset)
    assert basis.shape == (8, 16)
    np.testing.assert_allclose(basis.T @ basis, shifted.T @ shifted, atol=1e-10)
    np.testing.assert_allclose(singular, singular_shifted, atol=1e-10)


def test_rank_deficiency_abstains_and_invalid_shape_fails():
    basis, singular = action_subspace(np.zeros((8, 8, 16)))
    assert basis is None
    assert not singular.any()
    with pytest.raises(ValueError):
        action_subspace(np.zeros((7, 8, 16)))
