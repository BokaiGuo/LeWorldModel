import math

import numpy as np

from experiments.empirical_spectral_audit.run_external_cross_horizon_readout import (
    _state_features,
)


def test_cross_horizon_state_features_are_finite_and_use_current_latent_only():
    mean = np.zeros(4, dtype=np.float64)
    basis = np.eye(4, dtype=np.float64)
    rows = [
        {"true": np.asarray([[1.0, 2.0, 3.0, 4.0], [9.0, 9.0, 9.0, 9.0]])},
        {"true": np.asarray([[-1.0, 0.0, 1.0, 2.0], [8.0, 8.0, 8.0, 8.0]])},
    ]
    features = _state_features(rows, mean, basis, rank=2)
    assert features.shape == (2, 2)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features, [[1.0, 2.0], [-1.0, 0.0]])
    assert all(math.isfinite(float(value)) for value in features.ravel())
