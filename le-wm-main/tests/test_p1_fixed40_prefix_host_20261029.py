import numpy as np
import pytest

from experiments.spectral_reachability.p1_fixed40_prefix_host_20261029 import score


def test_score_identity_mse():
    assert score(np.ones(192), np.zeros(192)) == 1.0


def test_score_rejects_nonfinite_and_wrong_shape():
    with pytest.raises(ValueError):
        score(np.ones(191), np.zeros(192))
    with pytest.raises(ValueError):
        score(np.full(192, np.inf), np.zeros(192))
