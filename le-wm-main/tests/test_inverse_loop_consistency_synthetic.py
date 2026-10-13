import numpy as np
import torch

from experiments.empirical_spectral_audit.run_inverse_loop_consistency_synthetic import (
    _loop_metrics,
)


class AdditiveToy:
    def encode(self, state):
        return state

    def predict(self, latent, action):
        return latent + action


def test_exact_additive_model_has_zero_inverse_and_loop_error():
    metrics = _loop_metrics(AdditiveToy(), seed=4, samples=32)
    assert metrics["inverse_pair_error"] < 1e-6
    assert metrics["closed_loop_h4_error"] < 1e-6
    assert metrics["closed_loop_h6_error"] < 1e-6
    assert all(np.isfinite(value) for value in metrics.values())
