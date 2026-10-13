import numpy as np
import torch

from experiments.empirical_spectral_audit.run_external_forward_dynamics_readout import (
    BANDS,
    FEATURES,
    _action_features,
    _fit_probe,
)


def test_action_features_distinguish_future_and_past_blocks():
    raw = torch.arange(80, dtype=torch.float32).reshape(40, 2)
    mean = torch.zeros(2)
    std = torch.ones(2)
    future = _action_features(raw[None], 0, 0, 1, 0, "raw_div100", mean, std)
    past = _action_features(raw[None], 0, 0, 1, -1, "raw_div100", mean, std)
    assert future.shape == (6,)
    assert not np.allclose(future, past)


def test_fit_probe_recovers_linear_forward_readout():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(40, 3))
    weights = rng.normal(size=(3, 5))
    target = source @ weights + 0.1
    fit = _fit_probe(source, target, ridge=1e-8)
    prediction = (source - fit["source_mean"]) @ fit["weights"] + fit["target_mean"]
    assert np.mean((prediction - target) ** 2) < 1e-10


def test_protocol_features_and_bands_are_explicit():
    assert "state_plus_action" in FEATURES
    assert BANDS["top8"] == (0, 8)
    assert BANDS["middle8_32"] == (8, 32)
