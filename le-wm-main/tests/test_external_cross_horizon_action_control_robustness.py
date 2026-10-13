import json
from pathlib import Path

import numpy as np
import torch

from experiments.empirical_spectral_audit.run_external_cross_horizon_action_control_robustness import (
    _block_mean_history,
    _normalized_time_dct16,
)


def test_block_mean_history_has_common_padded_dimension_across_horizons():
    actions = torch.arange(160, dtype=torch.float32).reshape(1, 80, 2)
    mean = torch.zeros(2)
    std = torch.ones(2)
    short = _block_mean_history(actions, 0, 0, 1, "raw_div100", mean, std)
    long = _block_mean_history(actions, 0, 0, 8, "raw_div100", mean, std)
    assert short.shape == (16,)
    assert long.shape == (16,)
    assert np.isfinite(short).all()
    assert np.isfinite(long).all()
    np.testing.assert_allclose(short[2:], 0.0)
    assert not np.allclose(long[2:], 0.0)


def test_normalized_time_dct_uses_a_common_dimension_and_is_finite():
    actions = torch.arange(160, dtype=torch.float32).reshape(1, 80, 2)
    mean = torch.zeros(2)
    std = torch.ones(2)
    for horizon in (1, 4, 8):
        features = _normalized_time_dct16(
            actions, 0, 0, horizon, "raw_div100", mean, std
        )
        assert features.shape == (16,)
        assert np.isfinite(features).all()


def test_canonical_action_control_artifacts_are_complete_and_cuda_backed():
    root = Path(__file__).resolve().parents[1] / "results/pusht/lewm/external_cross_horizon_action_control_robustness_20261004"
    paths = sorted(root.glob("canonical_*.json"))
    assert len(paths) == 6
    for path in paths:
        report = json.loads(path.read_text())
        assert report["status"] == "complete"
        assert report["model"]["device"] == "cuda"
        assert report["source"]["episodes"] == 21
        assert report["protocol"]["bootstrap_repeats"] == 2000
        assert report["crossfit"]["condition_summary"]["compact6/h8"]["feature_dim"] == 6
        assert report["crossfit"]["condition_summary"]["block_mean_history_padded/h8"]["feature_dim"] == 16
        assert report["crossfit"]["condition_summary"]["normalized_time_dct16/h8"]["feature_dim"] == 16
        for control in ("compact6", "block_mean_history_padded"):
            key = f"{control}/h1_to_h4/dynamics_band_increment/dynamics_basis/k8"
            assert len(report["crossfit"]["summary"][key]["state_incremental_r2_over_action"]["ci95"]) == 2
        dct_key = "normalized_time_dct16/h1_to_h4/dynamics_band_increment/k8/dynamics_minus_random"
        assert len(report["crossfit"]["paired_contrasts"][dct_key]["ci95"]) == 2
