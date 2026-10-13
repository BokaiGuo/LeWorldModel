import numpy as np
import torch

from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import (
    _bootstrap_mean,
    _bootstrap_clustered,
    _make_action_blocks,
    _window_starts,
)


def test_window_starts_are_deterministic_and_in_range():
    starts = _window_starts(104, context_length=3, max_horizon=8, windows_per_episode=2)
    assert starts == [0, 10]
    assert all(start >= 0 for start in starts)


def test_public_action_blocks_flatten_five_steps():
    raw = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    blocks = _make_action_blocks(raw, start_obs=1, count=2, mode="observed")
    assert blocks.shape == (2, 10)
    assert torch.allclose(blocks[0, :2], (raw[5] / 100.0 - torch.tensor([-0.0087, 0.0068])) / torch.tensor([0.2019, 0.2002]))
    assert torch.allclose(blocks[1], _make_action_blocks(raw, 1, 2, "zero")[1] * 0.0)


def test_bootstrap_mean_is_finite():
    report = _bootstrap_mean(np.arange(10, dtype=np.float64), repeats=100, seed=4)
    assert report["n"] == 10
    assert np.isfinite(report["mean"])
    assert len(report["ci95"]) == 2


def test_cluster_bootstrap_uses_episode_means():
    report = _bootstrap_clustered(
        np.asarray([0.0, 2.0, 10.0, 14.0]),
        np.asarray([0, 0, 1, 1]),
        repeats=100,
        seed=5,
    )
    assert report["bootstrap_unit"] == "episode"
    assert report["clusters"] == 2
    assert report["windows"] == 4
    assert report["mean"] == 6.5
