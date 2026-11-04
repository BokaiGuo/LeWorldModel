"""Regression checks over the saved host replay; no image-to-position inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/pc_pointmaze_rgb_forward_clock_differences_20261103_CODEX.json"
ARRAYS = REPORT.with_suffix(".npz")
PREVIOUS = ROOT / "results/pc_pointmaze_cp12_native_position_stepwise_20261103_CODEX.npz"


def test_saved_replay_and_geometry_contract():
    report = json.loads(REPORT.read_text())
    assert hashlib.sha256(ARRAYS.read_bytes()).hexdigest() == report["arrays_sha256"]
    with np.load(ARRAYS, allow_pickle=False) as d:
        s, g, b, off = (d[k] for k in ("states", "geoms", "boundary_pre_forward", "geometry_offsets"))
        assert s.shape == (2, 64, 86, 5)
        np.testing.assert_array_equal(d["post_forward_state"], s[:, :, ::5, :])
        delta = s[1] - s[0]
        assert np.count_nonzero(np.any(delta != 0, axis=2)) == report["different_trajectory_tick_pairs"]
        np.testing.assert_array_equal(np.max(np.abs(delta), axis=(0, 1)), report["max_abs_delta_qpos_qvel_time"])
        ticks = np.arange(1, 18) * 5
        previous = s[0][:, ticks - 1, :2] + off[:, None, :]
        current = s[1][:, ticks, :2] + off[:, None, :]
        assert np.linalg.norm(b[0, :, 1:, :2] - previous, axis=2).max() < 1e-12
        assert np.linalg.norm(g[1][:, ticks, :2] - current, axis=2).max() < 1e-12
        assert np.count_nonzero(d["pixel_differences"][:, :, 0]) == report["different_rgb_frames"]
        with np.load(PREVIOUS, allow_pickle=False) as prior:
            np.testing.assert_array_equal(s[0, :, ::5, :4], prior["native_states"])
