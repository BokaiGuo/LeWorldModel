import numpy as np
import pytest

from experiments.spectral_reachability.pc_atlas_turn_amplitude_grid_20261103 import run_goal


def test_grid_complete_and_counts():
    row = run_goal(0)
    assert len(row["candidates"]) == 10
    assert row["counts"]["rollout_returned"] == 64
    assert row["counts"]["jacobian_returned"] == 11
    assert {r["velocity_kind"] for r in row["candidates"]} == {"stationary", "progress_velocity"}
    for item in row["candidates"]:
        assert abs(item["actions"][0][1]) <= np.pi / 2
        assert item["witness_sha256"]


def test_invalid_index():
    with pytest.raises(ValueError):
        run_goal(8)
