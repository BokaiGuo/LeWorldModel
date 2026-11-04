import numpy as np
import pytest

from experiments.spectral_reachability.pc_atlas_depth3_matched_root_source_20261103 import (
    ARMS, RootSourceAtlas, run_arm,
)


def test_matched_bank_counts_and_witnesses():
    rows = [run_arm(3, 42, arm) for arm in ARMS]
    assert all(row["counts"] == rows[0]["counts"] for row in rows)
    assert all(row["backend_action_steps"] == rows[0]["backend_action_steps"] for row in rows)
    assert all(row["bank_count"] == 13 for row in rows)
    assert all(row["root_bank"][:2] == rows[0]["root_bank"][:2] for row in rows)
    assert rows[0]["root_bank"][2:] != rows[1]["root_bank"][2:]
    for row in rows:
        assert len(row["programs"]) == 4
        assert all(p["witness_replayed"] for p in row["programs"])
        assert np.isfinite(row["terminal_error"])


def test_invalid_arm_and_panel_rejected():
    with pytest.raises((TypeError, ValueError)):
        RootSourceAtlas(None, None, "unknown")
    with pytest.raises(ValueError):
        run_arm(64, 42, "random")
