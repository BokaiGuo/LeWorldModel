import numpy as np
import pytest

from experiments.spectral_reachability import p1_exploratory_replay_quarantine_20261101_hypatia as mod


def test_actual_replay_and_exact_payload_fields():
    outputs = mod.evaluate()
    assert [x["slot"] for x in outputs] == [0, 1]
    for row in outputs:
        assert set(row) == {"slot", "rgb_history", "raw_action_history", "goal_rgb"}
        assert row["rgb_history"].shape == (3, 224, 224, 3)
        assert row["goal_rgb"].dtype == np.uint8
        assert row["raw_action_history"].shape == (10, 2)
        assert row["raw_action_history"].dtype == np.float32


def test_no_missing_slot_or_replacement():
    with pytest.raises(ValueError, match="fixed denominator"):
        mod.write_quarantine([])
