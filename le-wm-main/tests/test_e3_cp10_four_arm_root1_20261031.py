from copy import deepcopy

import pytest

from experiments.spectral_reachability import e3_cp10_four_arm_root1_20261031 as experiment


def test_fixed_goal_and_historical_early_success():
    entry, row, _ = experiment.inputs()
    assert row["root"] == 1
    assert row["pixel_sha256"] == entry["pixel_sha256"]
    assert len(entry["witness10_float32_hex"]) == 160
    assert entry["historical_first_success_step"] == 4


def test_rebind_refuses_wrong_physical_root():
    data = {"references": [None, {"states": [[0.0] * 7]}]}
    with pytest.raises(ValueError, match="physical root identity"):
        experiment.rebind({}, deepcopy(data))
