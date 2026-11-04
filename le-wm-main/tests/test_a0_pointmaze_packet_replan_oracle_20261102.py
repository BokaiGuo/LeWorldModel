"""Fixed packet ordering and saved cost tampering checks."""
import json

import numpy as np
import pytest

from experiments import a0_pointmaze_packet_replan_oracle_20261102 as exp
from experiments import a0_pointmaze_packet_replan_oracle_saved_20261102 as audit


def test_all_pairs_have_ordered_five_step_packets():
    bank = np.arange(16, dtype=np.float64).reshape(8, 2) / 20
    one, two = exp.action_sequences(bank)
    assert one.shape == (8, 5, 2) and two.shape == (64, 10, 2)
    for i in range(8):
        for j in range(8):
            np.testing.assert_array_equal(two[i*8+j, :5], one[i])
            np.testing.assert_array_equal(two[i*8+j, 5:], one[j])
    with pytest.raises(ValueError, match="bank"):
        exp.action_sequences(np.full((8, 2), 2.))


def test_saved_cost_tamper_fails_before_replay():
    path = exp.ROOT / "results/a0_pointmaze_packet_replan_oracle_20261102_DEV01.json"
    report = json.loads(path.read_text())
    report["rows"][0]["joint_costs"][0][0] += .2

    def must_not_run(*_):
        raise AssertionError("tampering should be rejected before selected replay")

    with pytest.raises(ValueError, match="cost arithmetic"):
        audit.analyze(report, exp.build(), must_not_run)
