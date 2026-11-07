import numpy as np
import pytest

from experiments.pc_tworoom_block_b_temporal_alignment_20261107_codex import temporal_summary


def test_temporal_success_distinguishes_full_trace_from_packet_samples():
    predicted = np.ones((64, 5, 2))
    actual = np.ones((64, 5, 2))
    predicted[0, -1] = 0
    predicted[1, 0] = 0
    actual[2, 2] = 0
    outcomes = []
    for i in range(64):
        trace = [{"distance": float(i + t), "terminated": i == 1 and t == 2} for t in range(25)]
        outcomes.append({"trace": trace, "ever": i == 1})
    row, _, _ = temporal_summary(predicted, actual, np.zeros(2), outcomes)
    assert row["ever"] == row["ever_not_final"] == row["ever_not_sampled"] == 1
    assert row["sampled"] == row["final"] == 0
    assert row["selections"]["pred_min5"]["index"] == 0  # stable tie
    assert row["selections"]["encoded_min5_oracle"]["index"] == 2
    assert [p["step"] for p in row["points"]] == [5, 10, 15, 20, 25]


def test_temporal_shapes_fail_closed():
    with pytest.raises(ValueError):
        temporal_summary(np.zeros((64, 5, 2)), np.zeros((64, 4, 2)), np.zeros(2), [{}] * 64)
