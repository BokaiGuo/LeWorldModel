import numpy as np
import pytest

from experiments.pc_tworoom_block_b_reencode_alignment_20261107_codex import summarize_root


def test_alignment_uses_same_candidate_and_distinguishes_oracles():
    observed = np.arange(64, dtype=float)[:, None]
    predicted = observed + 1
    goal = np.array([0.0])
    costs = np.arange(64, dtype=float)[::-1]
    outcomes = [{"ever": i == 63, "prefix_success": False, "final_distance": float(63 - i)}
                for i in range(64)]
    row, encoded = summarize_root(predicted, observed, goal, costs, outcomes)
    assert row["feature_mse"] == 1
    assert row["model_selected"]["index"] == 63
    assert row["encoded_oracle"]["index"] == 0
    assert row["physical_oracle"]["index"] == 63
    assert row["model_selected"]["ever"]
    assert not row["encoded_oracle"]["ever"]
    np.testing.assert_array_equal(encoded, np.arange(64) ** 2)


def test_alignment_rejects_misaligned_features():
    with pytest.raises(ValueError):
        summarize_root(np.zeros((64, 2)), np.zeros((63, 2)), np.zeros(2),
                       np.zeros(64), [{}] * 64)
