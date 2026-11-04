import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability.run_reacher_pc_q_common_suffix import ARMS, fixed_actions


def sample_row():
    row = {"arms": {}, "indices": {}}
    for index, arm in enumerate(ARMS):
        prefix = np.full((5, 2), index / 10, dtype=np.float32)
        row["indices"][arm] = index
        row["arms"][arm] = {
            "status": "complete", "candidate_index": index, "prefix_raw": prefix.tolist(),
            "suffix_raw": np.full((10, 2), index / 10, dtype=np.float32).tolist(),
            "advanced_action_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
        }
    return row


def test_all_fixed_suffixes_and_bindings():
    row = sample_row()
    prefixes, tails = fixed_actions(row, np.array([-1, -1]), np.array([1, 1]))
    assert set(prefixes) == set(ARMS)
    assert set(tails) == {"passive_zero", *(arm + "_tail" for arm in ARMS)}
    assert np.array_equal(tails["passive_zero"], np.zeros((10, 2)))
    row["arms"]["selected"]["prefix_raw"][0][0] = 0.5
    with pytest.raises(ValueError, match="digest"):
        fixed_actions(row, np.array([-1, -1]), np.array([1, 1]))


def test_rejects_censored_or_out_of_range_actions():
    row = sample_row()
    row["arms"]["zero"]["status"] = "error_or_censored"
    with pytest.raises(ValueError, match="incomplete"):
        fixed_actions(row, np.array([-1, -1]), np.array([1, 1]))
    row = sample_row()
    row["arms"]["zero"]["suffix_raw"][0][0] = 2
    with pytest.raises(ValueError, match="bounds"):
        fixed_actions(row, np.array([-1, -1]), np.array([1, 1]))
