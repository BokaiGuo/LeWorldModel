import numpy as np
import pytest

from experiments.spectral_reachability.run_reacher_pc_tangent_full_physical import action_bank


def valid_row():
    return {"status": "complete", "compiler_receipt": {"selected_proposal_index": None},
            "prefix_raw": np.zeros((7, 5, 2)).tolist(), "tail_raw": np.zeros((5, 10, 2)).tolist(),
            "prefix_normalized": np.zeros((7, 1, 10)).tolist(),
            "tail_normalized": np.zeros((5, 2, 10)).tolist()}


def test_raw_normalized_exact_binding_and_all_candidates():
    args = (np.zeros(10, np.float32), np.ones(10, np.float32),
            np.array([-1, -1]), np.array([1, 1]))
    row = valid_row()
    prefixes, tails = action_bank(row, *args)
    assert prefixes.shape == (7, 5, 2)
    assert tails.shape == (5, 10, 2)
    row["prefix_raw"][4][1][0] = 0.1
    with pytest.raises(ValueError, match="differ"):
        action_bank(row, *args)


def test_rejects_wrong_source_selection_and_bounds():
    args = (np.zeros(10, np.float32), np.ones(10, np.float32),
            np.array([-1, -1]), np.array([1, 1]))
    row = valid_row()
    row["compiler_receipt"]["selected_proposal_index"] = 3
    with pytest.raises(ValueError, match="selection"):
        action_bank(row, *args)
    row = valid_row()
    row["tail_raw"][2][1][0] = 2
    with pytest.raises(ValueError, match="physical action"):
        action_bank(row, *args)
