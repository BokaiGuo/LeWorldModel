import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_state_conditioned_support import (
    _nearest_source_indices,
    _source_anchor_rows,
)


def test_state_conditioned_support_orders_nearest_frozen_source_contexts_stably():
    bank = np.asarray([[3.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    # The equal-distance tie is resolved by source-index order, not RNG.
    assert _nearest_source_indices(np.asarray([0.0, 0.0]), bank, count=3).tolist() == [1, 2, 3]


def test_state_conditioned_support_keeps_source_windows_causal_and_complete():
    rows = _source_anchor_rows(np.asarray([20, 21, 30]), np.asarray([0, 20, 41]), count=8, seed=3)
    # Episode 0 is too short; returned anchors must belong to either [35,35]
    # or [56,65], whose predecessor history and next packet stay in episode.
    assert np.all(((rows == 35) | ((rows >= 56) & (rows <= 65))))


def test_state_conditioned_support_rejects_invalid_neighbor_request():
    with pytest.raises(ValueError, match="invalid neighbor"):
        _nearest_source_indices(np.zeros(2), np.zeros((2, 2)), count=3)
