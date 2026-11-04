import numpy as np
import pytest

from experiments.pc_causal_mixed_cp12_replan_saved_audit_20261103 import verify_tail


def test_tail_decode_and_selector_are_checked():
    proposals = np.zeros((1, 2, 1, 10), np.float32)
    proposals[0, 1, 0] = .5
    scores = np.array([2., 1.])
    mean, std = np.zeros(2, np.float32), np.ones(2, np.float32)
    verify_tail(proposals, scores, True, np.full((5, 2), .5), mean, std)
    with pytest.raises(ValueError):
        verify_tail(proposals, scores, False, np.full((5, 2), .5), mean, std)
    with pytest.raises(ValueError):
        verify_tail(proposals, scores, True, np.zeros((5, 2)), mean, std)
