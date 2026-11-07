import numpy as np
import pytest

from experiments.pc_tworoom_live_prefix_feedback_20261107_codex import candidate_tails


def test_candidate_tails_require_matching_legal_prefix():
    a = np.zeros((25, 2), dtype=np.float32)
    b = a.copy()
    b[10:] = .5
    assert candidate_tails(a, b).shape == (2, 15, 2)
    b[1, 0] = .1
    with pytest.raises(AssertionError):
        candidate_tails(a, b)
    b[1, 0] = 0
    b[24, 0] = 1.1
    with pytest.raises(ValueError):
        candidate_tails(a, b)
