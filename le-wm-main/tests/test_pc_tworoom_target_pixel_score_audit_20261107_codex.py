import numpy as np
import pytest

from experiments.pc_tworoom_target_pixel_score_audit_20261107_codex import sign_comparison


def test_parent_order_and_ties():
    values = np.r_[np.arange(12), 4., 0.]
    assert sign_comparison(values) == [1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1]


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        sign_comparison([1, 2])
    with pytest.raises(ValueError):
        sign_comparison(np.full(14, np.nan))
