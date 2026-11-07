import numpy as np
import pytest

from experiments.pc_tworoom_current_pixel_score_audit_20261107_codex import compare_order


def test_detects_flips_and_reports_near_ties_separately():
    old = np.array([0., 1., 2.])
    new = np.array([0., 2.00001, 2.])
    result = compare_order(old, new, atol=1e-4)
    assert result["strict_sign_changes"] == 1
    assert result["robust_reversals"] == 0
    assert result["near_ties_either"] == 1


def test_robust_reversal_is_not_hidden():
    result = compare_order([0., 1., 2.], [0., 3., 2.], atol=1e-4)
    assert result["strict_sign_changes"] == 1
    assert result["robust_reversals"] == 1


@pytest.mark.parametrize("left,right", [([0.], [0.]), ([0., np.nan], [0., 1.]),
                                        ([0., 1.], [0.]), ([[0., 1.]], [[0., 1.]])])
def test_rejects_malformed_comparisons(left, right):
    with pytest.raises(ValueError):
        compare_order(left, right, atol=1e-4)
