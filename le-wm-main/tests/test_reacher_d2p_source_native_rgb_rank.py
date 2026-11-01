import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_source_native_rgb_rank import rank_summary, rgb_difference


def test_rgb_rank_uses_paired_costs_and_reports_regret():
    result = rank_summary([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert result["spearman"] == -1.0
    assert not result["argmin_agrees"]
    assert result["observed_regret"] == 2.0


@pytest.mark.parametrize("pred,obs", [([1.0], [1.0]), ([1.0, 2.0], [1.0]), ([1.0, np.nan], [1.0, 2.0])])
def test_rgb_rank_rejects_invalid_or_unpaired_costs(pred, obs):
    with pytest.raises(ValueError, match="paired"):
        rank_summary(pred, obs)


def test_one_gray_discrepancy_is_reported_not_accepted():
    reference = np.zeros((4, 4, 3), np.uint8)
    actual = reference.copy()
    actual[2, 1, 0] = 1
    result = rgb_difference(actual, reference)
    assert not result["exact"]
    assert result["changed_components"] == 1
    assert result["max_pixel_error"] == 1
    assert result["changed_coordinates"] == [[2, 1, 0]]


def test_rgb_diagnostic_distinguishes_wrong_shape():
    result = rgb_difference(np.zeros((4, 4, 3), np.uint8), np.zeros((2, 2, 3), np.uint8))
    assert not result["exact"]
    assert "max_pixel_error" not in result
