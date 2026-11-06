import numpy as np
import pytest

from experiments.audit_pc_jepa_v12_neural_replay_20261105_codex import compare


def test_compare_keeps_signed_differences_and_counts_tolerance():
    old = np.zeros(64)
    new = old.copy()
    new[0] = -2e-6
    result = compare(old, new)
    assert result["within_1e_6"] == 63
    assert result["max_abs_difference"] == pytest.approx(2e-6)
    assert result["new_minus_original"][0] == pytest.approx(-2e-6)


@pytest.mark.parametrize("bad", [np.zeros(63), np.full(64, np.nan)])
def test_compare_rejects_incomplete_or_nonfinite_scores(bad):
    with pytest.raises(ValueError):
        compare(np.zeros(64), bad)
