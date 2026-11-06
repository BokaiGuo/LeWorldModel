import numpy as np
import pytest

from experiments.pc_pusht_scored_success_failure_visual_audit_dev_20261105_codex import chosen


def test_successes_then_lowest_scored_failures_stable():
    scores = np.arange(100, dtype=float)
    success = np.zeros(100, dtype=bool)
    success[[50, 90]] = True
    result = chosen(scores, success)
    np.testing.assert_array_equal(result[:2], [50, 90])
    np.testing.assert_array_equal(result[2:], [i for i in range(63) if i != 50])
    assert len(result) == 64


def test_cannot_silently_drop_successes():
    with pytest.raises(ValueError):
        chosen(np.arange(70), np.ones(70, dtype=bool))
