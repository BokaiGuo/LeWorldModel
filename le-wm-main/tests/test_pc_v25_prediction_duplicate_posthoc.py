import numpy as np
import pytest

from experiments.pc_v25_prediction_duplicate_posthoc_20261104_codex import inspect


def test_duplicates_require_equal_outputs_and_preserve_horizon():
    five = np.zeros((1, 5, 2), np.float32)
    fifteen = np.zeros((1, 15, 2), np.float32)
    result = inspect([(five, np.zeros((1, 2)), np.zeros(1)),
                      (five.copy(), np.zeros((1, 2)), np.zeros(1)),
                      (fifteen, np.zeros((1, 2)), np.zeros(1))])
    assert result["distinct_samples"] == 2
    assert result["repeated_prediction_tokens"] == 1
    with pytest.raises(AssertionError):
        inspect([(five, np.zeros((1, 2)), np.zeros(1)),
                 (five, np.ones((1, 2)), np.zeros(1))])
