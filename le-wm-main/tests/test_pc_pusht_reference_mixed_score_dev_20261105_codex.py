import numpy as np
import pytest

from experiments.pc_pusht_reference_mixed_score_dev_20261105_codex import make_bank
from experiments.pc_pusht_continuous_warmup_score_dev_20261105_codex import bank


def test_mixed_bank_preserves_independent_and_conditions_only_second_half():
    reference = np.full((15, 2), .1, np.float32)
    independent = bank()
    result = make_bank(9, reference, independent)
    np.testing.assert_array_equal(result[:32], independent[:32])
    np.testing.assert_array_equal(result, make_bank(9, reference, independent))
    assert np.max(np.abs(result[32:48] - reference)) <= .015001
    assert np.max(np.abs(result[48:] - reference)) <= .150001
    assert not np.array_equal(result[32:], make_bank(11, reference, independent)[32:])
    assert all(not np.array_equal(x, reference) for x in result)


def test_duplicate_reference_is_rejected():
    independent = bank()
    with pytest.raises(ValueError):
        make_bank(9, independent[0], independent)
