import numpy as np
import pytest

from experiments.pc_tworoom_hold_first_pool_physical_20261107_codex import sign_report


def test_ties_and_signs_separately_reported():
    result = sign_report([1, -2, 0, 2], [2, 3, 1, 0])
    assert result == {"total": 4, "nontie": 2, "correct": 1, "ties": 2}


def test_invalid_effects_rejected():
    with pytest.raises(ValueError):
        sign_report([np.nan], [0])
    with pytest.raises(ValueError):
        sign_report([1, 2], [1])
