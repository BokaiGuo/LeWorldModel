import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from experiments.pc_pusht_reset_tick_same_chain_20261107_codex import legal_actions


def test_legal_actions_validates_physical_not_normalized_bounds():
    scaler = StandardScaler().fit(np.array([[-1., -1.], [1., 1.]]))
    assert legal_actions(np.zeros((5, 10)), scaler).shape == (25, 2)
    with pytest.raises(ValueError):
        legal_actions(np.full((5, 10), 2.), scaler)
    with pytest.raises(ValueError):
        legal_actions(np.zeros((5, 2)), scaler)
