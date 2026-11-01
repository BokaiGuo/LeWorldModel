import numpy as np
import pytest
from experiments.spectral_reachability.p1_fixed40_reach_gain_20261029 import gain


def test_gain_uses_zero_feasible_cost():
    assert gain(np.zeros(192),np.ones(192),.5)['regularized_gain'] == .5


def test_rejects_impossible_or_nonfinite():
    with pytest.raises(ValueError): gain(np.zeros(192),np.ones(192),1.1)
    with pytest.raises(ValueError): gain(np.zeros(192),np.ones(192),float('nan'))
