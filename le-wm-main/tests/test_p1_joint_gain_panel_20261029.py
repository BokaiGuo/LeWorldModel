import numpy as np
import pytest
from experiments.spectral_reachability.p1_joint_gain_panel_20261029 import gain


def test_regularized_gain():
    assert gain(np.ones(192),np.zeros(192),.25) == .75


def test_rejects_invalid_or_infeasible():
    with pytest.raises(ValueError): gain(np.ones(191),np.zeros(192),0)
    with pytest.raises(ValueError): gain(np.ones(192),np.zeros(192),1.1)
