import numpy as np
import pytest
from experiments.spectral_reachability.p1_long_suffix_gradient_fd_20261029 import difference


def test_actual_float32_displacement():
    g=np.arange(20,dtype=np.float32).reshape(2,5,2)
    a=np.full((2,5,2),0.3,np.float32)
    plus=a.copy();minus=a.copy();plus[0]+=np.float32(.001);minus[0]-=np.float32(.001)
    expected=float(np.sum(g.astype(float)*(plus.astype(float)-minus.astype(float))/2))
    result=difference(expected,-expected,g,plus,minus)
    assert result['relative_error']==0


def test_reject_nonfinite():
    with pytest.raises(ValueError):
        difference(np.nan,0,np.zeros((2,5,2)),np.zeros((2,5,2)),np.zeros((2,5,2)))
