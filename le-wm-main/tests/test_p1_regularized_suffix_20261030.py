import numpy as np
from experiments.spectral_reachability import p1_regularized_suffix_host_20261030 as method


def test_penalty_gradient_is_applied_in_logit_coordinates():
    x=np.full(10,0.1)
    raw=np.tanh(x.astype(np.float32)).reshape(5,2)
    actual=method.proposal(x,np.zeros((5,2)),raw)
    expected=(x*(1-2*method.one.saved.DAMPING/192)).astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(actual,expected)


def test_zero_penalty_reduces_to_original():
    x=np.zeros(10)
    raw=np.zeros((5,2),np.float32)
    gradient=np.full((5,2),0.01)
    np.testing.assert_array_equal(method.proposal(x,gradient,raw),method.one.proposal(x,gradient,raw))
