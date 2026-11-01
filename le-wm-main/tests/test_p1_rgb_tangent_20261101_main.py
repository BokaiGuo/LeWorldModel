import numpy as np
import pytest

from experiments.spectral_reachability import p1_rgb_tangent_metric_20261101_main as metric


def rgb():
    a=np.full((224,224,3),255,np.uint8)
    a[90:110,90:120]=[143,163,184]
    a[50:60,50:60]=[65,105,225]
    return a


def test_identity_and_component_isolation():
    a=rgb()
    for axis in metric.AXES:
        assert np.array_equal(metric.perturb(a,axis,0),a)
        b=metric.perturb(a,axis,1)
        if axis.startswith("object"):
            assert np.array_equal(b[50:60,50:60],a[50:60,50:60])
        else:
            assert np.array_equal(b[90:110,90:120],a[90:110,90:120])


def test_metric_sqrt_and_normalization():
    jac=np.random.default_rng(17).normal(size=(192,5))
    sqrt, receipt=metric.construct(jac)
    assert np.allclose(sqrt,sqrt.T)
    assert np.isclose(np.trace(sqrt@sqrt),192)
    assert min(receipt["eigenvalues"])>0


def test_rank_failure():
    with pytest.raises(ValueError,match="unresolved"):
        metric.construct(np.zeros((192,5)))
