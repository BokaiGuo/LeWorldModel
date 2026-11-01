from fractions import Fraction
import inspect
import numpy as np
import pytest
from experiments.spectral_reachability import main_goal_area_weights_20261028_v1 as m


def test_fixed_area_integral_and_no_all196_requirement():
    counts=[0]*256;counts[2]=3;counts[8]=7
    result=m.area(counts)
    weights=[Fraction(int(x['numerator']),int(x['denominator'])) for x in result['weights_rational']]
    assert weights[2]==Fraction(3,10) and weights[8]==Fraction(7,10)
    assert sum(weights)==1 and result['pixel_count']==10
    assert list(inspect.signature(m.area).parameters)==['counts']


@pytest.mark.parametrize('counts', [[0]*256, [1]*255, [True]*256, [197]*256, [-1]*256, [1.]*256])
def test_missing_or_invalid_support_has_no_fallback(counts):
    with pytest.raises(ValueError):m.area(counts)


def test_arithmetic_operator_and_uniform_special_case():
    rng=np.random.default_rng(17)
    a=rng.normal(size=(256,384)).astype(np.float32);b=np.zeros_like(a)
    uniform=m.area([1]*256)
    assert np.isclose(m.weighted_cost(a,b,uniform['weights_float64']), np.mean(a.astype(np.float64)**2), rtol=1e-14)
    counts=[0]*256;counts[9]=5
    weighted=m.area(counts)
    assert m.weighted_cost(a,b,weighted['weights_float64'])==np.mean(a[9].astype(np.float64)**2)
    q=np.asarray(weighted['sqrt_diagonal_for_original_flat_MSE'])[:,None]
    assert np.isclose(np.mean((q*a.astype(np.float64))**2),m.weighted_cost(a,b,weighted['weights_float64']),rtol=1e-14)


def test_bad_patch_abi_rejected():
    a=np.zeros((256,384),np.float32)
    with pytest.raises(ValueError):m.weighted_cost(a[:255],a,np.ones(256)/256)
    with pytest.raises(ValueError):m.weighted_cost(a.astype(float),a,np.ones(256)/256)
    with pytest.raises(ValueError):m.weighted_cost(a,a,np.zeros(256))
