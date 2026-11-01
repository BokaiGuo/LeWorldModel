import pytest
from experiments.spectral_reachability import main_nonlinear_suffix_rgb_reuse_20261028_v1 as m


def test_raw10_exact_identity_including_signed_zero():
    a=[[0.,0.]]*10;m.exact(a,a)
    b=[[0.,0.]]*9+[[-0.,0.]]
    with pytest.raises(ValueError):m.exact(a,b)


@pytest.mark.parametrize('a',[[[0.,0.]]*5,[[2.,0.]]*10,[[float('nan'),0.]]*10])
def test_wrong_control_domain_refused(a):
    with pytest.raises(ValueError):m.exact(a,[[0.,0.]]*10)
