import numpy as np
import pytest
from experiments.spectral_reachability import main_observed_raw_fd_recompute_20261028_v1 as m


def test_opposite_direction_and_zero():
    x=np.arange(1,11,dtype=float)
    assert m.vector_metrics(x,-x)['cosine']==pytest.approx(-1)
    assert m.vector_metrics(x,x)['relative_L2']==0
    assert m.vector_metrics(np.zeros(10),x)['cosine'] is None
    assert m.vector_metrics(x,np.zeros(10))['relative_L2'] is None


@pytest.mark.parametrize('value',[np.zeros(9),np.full(10,np.nan)])
def test_bad_shape_nonfinite(value):
    with pytest.raises(ValueError):m.vector_metrics(value,np.zeros(10))


def test_actual_saved_all80_and_four_baselines():
    r=m.build()
    assert r['fixed_requests']==164 and r['fixed_pairs']==80
    assert len(r['panels'])==4 and not r['P1_admission']
    assert r['new_neural_calls']==r['new_physics_calls']==0
