import numpy as np
import pytest
from experiments.spectral_reachability import main_terminal_witness_physical_contrast_20261028_v1 as m


def test_whole_native_success_does_not_use_velocity_or_object_only():
    a=m.metrics([1,1,1,1,.01,10000,-10000],[0]*7)
    assert a['native_success'] and a['native_distance']>10000
    assert not m.metrics([100,100,0,0,0,0,0],[0]*7)['native_success']


def test_native_angle_is_preserved_not_replaced_by_wrapped_diagnostic():
    a=m.metrics([0,0,0,0,3*np.pi,0,0],[0]*7)
    assert a['native_angle_term']<0 and a['angle_wrapped']==pytest.approx(np.pi)


@pytest.mark.parametrize('state', [[0]*6,[0]*8,[float('nan')]*7])
def test_invalid_state(state):
    with pytest.raises(ValueError):m.metrics(state,[0]*7)


def test_pair_keeps_all_axes():
    a=m.pair([0]*7,[1,0,2,0,.1,3,4],[0]*7)
    assert a['delta_new_minus_old']['joint_xy']==np.sqrt(5)
    assert a['delta_new_minus_old']['velocity']==5
