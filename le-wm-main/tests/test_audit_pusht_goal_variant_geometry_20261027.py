import numpy as np
import pytest
from experiments.spectral_reachability.audit_pusht_goal_variant_geometry_20261027 import geometry


def test_actor_changes_do_not_change_object_label():
    a=np.zeros(7);b=a.copy();b[0]=10
    assert geometry(a,b)['object_geometry_unchanged']
    assert geometry(a,b)['agent_xy_displacement']==10


def test_angle_equivalent_modulo_two_pi():
    a=np.zeros(7);b=a.copy();b[4]=2*np.pi
    assert geometry(a,b)['object_geometry_unchanged']


@pytest.mark.parametrize('index',[2,3,4])
def test_real_object_change_detected(index):
    a=np.zeros(7);b=a.copy();b[index]=.001
    assert not geometry(a,b)['object_geometry_unchanged']


def test_invalid_state_rejected():
    with pytest.raises(ValueError):geometry([0]*6,[0]*7)
    with pytest.raises(ValueError):geometry([float('nan')]*7,[0]*7)
