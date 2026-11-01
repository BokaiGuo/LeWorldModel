"""Tiny NumPy/interface tests; no tmp fixtures, CUDA or simulator."""
import numpy as np
import pytest
from experiments.spectral_reachability import p1_sampling_physical_saved_audit_20261027_v1 as a


@pytest.mark.parametrize('turns',[-3,-1,0,1,3])
def test_wrapped_object_pose_periodicity(turns):
    goal=np.zeros(7);state=np.array([1.,2.,3.,4.,.3,0.,0.])
    baseline,_=a.delta_metrics(goal,state);state[4]+=2*np.pi*turns
    current,_=a.delta_metrics(goal,state)
    assert current['object_angle']==pytest.approx(baseline['object_angle'])
    assert current['object_pose_cost']==pytest.approx(baseline['object_pose_cost'])
    assert current['wrapped_state']==pytest.approx(baseline['wrapped_state'])


def test_native_multiturn_risk_not_silently_fixed():
    goal=np.zeros(7);state=np.array([0.,0.,0.,0.,2*np.pi+.4,0.,0.])
    m,_=a.delta_metrics(goal,state)
    assert m['negative_native_angle'] and m['native_success'] and not m['wrapped_success']


def test_stable_identity_and_duplicates():
    rows=[dict(root_sha256='r',arm=x) for x in ('cem','iid_random')]
    assert a.keyed(rows)==a.keyed(rows[::-1])
    with pytest.raises(ValueError):a.keyed(rows+[rows[0]])


def test_official_includes_agent_not_just_object():
    goal=np.zeros(7);state=np.array([21.,0.,0.,0.,0.,0.,0.])
    m,_=a.delta_metrics(goal,state)
    assert m['object_pose_cost']==0 and not m['wrapped_success']
