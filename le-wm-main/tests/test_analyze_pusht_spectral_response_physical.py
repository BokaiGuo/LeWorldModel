import numpy as np
import pytest
from experiments.spectral_reachability.analyze_pusht_spectral_response_physical import state_response


def test_agent_motion_does_not_count_as_object_response():
    e=float(np.float32(.05)); p=[]
    for axis in (0,1):
        for sign in (-1,1):
            state=np.array([0.,0.,20.,30.,.4,0.,0.]);state[axis]=sign*e
            p.append(dict(axis=axis,sign=sign,status='completed',state=state.tolist()))
    p.append(dict(axis=None,sign=0,status='completed',state=[0.,0.,20.,30.,.4,0.,0.]))
    r=state_response(p,.05)
    assert r['object_xy_jacobian_norm']==0
    assert r['object_angle_jacobian_norm']==0
    assert r['agent_xy_jacobian_norm']==pytest.approx(2**.5)
    assert r['all_five_object_poses_exact'] is True


def test_incomplete_probes_fail_closed():
    with pytest.raises(ValueError):state_response([],.05)
