import numpy as np
import pytest

from experiments import pc_pointmaze_rgb_forward_clock_audit_20261103_codex as audit


def test_geometry_clock_directly_not_pixels():
    states=np.zeros((2,64,86,5))
    for t in range(86):
        states[:,:,t,0]=t*.01
        states[:,:,t,4]=t*.01
    offset=np.broadcast_to([2.,3.],(64,2)).copy()
    geoms=np.zeros((2,64,86,3))
    boundary=np.zeros((2,64,18,3))
    for t in range(1,18):
        boundary[0,:,t,:2]=states[0,:,5*t-1,:2]+offset
        geoms[1,:,5*t,:2]=states[1,:,5*t,:2]+offset
    result=audit.analyze(states,geoms,boundary,offset)
    assert np.all(result["raw_minus_previous"]==0)
    assert np.all(result["sync_minus_current"]==0)
    np.testing.assert_allclose(result["raw_minus_current"][:,:,0],-.01,atol=1e-15)
    states[1,0,4,0]+=1
    with pytest.raises(ValueError,match="trajectory"):
        audit.analyze(states,geoms,boundary,offset)


def test_invalid_shapes_fail_closed():
    with pytest.raises(ValueError):
        audit.analyze(np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2))
