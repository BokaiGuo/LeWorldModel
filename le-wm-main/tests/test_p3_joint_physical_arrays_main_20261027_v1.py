import numpy as np
import pytest
from experiments.spectral_reachability import p3_joint_physical_arrays_main_20261027_v1 as m


def test_pose_wrap_xy_and_velocity_exclusion():
    assert m.pose([8, 9, 3, 4, 2*np.pi, 99, 99], [0]*7) == dict(xy_distance=5., angle_distance=0., pose_cost=5.)


def test_identity_not_array_order_and_duplicates_refused():
    rows = [dict(root_sha256='b', method='x'), dict(root_sha256='a', method='x')]
    assert m.index(rows) == m.index(rows[::-1])
    with pytest.raises(ValueError):
        m.index(rows+[rows[0]])


def test_same_q_mean_over_native192():
    q = np.zeros((192, 192)); q[0, 0] = 1
    assert m.qcost(np.ones(192), np.zeros(192), q) == 1/192


@pytest.mark.parametrize('state', [[0]*6, [float('nan')]*7])
def test_bad_physical_states_refused(state):
    with pytest.raises(ValueError):
        m.pose(state, [0]*7)


def test_scalar_disagreement_refused():
    with pytest.raises(ValueError):
        m.equal(1., 2.)


def test_official_agent_goal_is_not_object_only_success():
    r = m.official([100, 100, 0, 0, 0, 0, 0], [0]*7)
    assert not r['official_success'] and r['object_only_success']


def test_nonprojector_task_metric_refused():
    with pytest.raises(ValueError):
        m.qcost(np.ones(192), np.zeros(192), 2*np.eye(192))
