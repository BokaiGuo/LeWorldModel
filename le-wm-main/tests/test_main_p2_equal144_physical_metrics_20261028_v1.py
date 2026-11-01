import math
import pytest
from experiments.spectral_reachability import main_p2_equal144_physical_metrics_20261028_v1 as m


def test_metric_decomposition():
    x=m.metrics([3,4,0,12,.2,5,0],[0]*7)
    assert x['joint_xy_error']==13
    assert x['native7_squared_components']==dict(joint_xy=169.,raw_angle=.2**2,velocity=25.)
    assert x['native7_distance']**2==pytest.approx(194.04)


@pytest.mark.parametrize('state',[[1]*6,[float('nan')]*7,[float('inf')]*7])
def test_reject_bad_state(state):
    with pytest.raises(ValueError): m.metrics(state,[0]*7)


def test_native_not_object_only_success():
    assert not m.metrics([30,0,0,0,0,0,0],[0]*7)['native_success']
    assert not m.metrics([20,0,0,0,0,0,0],[0]*7)['native_success']
    assert not m.metrics([0,0,0,0,math.pi/9,0,0],[0]*7)['native_success']


def test_preserves_native_raw_angle_behavior():
    x=m.metrics([0,0,0,0,7,0,0],[0]*7)
    assert x['native_angle_term']<0
    assert x['native_success']  # native formula, NOT silently corrected wrap


def test_plan_reward_checks_ignore_old_goal_warmup():
    row=dict(own_returned=1,goal=[0]*7,selected_plan=dict(commands=[[.1,.2]]),final_state=[0]*7)
    event=dict(kind='returned_transition',branch=0,phase='plan',step=1,
               state=[0]*7,returned_goal_state=[0]*7,command=[.1,.2],reward=0.,terminated=True,truncated=False)
    warm=dict(kind='returned_transition',branch=0,phase='warmup',step=1)
    assert len(m.transitions([warm,event],[row]))==1
    event['reward']=1
    with pytest.raises(ValueError): m.transitions([event],[row])
