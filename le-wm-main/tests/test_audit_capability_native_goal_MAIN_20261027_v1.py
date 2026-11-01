import numpy as np
import pytest
from experiments.spectral_reachability import audit_capability_native_goal_MAIN_20261027_v1 as m


def test_agent_is_part_of_original_task_and_velocity_not_success():
    g=np.zeros(7);s=np.zeros(7);s[0]=25
    r=m.metrics(g,s)
    assert r['object_xy_l2']==0 and r['allxy_l2']==25 and not r['native_success']
    s[0]=0;s[5]=500
    r=m.metrics(g,s)
    assert r['native_success'] and r['state7_distance']==500


def test_native_raw_angle_bug_preserved_not_silently_wrapped():
    g=np.zeros(7);s=np.zeros(7);s[4]=7
    r=m.metrics(g,s)
    assert r['negative_native_angle'] and r['native_success'] and not r['wrapped_success']


@pytest.mark.parametrize('joint,angle,success',[(20.,0.,False),(19.,np.pi/9,False),(19.,0.,True)])
def test_exact_original_strict_thresholds(joint,angle,success):
    g=np.zeros(7);s=np.zeros(7);s[0]=joint;s[4]=angle
    assert m.metrics(g,s)['native_success'] is success
