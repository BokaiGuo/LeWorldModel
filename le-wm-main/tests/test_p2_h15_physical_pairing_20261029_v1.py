import copy
import pytest
from experiments.spectral_reachability import p2_h15_physical_pairing_20261029_v1 as m


def row(root,arm,steps):
    return dict(root_slot=root,arm=arm,condition='off',initial_exact=True,task_id='task',root_sha256='root',
      root_state=[0]*7,goal=[1]*7,goal_pose=[2]*3,goal_pixel_sha256='pixels',status='native_stopped' if len(steps)<15 else 'horizon_completed',
      own_returned=len(steps),responses=[dict(step=i,native_metrics=dict(native_distance=float(i),object_position_error=float(i),wrapped_object_angle_error=0.,native_success=False),terminated=False,truncated=False) for i in steps])


def test_intersection_and_missing_stop_are_not_padded():
    a=row(1,'h15_sra',range(1,16));b=row(1,'joint',range(1,15))
    pair=m.pair(a,b)
    assert pair['common_steps']==list(range(1,15))
    assert pair['step15']['comparison'] is None and pair['step15']['missing_arm']=='joint'
    assert pair['step14']['comparison']['native_distance_delta']==0


def test_goal_root_and_time_mismatch_rejected():
    a=row(0,'h15_sra',range(1,16));b=row(0,'joint',range(1,16))
    for key in ('task_id','root_sha256','root_state','goal','goal_pose','goal_pixel_sha256'):
        c=copy.deepcopy(b);c[key]='wrong'
        with pytest.raises(ValueError):m.pair(a,c)
    b['responses'][2]['step']=2
    with pytest.raises(ValueError):m.pair(a,b)


def test_real_saved_sources_and_fixed_denominator():
    r=m.build()
    assert len(r['pairs'])==4 and r['independent_generalization'] is False
    q=next(p for p in r['pairs'] if p['root_slot']==1 and p['control']=='joint')
    assert q['step15']['comparison'] is None and q['common_steps'][-1]==14
