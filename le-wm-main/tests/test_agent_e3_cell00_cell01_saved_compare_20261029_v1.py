from copy import deepcopy
import struct
import pytest
from experiments.spectral_reachability import agent_e3_cell00_cell01_saved_compare_20261029_v1 as m


def side(cell):
    goal=[0.,0.,0.,0.,0.,0.,0.]
    obs=dict(state=[1.,2.,3.,4.,0.,0.,0.],snapshot={'x':1},pixel_sha256='root',goal=goal,goal_rgb_sha256='goal')
    guard=dict(original_root_sha256='root-id',original_goal=goal,new_goal=goal,new_goal_pixel_sha256='goal',
               history_pixels=['a','b','c'],root_observation=deepcopy(obs))
    return dict(cell=cell,episode=f'cell{cell:02d}',guard=guard,observations={0:deepcopy(obs),5:deepcopy(obs)})


def test_same_root_and_missing_early_stop():
    a,b=side(0),side(1);del b['observations'][5]
    out=m.compare(a,b)
    assert out['root_goal_aligned'] and out['checkpoints'][0]['paired_joint_xy_delta']==0
    assert out['checkpoints'][1]['cell01'] is None and out['checkpoints'][1]['paired_joint_xy_delta'] is None
    assert all(row['paired_joint_xy_delta'] is None for row in out['checkpoints'][2:])


@pytest.mark.parametrize('key',['original_root_sha256','new_goal','new_goal_pixel_sha256','history_pixels'])
def test_different_root_or_goal_rejected(key):
    a,b=side(0),side(1);b['guard'][key]='bad'
    with pytest.raises(ValueError):m.compare(a,b)


def test_missing_guard_is_not_filled():
    a,b=side(0),side(1);b['guard']=None
    out=m.compare(a,b)
    assert not out['root_goal_aligned'] and out['blocked_reasons'] and all(x['paired_joint_xy_delta'] is None for x in out['checkpoints'])


def test_candidate_goal_drift_rejected():
    a,b=side(0),side(1);b['observations'][5]['goal_rgb_sha256']='other'
    with pytest.raises(ValueError,match='observation task goal'):m.compare(a,b)


def test_requires_main_terminal_confirmation_before_reading():
    with pytest.raises(ValueError,match='terminal confirmation'):m.build('a'*64,'b'*64,'not-confirmed')


def test_dx_mask_applies_to_new_actions_not_original_prefix():
    def rec(step,x,y):return dict(step=step,command=[x,y],raw_bits=struct.pack('<2f',x,y).hex())
    m.validate_dx_return(rec(5,0.,1.))
    m.validate_dx_return(rec(6,0.,0.))
    with pytest.raises(ValueError,match='positive float32 zero'):m.validate_dx_return(rec(6,0.,-0.))
    with pytest.raises(ValueError,match='positive float32 zero'):m.validate_dx_return(rec(6,0.,1.))


def test_main06_only_native_semantics():
    v=m.inspect_saved(str(m.BASE),m.BASE_REPORT,m.BASE_COMPLETION,'cell00',0,m.AUTH0)
    assert v['audit']['transport_complete'] and v['returned']==20 and v['terminal_status']=='completed20'
    assert v['audit']['full_saved_strict'] is False
