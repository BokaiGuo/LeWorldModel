from copy import deepcopy
import struct

import pytest

from experiments.spectral_reachability import agent_e3_cell08_cell09_saved_compare_20261029_v1 as m


def side(cell):
    goal = [0.]*7
    obs = dict(state=[1.,2.,3.,4.,0.,0.,0.], snapshot={'x':1}, pixel_sha256='root', goal=goal, goal_rgb_sha256='goal')
    guard = dict(original_root_sha256='root', original_goal=goal, new_goal=goal, new_goal_pixel_sha256='goal', history_pixels=['a','b','c'], root_observation=deepcopy(obs))
    return dict(cell=cell, guard=guard, cp5=dict(prefix_raw_bits='aa', observation=deepcopy(obs)), observations={0:deepcopy(obs),5:deepcopy(obs)})


def test_same_time_and_no_padding():
    a,b=side(8),side(9); del b['observations'][5]
    out=m.check_pair(a,b)
    assert out['aligned'] and out['cp5_aligned'] and out['checkpoints'][0]['dx_minus_dxy']==0
    assert all(x['dx_minus_dxy'] is None for x in out['checkpoints'][1:])


@pytest.mark.parametrize('key', ['original_root_sha256','original_goal','new_goal','new_goal_pixel_sha256','history_pixels'])
def test_different_root_or_goal_rejected(key):
    a,b=side(8),side(9);b['guard'][key]='bad'
    with pytest.raises(ValueError):m.check_pair(a,b)


def test_cp5_missing_or_prefix_mismatch():
    a,b=side(8),side(9);b['cp5']=None
    out=m.check_pair(a,b)
    assert not out['cp5_aligned'] and out['checkpoints'][1]['dx_minus_dxy'] is None
    b=side(9);b['cp5']['prefix_raw_bits']='bad'
    with pytest.raises(ValueError):m.check_pair(a,b)


def test_goal_drift_and_positive_zero():
    a,b=side(8),side(9);b['observations'][5]['goal_rgb_sha256']='bad'
    with pytest.raises(ValueError):m.check_pair(a,b)
    def r(step,y):return dict(step=step,command=[0.,y],raw_bits=struct.pack('<2f',0.,y).hex())
    m.old.validate_dx_return(r(5,1.));m.old.validate_dx_return(r(6,0.))
    with pytest.raises(ValueError):m.old.validate_dx_return(r(6,-0.))
