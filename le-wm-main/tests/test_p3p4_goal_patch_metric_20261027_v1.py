"""Goal RGB/array/metadata tests only; no neural or simulator execution."""
import builtins
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p3p4_goal_patch_metric_20261027_v1 as m

ROOT=Path(__file__).resolve().parents[1]


def goal():
    image=np.full((224,224,3),255,np.uint8)
    image[104:115,104:119]=[119,136,153]
    image[60:66,60:66]=[65,105,225]
    image[59,60:66]=[100,120,230] # outline must also disappear
    return image


def test_goal_only_fixed_cells_and_complete_sprite():
    image=goal();before=image.copy();plan=m.catalog(image)
    assert np.array_equal(image,before)
    assert len(plan['images'])==6+len(plan['cells'])
    assert plan['cells']==sorted(plan['cells'],key=lambda c:(c['row'],c['col']))
    clean=plan['images'][1]['rgb'];assert np.all(clean[59:66,60:66]==255)
    objectmask=np.all(image==[119,136,153],axis=-1)
    assert np.array_equal(clean[objectmask],image[objectmask])
    for cell,edit in zip(plan['cells'],plan['images'][6:]):
        left,top,right,bottom=cell['bounds'];expected=clean.copy();expected[top:bottom,left:right]=255
        assert np.array_equal(edit['rgb'],expected)
    assert m.physical_equal_catalog(m.catalog(image),{k:v for k,v in plan.items() if k!='images'})


@pytest.mark.parametrize('damage',['agent','object','dtype','edge','background','destination'])
def test_illegal_image_refuses_without_fallback(damage):
    image=goal()
    if damage=='agent':image[np.all(image==[65,105,225],axis=-1)]=255
    if damage=='object':image[np.all(image==[119,136,153],axis=-1)]=255
    if damage=='dtype':image=image.astype(float)
    if damage=='edge':image[0,0]=[65,105,225]
    if damage=='background':image[58,59]=0
    if damage=='destination':image[28,28]=0
    with pytest.raises(ValueError):m.catalog(image)


def vectors():
    legacy=np.zeros((5,192));legacy[1:,0]=[1,2,3,4]
    v=np.zeros((8,192));v[2:6,0]=[1,2,3,4];v[6,0]=3;v[6,1]=1;v[7,2]=1
    return v,legacy


def test_native_task_projector_annihilates_nuisance():
    events=[];v,g=vectors();r=m.compile_metric(v,g,events.append);q=np.asarray(r['projection'])
    assert r['task_rank']==2 and r['nuisance_rank']==1 and r['learned_parameters']==0
    assert np.allclose(q@q,q) and np.allclose(q,q.T)
    assert np.allclose(q[:,0],0) and q[1,1]==pytest.approx(1) and q[2,2]==pytest.approx(1)
    assert [e['role'] for e in events if e['completed']]==['nuisance','object']
    assert m.native_cost(np.eye(192)[0],np.zeros(192),q)==pytest.approx(0,abs=1e-28)
    assert m.native_cost(np.eye(192)[1],np.zeros(192),q)==pytest.approx(1/192)


def test_zero_task_rank_strict_refusal():
    v,g=vectors();v[6:]=0;v[6:,0]=1
    with pytest.raises(ValueError,match='surviving'):m.compile_metric(v,g)


def test_wrong_native_goal_or_patch_space_rejected():
    v,g=vectors();v[0,10]=1
    with pytest.raises(ValueError):m.compile_metric(v,g)
    with pytest.raises(ValueError):m.native_cost(np.zeros((256,192)),np.zeros(192),np.eye(192))


def test_roundoff_floor_refuses_false_task_rank():
    v,g=vectors();v[6:]=0;v[6:,0]=1
    with pytest.raises(ValueError,match='surviving'):m.compile_metric(v,g)
    v[6,1]=1e-4 # a real independently specified response survives
    result=m.compile_metric(v,g)
    assert result['task_rank']==1 and result['goal_anchor_max_difference']<1e-8


@pytest.mark.parametrize('damage',['bool','nan','wrongwidth'])
def test_invalid_encoder_outputs_rejected(damage):
    v,g=vectors()
    if damage=='bool':v=v.astype(bool)
    elif damage=='nan':v[6,1]=np.nan
    else:v=v[:,:191]
    with pytest.raises(ValueError):m.compile_metric(v,g)


def test_action_response_consistent_new_metric():
    v,g=vectors();metric=m.compile_metric(v,g);goalvec=np.zeros(192);goalvec[1]=1
    j=np.zeros((192,20));j[1,0]=1
    result=m.compile_action_response({'goal':goalvec},metric,j,np.zeros(192),2)
    assert result['signal']['chi']==pytest.approx(0)
    assert result['proposal_logits'][0]>0 and result['svd_calls']==2
    assert np.asarray(result['raw_proposal']).shape==(10,2)
    assert result['new_predict_calls']==result['new_jacobian_materializations']==0


def test_common_bank_repair_can_change_selection_without_state():
    packet=dict(goal=np.zeros(192),projection=np.eye(192));q=np.zeros((192,192));q[1,1]=1
    candidates=[]
    for index,(agent,obj) in enumerate([(0,2),(10,0)]):
        context=np.zeros((3,192));context[-1,:2]=[agent,obj]
        candidates.append(dict(horizon=2,index=index,actions=np.zeros((10,2)),context={'latent_history':context.tolist()}))
    r=m.select_common_bank(packet,{'projection':q},candidates)
    assert r['winners']['old_Q']['index']==0 and r['winners']['goal_patch_Q']['index']==1
    assert r['cost_evaluations']==4 and r['key_comparisons']==2 and r['physical_ready'] is False
    with pytest.raises(ValueError):m.select_common_bank(packet,{'projection':q},candidates+candidates[:1])


def test_stable_tie_uses_horizon_then_pool_index():
    p=dict(goal=np.zeros(192),projection=np.eye(192));metric={'projection':np.eye(192)}
    c=[dict(horizon=h,index=k,actions=np.zeros((5*h,2)),context={'latent_history':np.zeros((3,192)).tolist()}) for h,k in [(3,0),(2,2),(2,0)]]
    r=m.select_common_bank(p,metric,c)
    assert all(w['horizon']==2 and w['index']==0 for w in r['winners'].values())


def test_real_goal_assets_and_model_free_preparation(tmp_path,monkeypatch):
    original=builtins.__import__
    def no_runtime(name,*a,**kw):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    out=tmp_path/'prepared';receipt=m.prepare(ROOT,out)
    p=m.validate_prepared(out,m.io.sha((out/'prepare.json').read_bytes()))
    assert len(p['goals'])==4 and p['expected_encoder_samples']==sum(6+len(g['catalog']['cells']) for g in p['goals'])
    assert receipt['new_neural_calls']==receipt['new_physical_steps']==0
    assert all(g['catalog']['object_pixels']==183 for g in p['goals'])
    with pytest.raises(ValueError):m.prepare(ROOT,out)
    with pytest.raises(ValueError):m.validate_prepared(out,'0'*64)
    (out/'protocol.json').write_text('{}')
    with pytest.raises(ValueError):m.validate_prepared(out,m.io.sha((out/'prepare.json').read_bytes()))
