import copy
import json
import sys
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_physical_handoff_20261029_v1 as m

def example():
    task={'binding':{'task_id':'t','root_sha256':'r'}}
    ident=dict(slot=3,root=0,mode='fixedH1',K=17)
    row=dict(**ident,status='tree_complete_not_admitted')
    arm=dict(**row,plan_available=True)
    plan=dict(actions=[[0.0,-0.0]]*5,path=[{'x':1}])
    tree=dict(status=row['status'],mode='fixedH1',K=17,backend_kind='HOST_CUDA',
      root_task_sha256=m.digest(task),model_plan=plan,
      replays=[dict(mode=x,status='returned',path=copy.deepcopy(plan['path'])) for x in ('whole','segmented')])
    return task,row,arm,tree

def test_six_roster_and_budget():
    assert len(m.SLOTS)==6 and len(set(m.SLOTS))==6
    x=m.budget(); assert x['six_slot_upper']['physics_step']==924
    assert x['six_slot_upper']['png']==306 and not x['approved'] and x['wall_seconds'] is None

def test_literal_signed_zero_and_no_release():
    t,r,a,x=example(); out=m.plan_row(3,0,'fixedH1',t,r,a,x)
    assert out['first5_raw_f32_hex'].startswith('0000000000000080')
    assert out['physical_dispatch_ready'] is False and out['physical_outcome'] is None

@pytest.mark.parametrize('mutation',('root','task','domain','dtype','replay','plan','slot'))
def test_reject_wrong_identity_or_plan(mutation):
    t,r,a,x=example()
    if mutation=='root': a['root']=1
    if mutation=='task': x['root_task_sha256']='wrong'
    if mutation=='domain': x['model_plan']['actions'][0][0]=2.0
    if mutation=='dtype': x['model_plan']['actions'][0][0]=0.1
    if mutation=='replay': x['replays'][0]['path']=[]
    if mutation=='plan': x['model_plan']=None
    if mutation=='slot': r['slot']=4
    with pytest.raises(ValueError): m.plan_row(3,0,'fixedH1',t,r,a,x)

def test_four_physical_rows_all_bound():
    tasks=[{'binding':{'task_id':str(i),'root_sha256':str(i)}} for i in range(2)]
    rows=[dict(root_slot=i//2,task_id=str(i//2),root_sha256=str(i//2),goal=[1],goal_pose=[2],root_state=[3],initial_exact=True) for i in range(4)]
    rec=dict(schema='agent_p3_h3_witness_physical_20261029_v1.consume',status='h3_witness4_control20_native_recorded_not_admitted',physical_report_sha256=m.PHYSICAL_SHA,saved_PNG_verified=384,rows=copy.deepcopy(rows))
    assert len(m.physical_roots(tasks,{'rows':rows},rec))==2
    rec['rows'][3]['goal']=[9]
    with pytest.raises(ValueError): m.physical_roots(tasks,{'rows':rows},rec)

def test_export_history_byte_identity_and_mutations():
    context=dict(action_history=[[0.0]*10,[0.0]*10],latent_history=[[0.0],[0.0],[0.0]])
    t=dict(binding=dict(task_id='t',root_sha256='r',task_sha256='s',blind_sha256='b',task_contract_sha256='c'),
      root_context=context,goal=[0],projection=[[1]],action_mean=[0.0,0.0],action_std=[1.0,1.0])
    ex=dict(task_id='t',blind_sha256='b',history_semantics='actual_moving_raw_between_observed0_5_10',
      task={k:copy.deepcopy(t[k]) for k in ('root_context','goal','projection','action_mean','action_std')})
    ex['task'].update(task_contract_sha256='c',root_raw_action_history=[[0.0,0.0] for _ in range(10)])
    assert m.history_export(t,ex,'s')['saved_history_binding_verified']
    for mutation in ('time','raw','source','latent'):
        e=copy.deepcopy(ex)
        if mutation=='time': e['history_semantics']='actual_cp5'
        if mutation=='raw': e['task']['root_raw_action_history'][0][0]=0.5
        if mutation=='source': e['blind_sha256']='other'
        if mutation=='latent': e['task']['root_context']['latent_history'][0][0]=1.0
        with pytest.raises(ValueError): m.history_export(t,e,'s')

def test_tempdisk_exclusive_and_verify(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path); (tmp_path/'results').mkdir()
    monkeypatch.setattr(m,'build',lambda:{'denominator':6,'physical_release':False})
    path=tmp_path/'results'/(m.NAME+'_fixture.json')
    monkeypatch.setattr(sys,'argv',['x','analyze','--output',str(path)]); m.main()
    saved_sha=m.sha(path); assert json.loads(path.read_bytes())['denominator']==6
    monkeypatch.setattr(sys,'argv',['x','verify','--output',str(path),'--sha256',saved_sha]); m.main()
    monkeypatch.setattr(sys,'argv',['x','analyze','--output',str(path)])
    with pytest.raises(ValueError): m.main()
    path.write_text('{}')
    monkeypatch.setattr(sys,'argv',['x','verify','--output',str(path),'--sha256',saved_sha])
    with pytest.raises(ValueError): m.main()
