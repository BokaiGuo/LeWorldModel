"""Read-only actual fixtures; mutations are in memory, no tmp/simulator/model."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p1_same_prefix_saved_audit_20261027_v1 as m


@pytest.fixture(scope='module')
def data():
    r=m.Reader();p,report,packets=m.load_inputs(r)
    return p,report,packets


def test_probe_exact_coordinate_census():
    ps=m.probes()
    assert len(ps)==41 and ps[0]['logits']==[0.]*10
    assert {(p['coordinate'],p['radius'],p['sign']) for p in ps[1:]}=={
        (i,r,s) for i in range(10) for r in (.25,.5) for s in (-1,1)}
    assert all(np.count_nonzero(p['logits'])==1 for p in ps[1:])


def test_actual246_and_native_responses(data):
    p,r,packets=data
    for plan,row in zip(p['inputs']['plans'],r['rows']): m.check_row(plan,row,packets[plan['root_sha256']])
    a,objects=m.responses(p['inputs']['plans'],r['rows'],{k:v['goal'] for k,v in packets.items()})
    assert a==json.loads((m.ACTUAL/'analysis.json').read_text())
    assert len(objects)==6 and all(len(x['object_responses'])==41 for x in objects)
    assert all(x['object_responses'][0]['object_position_error_change']==0. for x in objects)


@pytest.mark.parametrize('defect',['root','selection','command','order','terminated','partial','reward','native','object','checkpoint','velocity','missing_body','hidden_angle','goal','guard'])
def test_row_saved_tampering_rejected(data,defect):
    p,r,packets=data; plan=p['inputs']['plans'][0];row=deepcopy(r['rows'][0])
    if defect=='root':row['root_sha256']='bad'
    elif defect=='selection':row['selection_sha256']='bad'
    elif defect=='command':row['trace'][0]['command'][0]=.1
    elif defect=='order':row['trace'][0]['step']=2
    elif defect=='terminated':row['trace'][0]['terminated']=True
    elif defect=='partial':row['full_horizon']=False
    elif defect=='reward':row['trace'][0]['reward']+=1
    elif defect=='native':row['trace'][0]['native_official_evaluation']['success']=True
    elif defect=='object':row['trace'][0]['pose_diagnostics']['independent']['object_position_error']+=1
    elif defect=='checkpoint':row['checkpoints'][1]['state'][0]+=.1
    elif defect=='velocity':row['body_dynamics']['5']['agent']['velocity'][0]+=.1
    elif defect=='missing_body':del row['body_dynamics']['5']['block']['angular_velocity']
    elif defect=='hidden_angle':row['body_dynamics']['5']['block']['unwrapped_angle']+=.1
    elif defect=='goal':row['trace'][0]['official_info']['goal_state'][0]+=.1
    elif defect=='guard':row['guard_receipt']['observations'][1]['canonical_agent_pixels']=0
    with pytest.raises(ValueError):m.check_row(plan,row,packets[plan['root_sha256']])


@pytest.mark.parametrize('defect',['block_velocity','block_angular_velocity','force','sleep','rgb','partial','census'])
def test_exact_prefix_even_hidden_from_state7(data,defect):
    p,r,packets=data;rows=deepcopy(r['rows'])
    if defect=='block_velocity':rows[1]['body_dynamics']['5']['block']['velocity'][0]+=.1
    elif defect=='block_angular_velocity':rows[1]['body_dynamics']['5']['block']['angular_velocity']+=.1
    elif defect=='force':rows[1]['body_dynamics']['5']['block']['force'][0]+=.1
    elif defect=='sleep':rows[1]['body_dynamics']['5']['block']['is_sleeping']=True
    elif defect=='rgb':rows[1]['checkpoints'][1]['rgb_sha256']='bad'
    elif defect=='partial':rows[1]['full_horizon']=False
    elif defect=='census':rows.pop()
    with pytest.raises(ValueError):m.responses(p['inputs']['plans'],rows,{k:v['goal'] for k,v in packets.items()})


def test_native_negative_angle_not_silently_repaired():
    goal=[0.]*7;state=[0.,0.,0.,0.,4*np.pi,0.,0.]
    d=m.metrics(goal,state)
    assert d['native_formula']['negative_angle_term'] and d['native_formula']['success']
    assert d['independent']['wrapped_angle_error']<1e-12
    assert m.metrics([0.]*7,[21.,0.,0.,0.,0.,0.,0.])['native_formula']['success'] is False


def event_lines(values):
    previous=None;lines=[]
    for i,value in enumerate(values):
        v=dict(value,index=i,previous_sha256=previous);previous=m.digest(v)
        lines.append(m.core.canonical(dict(v,event_sha256=previous))+'\n')
    return lines


def test_events_exact_chain():
    e=m.Events(event_lines([dict(kind='one',value=1),dict(kind='two')]))
    e.exact('one',value=1);e.exact('two');e.finish()
    assert e.index==2


@pytest.mark.parametrize('defect',['hash','index','missing','extra','payload','duplicate'])
def test_event_faults(defect):
    lines=event_lines([dict(kind='one',value=1),dict(kind='two')])
    if defect=='hash':lines[0]=lines[0].replace('"value": 1','"value": 2')
    elif defect=='index':lines[0]=lines[0].replace('"index": 0','"index": 2')
    elif defect=='missing':lines.pop()
    elif defect=='extra':lines+=event_lines([dict(kind='extra')])
    elif defect=='payload':lines=event_lines([dict(kind='one',value=2),dict(kind='two')])
    elif defect=='duplicate':lines[1]=lines[0]
    e=m.Events(lines)
    with pytest.raises(ValueError):e.exact('one',value=1);e.exact('two');e.finish()


def test_actual_journal_exact_calls(data):
    p,r,packets=data
    with (m.ACTUAL/'journal.jsonl').open() as f:result=m.journal(f,p,r,packets)
    assert result['events']==34196 and result['counts']==m.COUNTS


def test_missing_or_extra_actual_call_rejected(data):
    p,r,packets=data;bad=deepcopy(r);bad['counts']['step']['completed']-=1
    with (m.ACTUAL/'journal.jsonl').open() as f,pytest.raises(ValueError,match='public calls'):
        m.journal(f,p,bad,packets)


@pytest.mark.parametrize('defect',['index','Q5','winner','cost','prefix'])
def test_role_or_plan_tamper(data,defect):
    p,_,_=data
    parent=json.loads((m.ROOT/'results/p1_witness_physical_20261027_v1_prepared01/prepare.json').read_text())
    wp=m.ROOT/'results/p1_two_stage_witness_20261027_v1_prepared01/protocol.json'
    task=json.loads(wp.read_text())['inputs']['tasks'][0]['task'];old=next(x for x in parent['inputs']['plans'] if x['arm']=='two_stage')
    saved=json.loads(Path(old['origin']).read_text())
    if defect=='index':saved['candidates'][0]['index']=2
    elif defect=='winner':saved['winner']=saved['candidates'][0]
    elif defect=='cost':saved['candidates'][0]['cost']+=1
    elif defect=='prefix':saved['candidates'][1]['commands'][0][0]+=.1
    if defect in ('index','winner','cost','prefix'):
        with pytest.raises(ValueError):m.model_roles(task,saved)
    else:
        roles=deepcopy(p['inputs']['roles']);roles[1]['model_Q5']+=1
        with pytest.raises(ValueError):m.check_roles(roles,p['inputs']['roles'])


def test_plan_genuine_probe_not_relabelled(data):
    p,_,_=data
    parent=json.loads((m.ROOT/'results/p1_witness_physical_20261027_v1_prepared01/prepare.json').read_text())
    old=next(x for x in parent['inputs']['plans'] if x['arm']=='two_stage')
    task=json.loads((m.ROOT/'results/p1_two_stage_witness_20261027_v1_prepared01/protocol.json').read_text())['inputs']['tasks'][0]['task']
    role={k:p['inputs']['roles'][0][k] for k in ('role','bank_index','prefix','model_Q5')}
    bad=deepcopy(p['inputs']['plans'][1]);bad['probe']['coordinate']=9
    with pytest.raises(ValueError):m.check_plan(bad,old,role,m.probes()[1],task)
