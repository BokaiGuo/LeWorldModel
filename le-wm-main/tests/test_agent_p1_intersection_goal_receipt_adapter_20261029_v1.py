import copy
import importlib.util
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p1_intersection_goal_adapter_20261029_v2 as v2

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('receipt_adapter',ROOT/'experiments/spectral_reachability/agent_p1_intersection_goal_receipt_adapter_20261029_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def items(tmp_path):
    prep=tmp_path/'prepare.json';report=tmp_path/'report.json';receipt=tmp_path/'receipt.json';ps='a'*64;ss='b'*64;rs='c'*64;cs='d'*64
    goals=[[0.,0.,0.,0.,0.,3.,4.],[1.,1.,1.,1.,0.,-1.,2.]]
    plans=[];slots=[];rows=[]
    for i,(root,cond,arm) in enumerate(v2.ORDER):
        goal=goals[root];state=goal[:];state[0]+=30
        plan=dict(index=i,root_slot=root,condition=cond,arm=arm,task_id='task'+str(root),selection_digest=str(i)*64)
        responses=[dict(step=j,state=state[:],native_success=False,terminated=False,reward=-30.) for j in range(1,16)]
        cps=[dict(step=j,state=state[:],snapshot=dict(state=state[:])) for j in m.CP]
        plans.append(plan);slots.append(dict(slot_id=f'{i}:{arm}',root_id=v2.ROOTS[root],goal7=goal[:],goal_authority_sha256=ps))
        rows.append(dict(index=i,root_slot=root,condition=cond,arm=arm,task_id=plan['task_id'],selection_digest=plan['selection_digest'],plan=copy.deepcopy(plan),status='horizon_completed',initial_exact=True,goal=goal[:],root_sha256=v2.ROOTS[root],root_state=state[:],own_returned=15,responses=responses,checkpoints=cps,missing_checkpoints=[],first_error=None,secondary_errors=[]))
    p=dict(plans=plans,planner_authority={'fixture':'only'});s=dict(slots=slots,checkpoints=list(m.CP),rgb_pairs=[])
    r=dict(schema=v2.PHYSICAL+'.consume',status='intersection6_native_recorded_not_admitted',fixed_denominator=6,known_roots=2,fresh_roots=0,inherited=0,physical_report_sha256=rs,plans=copy.deepcopy(plans),planner_authority=p['planner_authority'],bindings={str(prep):ps,str(v2.SOURCE):ss,str(report):rs},rows=rows)
    for flag in ('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness','equal_FLOPs','equal_total_compute','independent_confirmation','physical_release'):r[flag]=False
    return s,p,r,receipt,cs,prep,ps,ss,report,rs

def test_complete_24_points_and_cp0_origin(items):
    x=m.normalize(*items);assert x['denominator']==dict(slots=6,checkpoints=24)
    assert sum(len(r['points']) for r in x['receipt']['rows'])==24
    assert x['details'][0]['native_checkpoint_provenance'][0]['source']=='derived_from_bound_cp0_not_native_return'
    assert x['receipt']['rows'][0]['points'][1]['native_distance']==30

def test_early_stop_and_failure_preserve_missing(items):
    s,p,r,*rest=items;r['rows'][0].update(status='native_stopped',own_returned=5,responses=r['rows'][0]['responses'][:5],checkpoints=r['rows'][0]['checkpoints'][:2],missing_checkpoints=[10,15])
    r['rows'][1].update(status='engineering_failed',initial_exact=False,own_returned=0,responses=[],checkpoints=[],missing_checkpoints=list(m.CP),first_error={'kind':'fixture'})
    x=m.normalize(s,p,r,*rest)
    assert [q['state7'] is None for q in x['receipt']['rows'][0]['points']]==[False,False,True,True]
    assert all(q['state7'] is None for q in x['receipt']['rows'][1]['points'])
    assert x['details'][1]['first_error']=={'kind':'fixture'}

def test_checkpoint_without_response_does_not_fabricate(items):
    s,p,r,*rest=items;r['rows'][0]['responses']=r['rows'][0]['responses'][:4];r['rows'][0]['status']='engineering_failed'
    x=m.normalize(s,p,r,*rest);p5=x['receipt']['rows'][0]['points'][1]
    assert p5['state7'] is None and p5['missing_reason']=='checkpoint_observed_but_return_label_missing'

@pytest.mark.parametrize('kind',['denominator','goal','plan','row','checkpoint','snapshot','reward','response','source','report','admission'])
def test_tamper_rejected(items,kind):
    s,p,r,*rest=copy.deepcopy(items)
    if kind=='denominator':r['fixed_denominator']=4
    if kind=='goal':r['rows'][0]['goal'][0]+=1
    if kind=='plan':r['plans'][0]['task_id']='other'
    if kind=='row':r['rows'][0]['root_slot']=1
    if kind=='checkpoint':r['rows'][0]['missing_checkpoints']=[5]
    if kind=='snapshot':r['rows'][0]['checkpoints'][1]['snapshot']['state'][0]+=1
    if kind=='reward':r['rows'][0]['responses'][4]['reward']=1
    if kind=='response':r['rows'][0]['responses'][4]['state'][0]+=1
    if kind=='source':r['bindings'][str(v2.SOURCE)]='0'*64
    if kind=='report':r['physical_report_sha256']='0'*64
    if kind=='admission':r['P1_admission']=True
    with pytest.raises(ValueError):m.normalize(s,p,r,*rest)

def test_temporary_full_roundtrip(items,tmp_path):
    result=m.normalize(*items);raw=v2.encode(result);p=tmp_path/'result.json'
    with p.open('xb') as f:f.write(raw)
    assert p.read_bytes()==v2.encode(m.normalize(*items))
