"""RAM-only regression and MAIN saved fault injection, no producer/model imports."""
import ast
from copy import deepcopy
import gzip
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_blind_planners_saved_audit_20261027_v1 as a

REPORT=a.io.ROOT/'results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json'
REPORT_SHA='4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d'
PREPARE_SHA='30b58155a59f73fdc8cc132ff069e645ed6c2bbff9434f474b1f7b9c412f7eb4'


@pytest.fixture(scope='module')
def actual():
    if not REPORT.exists(): pytest.skip('MAIN actual planner receipt not installed')
    r=a.read(REPORT,REPORT_SHA); p=a.read(r['prepared'],PREPARE_SHA)
    wrapped=[a.io.parse(line) for line in a.load(REPORT.parent/'events.jsonl.gz').splitlines()]
    values={}
    for arm in a.ARMS:
        row=next(row for row in r['rows'] if row['order']==0 and row['arm']==arm)
        values[arm]=(a.export_task(p['slots'][0]['task_export'],p['slots'][0]['task_sha256']),
                     a.read(REPORT.parent/row['result_file']),
                     [e['event'] for e in wrapped if e['root_order']==0 and e['arm']==arm])
    return r,p,values


def test_no_planner_validation_imports():
    tree=ast.parse(Path(a.__file__).read_text())
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [v.name for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names]
    assert not any(any(word in name for word in ('torch','encode_cuda','planners_cuda','line_search_gn_cuda','consistent_compiler')) for name in imports)


def test_zero_token_nonzero_and_time_major():
    t=dict(action_mean=[-.0087,.0068],action_std=[.2019,.2002])
    got=a.tokens(t,np.zeros((10,2)))
    assert got.shape==(2,10) and got.any()
    raw=np.arange(20).reshape(10,2)/100
    want=((raw.astype(np.float32)-np.asarray(t['action_mean'],np.float32))/np.asarray(t['action_std'],np.float32)).reshape(2,10)
    assert np.array_equal(a.tokens(t,raw),want)


def test_project_feasible_float32():
    p=a.project(np.full((2,10),100.))
    assert np.array_equal(p,p.astype(np.float32))
    assert (np.linalg.norm(p,axis=1)<=a.CAP).all()


def test_ridge_ball_and_residual_independent():
    k=np.zeros((192,10)); k[:10]=np.eye(10); r=np.ones(192)*10
    x,value,s,lam=a.ridge(k,r)
    assert np.linalg.norm(x)<=a.CAP+1e-14 and lam>0
    assert value==float((np.sum((r-k@x)**2)+.002*np.sum(x*x))/192)


def test_unequal_and_forward_not_added_twice():
    assert sum(a.BUDGETS[arm]['predict'] for arm in a.ARMS)==144
    assert a.BUDGETS[a.ARMS[0]]['predict']==78
    assert a.BUDGETS[a.ARMS[1]]['predict']==66
    assert a.API[a.ARMS[1]]==dict(grad=11,jacobian=11)


def test_lane_nested():
    def event(kind,index,completed,resource): return dict(kind=kind,entry=dict(index=index,completed=completed,resource=resource))
    events=[event('a',0,False,'jacobian'),event('a',1,False,'grad'),event('b',1,True,'grad'),event('b',0,True,'jacobian')]
    a.lane(events,'a','b','entry',dict(jacobian=1,grad=1))
    events[2],events[3]=events[3],events[2]
    with pytest.raises(ValueError): a.lane(events,'a','b','entry',dict(jacobian=1,grad=1))


@pytest.mark.parametrize('mutation',['missing','duplicate','wrongreturn','unknown'])
def test_lane_rejects(mutation):
    events=[dict(kind='a',entry=dict(index=0,resource='predict',completed=False)),
            dict(kind='b',entry=dict(index=0,resource='predict',completed=True))]
    if mutation=='missing': events.pop()
    if mutation=='duplicate': events.append(events[-1])
    if mutation=='wrongreturn': events[1]['entry']['resource']='action_encoder'
    if mutation=='unknown': events[0]['entry']['resource']=events[1]['entry']['resource']='unbudgeted'
    with pytest.raises(ValueError): a.lane(events,'a','b','entry',dict(predict=1))


def test_actual_whole_saved_audit(actual):
    result=a.audit(REPORT,REPORT_SHA,PREPARE_SHA)
    assert result['status']=='saved_model_plans_consistent_not_admitted'
    assert len(result['rows'])==4 and result['event_count']==9664
    assert result['resources'][a.ARMS[0]]['predict']['completed']==156
    assert result['resources'][a.ARMS[1]]['predict']['completed']==132
    assert result['new_neural']==result['new_physics']==0
    assert result['P1_admission'] is False and result['equal_budget_claim'] is False


@pytest.mark.parametrize('mutation',['step','scale','zero_padding','task_id','jacrow','chain','gradient','delta','cost','drop_evaluation','reorder'])
def test_gn_saved_mutation(actual,mutation):
    t,result,events=deepcopy(actual[2][a.ARMS[0]])
    if mutation=='step': result['steps'][0]['selected_index']=999
    if mutation=='scale': next(e for e in events if e['kind']=='line_proposals_frozen')['scales'][0]=.9
    if mutation=='zero_padding': next(e for e in events if e['kind']=='evaluation_complete')['prediction']['contexts'][0]['action_history']=np.zeros((2,10)).tolist()
    if mutation=='task_id': result['task_id']='0'*64
    if mutation=='jacrow': next(e for e in events if e['kind']=='jacobian_row')['row']=1
    if mutation=='chain': next(e for e in events if e['kind']=='jacobian_row')['raw_control_derivative'][0]+=1
    if mutation=='gradient': next(e for e in events if e['kind']=='derivative_complete')['derivative']['objective_gradient'][0]+=1
    if mutation=='delta': next(e for e in events if e['kind']=='line_proposals_frozen')['delta'][0]+=1
    if mutation=='cost': next(e for e in events if e['kind']=='evaluation_complete')['prediction']['cost']+=1
    if mutation=='drop_evaluation': events.remove(next(e for e in events if e['kind']=='evaluation_complete'))
    if mutation=='reorder':
        i=next(i for i,e in enumerate(events) if e['kind']=='line_proposals_frozen'); j=next(i for i,e in enumerate(events) if e['kind']=='selection_complete')
        events[i],events[j]=events[j],events[i]
    with pytest.raises(ValueError): a.gn_audit(t,result,events)


@pytest.mark.parametrize('mutation',['candidate','objective','stage','suffix','Q','bank','derivative','winner','command','context','replay','operations'])
def test_d2p_saved_mutation(actual,mutation):
    t,result,events=deepcopy(actual[2][a.ARMS[1]])
    if mutation=='candidate': result['candidates'].pop()
    if mutation=='objective': result['candidates'][0]['objective']+=1
    if mutation=='stage': result['candidates'][0]['stage_cost']+=1
    if mutation=='suffix': result['candidates'][0]['suffix_params'][0]+=.1
    if mutation=='Q': result['q_sha256']='0'*64
    if mutation=='bank': result['bank']['params'][1][0]+=1
    if mutation=='derivative': next(e for e in events if e['kind']=='derivative')['jacobian'][0][0]+=1
    if mutation=='winner': result['winner']=result['candidates'][(result['winner']['index']+1)%10]
    if mutation=='command': result['candidates'][0]['commands'][0][0]+=.1
    if mutation=='context': result['candidates'][0]['context']['action_history'][0][0]+=.1
    if mutation=='replay': next(e for e in events if e['kind']=='replay' and e['completed'])['context']['latent_history'][0][0]+=1
    if mutation=='operations': result['operations'][0]['mathematical_vjp_directions']=1
    with pytest.raises(ValueError): a.d2p_audit(t,result,events)


@pytest.mark.parametrize('arm',a.ARMS)
def test_actual_ledgers_and_missing_return(actual,arm):
    _,_,events=deepcopy(actual[2][arm]); a.ledger_audit(events,arm)
    events.remove(next(e for e in events if e['kind']=='AD_API_completed'))
    with pytest.raises(ValueError): a.ledger_audit(events,arm)


def test_wrong_external_sha(actual):
    with pytest.raises(ValueError,match='SHA'): a.audit(REPORT,'0'*64,PREPARE_SHA)


def test_partial_report_never_releases(actual,monkeypatch):
    r=deepcopy(actual[0]); r['status']='partial_or_failed'
    original=a.read
    monkeypatch.setattr(a,'read',lambda path,expected=None: r if Path(path)==REPORT else original(path,expected))
    with pytest.raises(ValueError,match='partial'): a.audit(REPORT,REPORT_SHA,PREPARE_SHA)
