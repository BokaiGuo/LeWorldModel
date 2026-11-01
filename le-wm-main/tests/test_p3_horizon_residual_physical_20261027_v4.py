"""Saved-array and fake-environment tests only, no neural or real physics."""
from copy import deepcopy
from pathlib import Path
import builtins
import numpy as np
import pytest
from experiments.spectral_reachability import p3_horizon_residual_physical_20261027_v4 as p

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/p3_horizon_residual_budget_20261027_v4_cuda_main01'
REPORT_SHA='5ce206a88de000141add48e58c4900d8b66e55ae5ee1ec6941bd72110180b57f'


class Env:
    def __init__(self,missing=None,fail=None,renderfail=None,terminate=None):
        self.n=0;self.missing=missing;self.fail=fail;self.renderfail=renderfail;self.terminate=terminate
    def render(self):
        if self.n==self.renderfail:raise RuntimeError('render fixture')
        rgb=np.zeros((224,224,3),np.uint8)
        if self.n!=self.missing:rgb[20:22,20:22]=[65,105,225]
        return rgb
    def step(self,action):
        self.n+=1
        if self.n==self.fail:raise RuntimeError('step fixture')
        return None,0.,self.n==self.terminate,False,{}


def fixture_plan(h=2,root=0,method='uniform'):
    plan=dict(schema=p.NAME+'.plan',root_index=root,root_sha256=str(root)*64,method=method,
        planned_horizon_tokens=h,planned_commands=5*h,corrected_search_horizon=3,commands=np.zeros((5*h,2)).tolist(),
        goal=np.zeros(192).tolist(),projection=np.eye(192).tolist(),goal_vector_sha256='goal',projection_sha256='Q')
    plan['selection_sha256']=p.planning.digest(plan);return plan


def branch(tmp_path,h=2,env=None,reset=lambda:dict(context='same'),plan=None):
    plan=plan or fixture_plan(h)
    return p.physical_branch(plan=plan,output=tmp_path,reset=reset,env=env or Env(),state_of=lambda e:np.array([e.n]*7,float),
        evaluate=lambda s:dict(official={'official_state_distance':float(s[0])},pose={'pose_cost':float(s[0])}))


@pytest.mark.parametrize('h',[2,3])
def test_full_exact_horizon(tmp_path,h):
    row=branch(tmp_path,h);assert p.verify_branch(fixture_plan(h),row)
    assert row['full_horizon_completed'] and row['completed_commands']==5*h
    assert row['endpoint_scope']==f'full_H{h}' and row['endpoint_fresh']
    assert len(list(tmp_path.glob('*.png')))==5*h+1


@pytest.mark.parametrize('h',[2,3])
@pytest.mark.parametrize('kwargs,attempts,returns,fresh',[
    ({'missing':0},0,0,True),({'missing':6},6,6,True),({'fail':7},7,6,False),
    ({'renderfail':7},7,7,False),({'terminate':7},7,7,True)])
def test_partials_never_promoted(tmp_path,h,kwargs,attempts,returns,fresh):
    row=branch(tmp_path,h,Env(**kwargs));assert p.verify_branch(fixture_plan(h),row)
    assert row['physical_env_step_attempts']==attempts and row['physical_env_step_returns']==returns
    assert not row['full_horizon_completed'] and row['endpoint_fresh']==fresh
    assert row['endpoint_scope'] not in ('full_H2','full_H3')


def test_reset_error_durable(tmp_path):
    def fail():raise ValueError('root/goal mismatch fixture')
    row=branch(tmp_path,reset=fail)
    assert row['status']=='branch_failed' and not row['endpoint_fresh']
    assert len(list(tmp_path.glob('*.physical.json')))==1


@pytest.mark.parametrize('damage',['count','action','horizon','hash'])
def test_invalid_plan_before_reset(tmp_path,damage):
    plan=fixture_plan()
    if damage=='count':plan['planned_commands']=15
    if damage=='action':plan['commands'][0][0]=float('nan')
    if damage=='horizon':plan['planned_horizon_tokens']=True
    if damage=='hash':plan['selection_sha256']='0'*64
    def forbidden():raise AssertionError('must reject before reset')
    with pytest.raises(ValueError):branch(tmp_path,plan=plan,reset=forbidden)


def test_journal_tamper_rejected(tmp_path):
    row=branch(tmp_path)
    with Path(row['journal_path']).open('a') as stream:stream.write('{}\n')
    with pytest.raises(ValueError):p.verify_branch(fixture_plan(),row)


def test_durable_before_any_encoding_setup(tmp_path):
    with pytest.raises(ValueError):p.optional_encodings([],[],tmp_path,lambda:pytest.fail('early model load'))


def test_exact_normalized_warmup_tokens():
    history=np.asarray([[0.,0.],[.1,-.1],[.2,.1]],dtype=np.float32)
    packet=np.repeat(history[:,None,:],5,axis=1).reshape(3,10)
    expected=((packet-np.tile(np.asarray(p.NORMALIZATION['mean'],np.float32),5))/np.tile(np.asarray(p.NORMALIZATION['std'],np.float32),5))[1:]
    receipt=p.validate_reset_history(history,expected.tolist())
    assert receipt['normalized_between_history']==expected.tolist()
    damaged=expected.copy();damaged[0,0]=np.nextafter(damaged[0,0],np.float32(100))
    with pytest.raises(ValueError):p.validate_reset_history(history,damaged)
    with pytest.raises(ValueError):p.validate_reset_history(history[:2],expected)
    with pytest.raises(ValueError):p.validate_reset_history(history+2,expected)


@pytest.mark.parametrize('damage',[None,'foreign','duplicate','goalshape','nan'])
def test_all_root_goal_bindings_upfront(damage):
    roots=[dict(state=np.full(7,i).tolist(),goal=np.full(7,i+1).tolist()) for i in range(4)]
    fingerprint=lambda r:str(int(r['state'][0]))
    refs={str(i):{} for i in range(4)}
    if damage=='foreign':roots[3]['state'][0]=9
    if damage=='duplicate':roots[3]=deepcopy(roots[0])
    if damage=='goalshape':roots[3]['goal'].pop()
    if damage=='nan':roots[3]['goal'][0]=float('nan')
    if damage:
        with pytest.raises(ValueError):p.validate_roots(roots,refs,fingerprint)
    else:assert len(p.validate_roots(roots,refs,fingerprint))==4


@pytest.mark.parametrize('kwargs,attempts,status',[({},1,'completed'),({'missing':6},1,'completed'),({'fail':7},0,'blocked_stale_endpoint'),({'renderfail':7},0,'blocked_stale_endpoint')])
def test_endpoint_encoding_scope_and_stale_guard(tmp_path,kwargs,attempts,status):
    row=branch(tmp_path,env=Env(**kwargs));p.write(tmp_path/'physical_report.json',dict(rows=[row]))
    before=next(tmp_path.glob('*.physical.json')).read_bytes()
    def setup():
        assert (tmp_path/'physical_report.json').is_file()
        assert next(tmp_path.glob('*.physical.json')).read_bytes()==before
        return lambda path:[0.]*192
    receipts=p.optional_encodings([fixture_plan()],[row],tmp_path,setup)
    assert receipts[0]['attempted_encode_samples']==attempts and receipts[0]['status']==status
    if attempts:assert receipts[0]['original_Q_cost']==0 and receipts[0]['endpoint_scope']==row['endpoint_scope']


@pytest.mark.parametrize('stage',['setup','encode','shape','RGB'])
def test_encoding_failures_durable(tmp_path,stage):
    row=branch(tmp_path);p.write(tmp_path/'physical_report.json',dict(rows=[row]))
    if stage=='RGB':Path(row['terminal_rgb_path']).write_bytes(b'broken')
    def fail():raise RuntimeError('injected setup')
    def encode(path):
        if stage=='encode':raise RuntimeError('injected forward')
        return [0.]*3
    receipts=p.optional_encodings([fixture_plan()],[row],tmp_path,fail if stage=='setup' else lambda:encode)
    assert receipts[0]['status'] in ('encoding_setup_failed','encoding_failed')
    assert len(list(tmp_path.glob('*.encoding.json')))==1
    assert (tmp_path/'encoding.events.jsonl').stat().st_size>0


def pair_fixture(horizons=(2,3),partial=False):
    plans=[];rows=[]
    for root in range(4):
        for mode,h in zip(p.METHODS,horizons):
            plan=fixture_plan(h,root,mode);plans.append(plan)
            rows.append(dict(root_sha256=plan['root_sha256'],method=mode,selection_sha256=plan['selection_sha256'],
                planned_horizon_tokens=h,planned_commands=5*h,context_identity={'same':'context'},
                full_horizon_completed=not partial,endpoint_fresh=True,physical_env_step_returns=6 if partial else 5*h,
                pose={'pose_cost':1 if mode=='uniform' else 2}))
    return plans,rows


@pytest.mark.parametrize('horizons,partial,label',[( (2,2),False,'full_H2_pair'),((3,3),False,'full_H3_pair'),
    ((2,3),False,'full_mixed_H2_H3'),((2,3),True,'partial_same_steps')])
def test_strata_no_cross_horizon_pool(horizons,partial,label):
    pairs=p.comparisons(*pair_fixture(horizons,partial))
    assert all(pair['classification']==label for pair in pairs)
    assert all((pair['dynamic_minus_uniform_pose_cost'] is None)==(partial or horizons[0]!=horizons[1]) for pair in pairs)


def test_missing_context_and_stale_never_full():
    plans,rows=pair_fixture();rows[0].pop('context_identity');rows[2]['endpoint_fresh']=False
    pairs=p.comparisons(plans,rows)
    assert pairs[0]['classification']==pairs[1]['classification']=='partial_or_unmatched'


def test_all_eight_missing_slots(tmp_path):
    plans,_=pair_fixture();rows=[];p.materialize_missing(plans,rows,tmp_path)
    assert len(rows)==8 and all(r['status']=='blocked' and not r['endpoint_fresh'] for r in rows)
    assert sum(r['physical_env_step_attempts'] for r in rows)==0


@pytest.fixture(scope='module')
def source_exports():
    body,identity=p.planning.validate_prepared(ROOT/p.PLANNING_PREPARED,p.PLANNING_SHA)
    return dict(roots=body['roots'],packets=body['packets'],arm_order=body['arm_order'],planner_protocol_id=identity,planning_report_sha256=REPORT_SHA),p.planning.selected_actions(RUN)


def test_actual_eight_exported_plans(source_exports):
    protocol,exports=source_exports;plans=p.plans_from_exports(protocol,exports)
    assert len(plans)==8 and [r['method'] for r in plans]==[m for order in protocol['arm_order'] for m in order]
    assert sum(x['planned_commands'] for x in plans)==sum(5*x['horizon'] for x in exports)
    for plan in plans:p.validate_plan(plan)


@pytest.mark.parametrize('damage',['missing','duplicate','goal','Q','context','action','report','root','hash','horizon'])
def test_export_mutations_rejected(source_exports,damage):
    protocol,exports=deepcopy(source_exports);e=exports[0]
    if damage=='missing':exports.pop()
    elif damage=='duplicate':exports[1]=deepcopy(e)
    elif damage=='goal':e['goal'][0]+=.1
    elif damage=='Q':e['projection'][0][0]+=.1
    elif damage=='context':e['root_context']['latent_history'][0][0]+=.1
    elif damage=='action':e['actions'][0][0]+=2
    elif damage=='report':e['report_sha256']='0'*64
    elif damage=='root':e['root_sha256']='0'*64
    elif damage=='hash':e['selection_sha256']='0'*64
    else:e['horizon']=3
    with pytest.raises(ValueError):p.plans_from_exports(protocol,exports)


def test_real_two_stage_prepare_bind_verify_no_runtime(tmp_path,monkeypatch):
    original=builtins.__import__
    def no_runtime(name,*a,**kw):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    source=tmp_path/'source';bound=tmp_path/'bound'
    receipt=p.prepare(ROOT,source);source_sha=p.durable.digest(source/'prepare.json')
    assert receipt['status']=='source_prepared_awaiting_report'
    with pytest.raises(ValueError):p.execute_physics(source,source_sha,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()
    p.bind_report(source,source_sha,RUN,REPORT_SHA,bound)
    protocol=p.validate_prepared(bound,p.durable.digest(bound/'prepare.json'))
    assert protocol['stage']=='report_bound_ready' and len(protocol['plans'])==8
    assert receipt['new_predict_calls']==receipt['new_physical_steps']==0
    with pytest.raises(ValueError):p.prepare(ROOT,source)
    with pytest.raises(ValueError):p.bind_report(source,source_sha,RUN,'0'*64,tmp_path/'bad')
    with pytest.raises(ValueError):p.validate_prepared(bound,'0'*64)
