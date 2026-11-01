"""RAM/saved-schema regressions only; never new CUDA/native execution evidence."""
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_paired_quota_physical_handoff_20261028_v1 as m


@pytest.fixture(scope='module')
def saved():
    path=m.ROOT/'results/agent_p3_paired_quota_runtime_20261028_v1_prepare_MAIN01.json'
    assert m.sha(path)=='46e191695e6e121324daf691bafe1b2862a36349e11232822957d8ca6785e6d8'
    p=m.READ(path,64*1024**2)
    oldr=m.READ(m.saved.ACTUAL_OUTPUT/'report.json',m.physical.FINAL)
    assert m.sha(m.saved.ACTUAL_OUTPUT/'report.json')==m.saved.ACTUAL_REPORT_SHA
    oldp=m.READ(oldr['prepared'],64*1024**2)
    oldhost=m.READ(m.ROOT/'results/agent_p3_child_budget_host_cuda_20261028_v1_cuda_MAIN01/report.json',256*1024**2)
    rows=m.runtime.initial_rows(p)
    for i in range(24):
        if p['wiring']['children'][i]['status']=='not_reached':continue
        quota=[m.runtime.compiler.quota(r['decision']) for r in oldhost['session']['rows'][2*i:2*i+2]]
        allowed=all(q['allowed'] for q in quota)
        for j in range(2):
            k=2*i+j;row=rows[k];old=oldhost['session']['rows'][k]
            row.update(signal=old['signal'],signal_resources=old['signal_resources'],decision=old['decision'],
                execution_quota=quota[j],pair_seal_sha256=m.digest(['RAM_pairseal',i]),
                status='compiled' if allowed else 'paired_paid_refusal')
            if allowed:
                branch=m.runtime.pair.branch_for(p['wiring']['children'][i],row['mode'],row)
                row['compiler']=dict(status=m.runtime.compiler.FINAL,error=None,
                    selected=deepcopy(oldp['rows'][k]['selected']),branch_sha256=m.digest(branch))
    # Projection fixture ONLY; deliberately not an actual new runtime report or strict authority.
    r=dict(session=dict(rows=rows));strict=dict(complete=True,rows=oldr['rows'],resources=oldr['resources'])
    return p,r,oldp,oldr,strict,m.sha(oldp['source_manifest'])


def projected(saved):return m.project(*saved[:5],'strict_saved',saved[5])


def test_full48_exact_reuse_no_duplicated_old_physics(saved):
    p,r,oldp,oldr,strict,manifest=saved;out=projected(saved)
    assert len(out['rows'])==48 and out['census']==dict(pending_new_physics=0,reused_saved=36,paid_refused=10,not_reached=2)
    assert out['new_budget']['execute_slots']==0 and all(v==0 for v in out['new_budget']['caps'].values())
    assert out['new_budget']['new_actual_resources'] is None
    assert out['historical_resources']['resources']==oldr['resources']
    for row in out['rows']:
        if row['route']=='reused_saved':
            i=row['ordinal'];e=row['reuse_evidence']
            assert e['fresh_execution'] is False and e['new_physics_calls']==0
            assert e['full_old_row']==oldr['rows'][i] and e['original_plan_sha256']==m.digest(oldp['rows'][i])
        else:assert row['runner_plan']['selected'] is None
    assert all(out[k] is False for k in m.FLAGS)


def test_one_raw_float32_bit_changed_pending_not_reused(saved):
    p,r,oldp,oldr,strict,manifest=saved;r=deepcopy(r)
    a=m.raw(r['session']['rows'][0]['compiler']['selected']['raw5'],5)
    a[0,0]=np.nextafter(a[0,0],np.float32(0 if a[0,0] else 1),dtype=np.float32)
    r['session']['rows'][0]['compiler']['selected']['raw5']=a.tolist()
    out=m.project(p,r,oldp,oldr,strict,'strict_saved',manifest)
    assert out['census']==dict(pending_new_physics=1,reused_saved=35,paid_refused=10,not_reached=2)
    row=out['rows'][0];assert row['reuse_evidence'] is None and row['runner_plan']['route']=='execute'
    assert out['new_budget']['caps']==dict(constructor=1,close=1,reset=2,set_state=4,goal_render=2,goal_setter=3,
        goal_bind=1,public_step=20,physics_step=204,explicit_render=22)


def test_signed_zero_raw_identity_not_numeric_equality():
    a=np.zeros((5,2),np.float32);b=a.copy();b[0,0]=-0.
    assert np.array_equal(a,b) and m.raw_identity(a.tolist(),5)!=m.raw_identity(b.tolist(),5)


@pytest.mark.parametrize('fault',['drop','mode','context','source','history','goal','snapshot','old_ordinal','no_strict','partial','policy'])
def test_identity_missing_authority_and_cross_child_rejected(saved,fault):
    p,r,op,orr,strict,manifest=saved;policy='strict_saved'
    if fault=='drop':r=deepcopy(r);r['session']['rows'].pop()
    elif fault=='mode':r=deepcopy(r);r['session']['rows'][0]['mode']='adaptive'
    elif fault in ('context','source','history'):
        p=deepcopy(p);c=p['wiring']['children'][0]
        if fault=='context':c['context_sha256']='0'*64
        elif fault=='source':c['ancestry']['source_report_sha256']='0'*64
        else:c['raw_history10'][0][0]=float(np.nextafter(np.float32(c['raw_history10'][0][0]),np.float32(1),dtype=np.float32))
    elif fault in ('goal','snapshot','old_ordinal'):
        orr=deepcopy(orr);strict=dict(strict,rows=orr['rows']);row=orr['rows'][0]
        if fault=='goal':row['boundary']['goal'][0]+=1
        elif fault=='snapshot':row['boundary']['snapshot']['bodies']['block']['unwrapped_angle']+=2*np.pi
        else:row['ordinal']=1
    elif fault=='no_strict':strict=None
    elif fault=='partial':strict=dict(strict,complete=False)
    else:policy='none'
    with pytest.raises((ValueError,KeyError)):m.project(p,r,op,orr,strict,policy,manifest)


def test_route_selection_does_not_use_reward_or_success(saved):
    p,r,op,orr,strict,manifest=saved;a=projected(saved)
    altered=dict(orr,rows=[dict(row,final_metrics={'RAM_only_changed_outcome':True}) for row in orr['rows']])
    b=m.project(p,r,op,altered,dict(strict,rows=altered['rows']),'strict_saved',manifest)
    assert [x['route'] for x in a['rows']]==[x['route'] for x in b['rows']]


def test_reused_outcomes_earlystop_missing_preserved_zero_new_resources(saved):
    out=projected(saved);p=dict(rows=[r['runner_plan'] for r in out['rows']],handoff=out)
    p['profile']=m.profile(out)
    report=dict(rows=[m.new_row(i,plan) for i,plan in enumerate(p['rows'])],profile=p['profile'],fixed_denominator=48,
        child_denominator=24,resources={k:dict(attempted=0,completed=0) for k in p['profile']['caps']},pending=[],
        errors=[],secondary_errors=[],status='recorded',wall_seconds=0.,**m.FLAGS)
    events=[dict(index=0,kind='run_started',profile=p['profile'])]
    native=m.validate_saved(p,report,events,m.saved.ACTUAL_OUTPUT)
    assert native['complete'] and native['routes']==dict(execute=0,reused_saved=36,paid_refused=10,not_reached=2)
    rows=m.results(p,report);assert len(rows)==48
    for row in rows:
        if row['route'] in ('paid_refused','not_reached'):
            assert row['physical_result'] is None and row['missing_reason']==row['route']
        else:
            assert not row['fresh_execution'] and row['physical_result']==saved[3]['rows'][row['ordinal']]
            n=row['physical_result']['own_returned'];assert 1<=n<=5


@pytest.fixture(scope='module')
def native_boundary(saved):
    # Saved-only regression for the original actual native branch under the new
    # generic validator. This fixture is NOT a newly executed paired trajectory.
    out=projected(saved);plans=[deepcopy(r['runner_plan']) for r in out['rows']]
    plans[0]['route']='execute'
    assert plans[0]==saved[2]['rows'][0]
    b=deepcopy(out);b['rows'][0]['route']='pending_new_physics'
    b['new_budget']=m.new_budget(b['rows']);p=dict(rows=plans,handoff=b,profile=m.profile(b))
    eid=saved[3]['rows'][0]['episode_id']
    es=m.physical.journal(m.saved.ACTUAL_OUTPUT/'journal.jsonl',m.physical.JOURNAL_CAP)
    es=[dict(e) for e in es if e['kind']=='run_started' or e.get('episode_id')==eid]
    es[0]=dict(es[0],profile=p['profile'])
    report=deepcopy(saved[3]);report.update(profile=p['profile'],rows=[m.new_row(i,plan) for i,plan in enumerate(plans)],**m.FLAGS)
    report['rows'][0]=deepcopy(saved[3]['rows'][0]);counts,pending=m.ledger(es,p['profile']['caps'])
    report.update(resources=counts,pending=pending)
    return p,report,es,m.saved.ACTUAL_OUTPUT


def test_saved_actual_native_complete_full_PNG_and_census(native_boundary):
    p,r,es,d=native_boundary;result=m.validate_saved(p,r,es,d)
    assert result['complete'] and r['resources']['reset']['completed']==2 and r['resources']['set_state']['completed']==4
    assert r['resources']['public_step']['completed']==10+p['rows'][0]['child']['boundary']+r['rows'][0]['own_returned']
    assert r['resources']['physics_step']['completed']==4+10*r['resources']['public_step']['completed']


@pytest.mark.parametrize('fault',['raw','context','angle2pi','goal','metric','padding','resources','gate','unknown','foreign','guard'])
def test_saved_native_negative_regressions(native_boundary,fault):
    p,r,es,d=native_boundary;p=deepcopy(p);r=deepcopy(r);es=deepcopy(es)
    if fault=='raw':next(e for e in es if e['kind']=='step_return')['command'][0]=0.
    elif fault=='context':next(e for e in es if e['kind']=='child_boundary_sealed')['context_sha256']='0'*64
    elif fault=='angle2pi':next(e for e in es if e['kind']=='fresh_observation')['receipt']['snapshot']['bodies']['block']['unwrapped_angle']+=2*np.pi
    elif fault=='goal':next(e for e in es if e['kind']=='fresh_observation')['receipt']['goal'][0]+=1
    elif fault=='metric':r['rows'][0]['final_metrics']['native7_reward']+=1
    elif fault=='padding':r['rows'][0]['own_returned']=6
    elif fault=='resources':r['resources']['reset']['completed']+=1
    elif fault=='gate':r['P3_admission']=True
    elif fault=='unknown':es.append(dict(kind='outcome_override',index=len(es)))
    elif fault=='foreign':es[-1]['episode_id']='foreign'
    else:es.pop(next(i for i,e in enumerate(es) if e['kind']=='guard_passed'))
    with pytest.raises((ValueError,KeyError)):m.validate_saved(p,r,es,d)


def test_native7_reward_not_joint_task_success():
    goal=np.zeros(7);state=goal.copy();state[5]=1000
    assert m.metric(goal,state)['native_success'] and m.metric(goal,state)['native7_reward']==-1000
    state[2]=21;assert not m.metric(goal,state)['native_success']


def test_missing_future_authority_rejected_before_any_consume(tmp_path,monkeypatch):
    calls=[];runtime=SimpleNamespace(**vars(m.runtime));runtime.consume=lambda *a:calls.append('BAD')
    monkeypatch.setattr(m,'runtime',runtime)
    a=m.inputs(str(m.RUNTIME_REPORT),m.RUNTIME_REPORT_SHA,str(tmp_path/'receipt.json'),'1'*64)
    with pytest.raises(ValueError,match='actual external'):m.preflight(a)
    assert calls==[]


def test_full_consumers_explicit_no_new_neural_source():
    source=inspect.getsource(m)
    assert 'actual=runtime.consume(' in source and "bridge.load_physical('strict_saved')" in source
    assert 'import torch' not in source and 'load_compatible_pretrained' not in source
    assert m.execute is m.physical.execute and m.Store is m.physical.Store
    m.contract.check_sources(m.PINS)


def test_run_default_refused_before_verify(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('verify before auth'))
    with pytest.raises(ValueError,match='authorization'):m.run('/RAM','0'*64,tmp_path/'out')


def test_CLI_exact_entrypoints():
    p=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'--help'],cwd=m.ROOT,capture_output=True,text=True)
    assert p.returncode==0 and all(x in p.stdout for x in ('prepare','verify','run','consume'))


def test_exclusive_NEW_paths_only(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir();p=tmp_path/'results'/(m.NAME+'_RAM.json')
    assert m.exclusive(str(p))==p
    m.physical.write_new(p,{'RAM_only':True})
    with pytest.raises(ValueError):m.exclusive(str(p))
    with pytest.raises(ValueError):m.exclusive('relative')


def test_real_terminal_paired_report_ABI_only_no_consumer():
    assert m.sha(m.RUNTIME_REPORT)==m.RUNTIME_REPORT_SHA
    r=m.READ(m.RUNTIME_REPORT,256*1024**2)
    assert r['schema']==m.runtime.NAME+'.report' and r['errors']==[] and r['status']=='complete_native_panel_not_admitted'
    assert r['prepare_sha256']=='46e191695e6e121324daf691bafe1b2862a36349e11232822957d8ca6785e6d8'
    assert len(r['session']['rows'])==48 and r['session']['gate']['cursor']==24
    for i in range(24):
        a,b=r['session']['rows'][2*i:2*i+2]
        assert a['status']==b['status'] and a['status'] in ('compiled','paired_paid_refusal','not_reached')
        for row in (a,b):
            if row['status']=='compiled':assert m.raw(row['compiler']['selected']['raw5'],5).shape==(5,2)
    # A terminal report and this schema smoke are NOT the awaited full strict receipt.


@pytest.mark.parametrize('fault',['producer_error','summary'])
def test_full_producer_error_and_summary_receipt_fail_before_old_consumer(tmp_path,monkeypatch,fault):
    receipt=tmp_path/'RAM_not_MAIN_receipt.json';m.physical.write_new(receipt,{})
    calls=[];runtime=SimpleNamespace(**vars(m.runtime))
    def consume(*a):
        calls.append('full_producer')
        if fault=='producer_error':raise ValueError('RAM producer refused')
        return {'RAM_full_not_a_valid_MAIN_receipt':True}
    runtime.consume=consume;monkeypatch.setattr(m,'runtime',runtime)
    monkeypatch.setattr(m,'load_physical',lambda:pytest.fail('old consumer before full paired strict'))
    authority=m.inputs(str(m.RUNTIME_REPORT),m.RUNTIME_REPORT_SHA,str(receipt),m.sha(receipt))
    with pytest.raises(ValueError):m.load_runtime(authority)
    assert calls==['full_producer']


def ram_physical_input(saved):
    h=projected(saved)
    return dict(rows=[deepcopy(r['runner_plan']) for r in h['rows']],handoff=h,profile=m.profile(h),
        sources={},source_manifest='/RAM/manifest.json',authority={'RAM_only':True},runtime_actual_resources={})


def ram_source(monkeypatch,native_factory):
    alias=SimpleNamespace(**vars(m.old))
    alias.read=lambda *a:dict(sourcepack='RAM',source_prepared='RAM',source_prepare_sha256='0'*64)
    alias.kernel=SimpleNamespace(load_source=lambda *a:dict(references=[None,None],plan=None))
    alias.native_factory=native_factory
    monkeypatch.setattr(m,'old',alias) # owned-module reference only; never patch frozen globals


def test_zero_new_native_full_run_and_strict_saved_consumer(saved,tmp_path,monkeypatch):
    p=ram_physical_input(saved)
    def forbidden(*a):pytest.fail('all exact reuses must not construct native environment')
    ram_source(monkeypatch,forbidden)
    output=tmp_path/'RAM_zero_native';prepared='/RAM/prepared.json';h='a'*64
    summary=m._execute_verified(p,prepared,h,output,m.AUTH)
    assert summary['status']=='recorded' and summary['errors']==[]
    monkeypatch.setattr(m,'verify',lambda *a:p)
    result=m.consume(str(output),summary['report_sha256'])
    assert result['complete'] and len(result['rows'])==48 and result['neural_calls']==0
    assert result['census']==dict(pending_new_physics=0,reused_saved=36,paid_refused=10,not_reached=2)
    assert all(v==dict(attempted=0,completed=0) for v in result['new_physical_resources'].values())
    assert all(result[k] is False for k in m.FLAGS)


def test_failed_native_RAM_callback_preserves_all48_attempt_return_and_closes(saved,tmp_path,monkeypatch):
    p=ram_physical_input(saved);p['rows'][0]['route']='execute'
    p['handoff']['rows'][0]['route']='pending_new_physics'
    p['handoff']['new_budget']=m.new_budget(p['handoff']['rows']);p['profile']=m.profile(p['handoff'])
    called=[]
    def factory(meter):
        called.append('constructor')
        return SimpleNamespace(close=lambda:called.append('close'))
    ram_source(monkeypatch,factory)
    def fault(*a):raise RuntimeError('RAM native failure before any physics')
    monkeypatch.setattr(m,'execute',fault)
    output=tmp_path/'RAM_failed_native';summary=m._execute_verified(p,'/RAM/prepared.json','a'*64,output,m.AUTH)
    r=m.READ(output/'report.json',m.physical.FINAL)
    assert summary['status']=='engineering_failed' and len(r['rows'])==48 and r['rows'][0]['own_returned']==0
    assert r['rows'][0]['final_metrics'] is None and r['rows'][0]['status']=='engineering_failed'
    assert called==['constructor','close'] and r['resources']['constructor']==r['resources']['close']==dict(attempted=1,completed=1)
    assert r['resources']['physics_step']==dict(attempted=0,completed=0)
    monkeypatch.setattr(m,'verify',lambda *a:p)
    with pytest.raises(ValueError,match='engineering failure'):m.consume(str(output),summary['report_sha256'])


def test_actual_MAIN_full_strict_ABI_and_every_compiled_raw_bit_route(saved):
    path=m.ROOT/'results/agent_p3_paired_quota_runtime_20261028_v1_consume_MAIN01.json'
    assert m.sha(path)=='3b62bc4cbf669ecbea10f5ecbf9863a1d96d2432ce029b75059c4edbc2838e6f'
    receipt=m.READ(path,256*1024**2);r=m.READ(m.RUNTIME_REPORT,256*1024**2)
    assert receipt['report_sha256']==m.RUNTIME_REPORT_SHA and receipt['full_rows']==r['session']['rows']
    assert receipt['census']==dict(compiled=36,paired_paid_refusal=10,not_reached=2)
    assert receipt['actual_resources']['fixed']['completed']['predict_API']==receipt['actual_resources']['adaptive']['completed']['predict_API']==634
    p,_,op,orr,old_RAM_fixture,manifest=saved
    # Pure route arithmetic, NOT material/producer.consume/old.consume or a new
    # physical evidence receipt: old_RAM_fixture has no external authority.
    out=m.project(p,r,op,orr,old_RAM_fixture,'strict_saved',manifest)
    assert len(out['rows'])==48
    for i,row in enumerate(out['rows']):
        actual=r['session']['rows'][i]
        if actual['status']=='compiled':
            selected=actual['compiler']['selected'];command=m.raw_identity(selected['raw5'],5)
            assert row['command']==command and row['selection_digest']==selected['selection_digest']
            same=op['rows'][i]['selected'] is not None and command==m.raw_identity(op['rows'][i]['selected']['raw5'],5)
            assert (row['route']=='reused_saved')==same
            assert row['route'] in ('reused_saved','pending_new_physics')
        else:assert row['command'] is None and row['runner_plan']['selected'] is None
