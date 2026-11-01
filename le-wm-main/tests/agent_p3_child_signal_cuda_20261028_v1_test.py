"""RAM array arithmetic only; fixture HOST labels never certify CUDA execution."""
from copy import deepcopy
from functools import lru_cache
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_child_signal_cuda_20261028_v1 as m


def load_test(stem):
    path=m.core.ROOT/f'tests/{stem}_test.py'
    spec=importlib.util.spec_from_file_location('_RAM_'+stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


c=load_test('agent_p3_child_scheduler_contract_20261028_v1')
original=load_test('agent_p3_local_signal_cuda_20261028_v1')


class Sink:
    def __init__(self):self.events=[]
    def event(self,e):self.events.append(deepcopy(e))


class RamChild(original.RamSignal):
    kind='RAM_FIXTURE'
    source_sha256=m.sha(m.__file__)
    def __init__(self,task,child,sink,fault=False):
        t=deepcopy(task);t['root_context']=deepcopy(child['context'])
        super().__init__(t,sink,fault);self.event_sink=sink.event
    def signal(self,req,ledger):
        scope=deepcopy(ledger.scope);ledger.scope=dict(scope,horizon=req['horizon'])
        try:
            result=super().signal(req,ledger);result['producer_sha256']=self.source_sha256;return result
        finally:ledger.scope=scope


@lru_cache(maxsize=2)
def fixture(sharing='independent'):
    child=c.child();tasks=[c.task(),c.task(1)];traces={}
    def factory(actual,group):
        sink=Sink();traces[child['child_id']+':'+group]=sink.events
        return RamChild(tasks[0],actual,sink)
    result=m.contract.plan_child(child,tasks,factory,sharing=sharing,permit_ram=True)
    assert result['status']=='ram_planned_not_evidence'
    # Exercise saved ABI only. These declarations are never actual CUDA proof.
    result['status']='child_plan_not_admitted'
    for a in result['acquisitions']:a['backend_kind']='HOST_CUDA'
    return child,tasks,result,traces


@pytest.mark.parametrize('sharing',['independent','shared_equal'])
def test_full_paid_child_arrays_and_original_decision_saved_consumer(sharing):
    child,tasks,result,traces=fixture(sharing)
    m.validate_child_result(child,tasks,result,traces,sharing)
    assert result['fixed_denominator']==2 and all(row['commands'] is None for row in result['rows'])
    for a in result['acquisitions']:
        r=a['resource_receipt'];assert r['completed']['predict_API']==r['completed']['predict_samples']==8
        assert r['completed']['jacobian_API']==2 and r['completed']['VJP_directions']==384
        assert r['completed']['grad_API']==r['completed']['raw_backward_callbacks']==384
        assert r['completed']['encoder_calls']==r['completed']['projector_calls']==0
        assert r['FLOPs'] is r['cuda_ns'] is r['backward_engine_calls'] is None
    assert result['rows'][1]['decision']['extension_type']=='reconfiguration'
    assert sum(row['signal_charge']['numerator']/row['signal_charge']['denominator'] for row in result['rows'])==len(result['acquisitions'])


@pytest.mark.parametrize('fault',['origin_root','zero_history','history_scale','child_identity','source','J','raw_grad','basis',
    'token','module_input','module_shape','nominal','AD','direction','extra_direction','resource','journal_order',
    'chi','gain','quota','budget','charge','drop_method','drop_acquisition','queue','future','gate'])
def test_full_array_saved_tampering_rejected(fault):
    child,tasks,result,traces=deepcopy(fixture());a=result['acquisitions'][0];events=traces[a['group']]
    def array_event(kind,key):return next(e for e in events if e['kind']==kind and key in e.get('arrays',{}))
    def mutate(kind,key,fn):
        e=array_event(kind,key);v=m.base.unblob(e['arrays'][key]).copy();fn(v);e['arrays'][key]=m.base.blob(v)
    if fault=='origin_root':
        array_event('origin','latent')['arrays']['latent']=m.base.blob(np.asarray(tasks[0]['root_context']['latent_history'],np.float32)[None])
    elif fault=='zero_history':mutate('origin','history',lambda a:a.fill(0))
    elif fault=='history_scale':mutate('origin','history',lambda a:a.__itruediv__(100))
    elif fault=='child_identity':result['child_sha256']='0'*64
    elif fault=='source':a['signal']['responses'][0]['producer_sha256']='0'*64
    elif fault=='J':a['signal']['responses'][0]['jacobian'][0][0]=1
    elif fault=='raw_grad':mutate('raw_backward','value',lambda a:a.fill(1))
    elif fault=='basis':mutate('io_input','grad:basis',lambda a:a.fill(0))
    elif fault=='token':mutate('packet','token',lambda a:a.fill(1))
    elif fault=='module_input':mutate('io_input','predictor:0',lambda a:a.fill(3))
    elif fault=='module_shape':next(op for op in a['resource_receipt']['events'] if op['resource']=='predict')['shape'][0]=2
    elif fault=='nominal':mutate('nominal','value',lambda a:a.fill(1))
    elif fault=='AD':mutate('AD_terminal','value',lambda a:a.fill(1))
    elif fault=='direction':next(e for e in events if e['kind']=='direction')['direction']=1
    elif fault=='extra_direction':
        e=deepcopy(next(e for e in events if e['kind']=='direction'));e['scope']['horizon']=2;events.append(e)
    elif fault=='resource':a['resource_receipt']['completed']['predict_API']-=1
    elif fault=='journal_order':
        i=next(i for i,e in enumerate(events) if e['kind']=='resource_attempt');events[i],events[i+1]=events[i+1],events[i]
    elif fault=='chi':a['signal']['scheduling_signal']['unreachable_fraction']=0
    elif fault=='gain':a['signal']['signed_horizon_coverage_gain']=1
    elif fault=='quota':result['rows'][1]['decision']['candidates']=4
    elif fault=='budget':result['rows'][1]['search_predict_cap']=48
    elif fault=='charge':result['rows'][0]['signal_charge']['denominator']=2
    elif fault=='drop_method':result['rows'].pop()
    elif fault=='drop_acquisition':result['acquisitions'].pop()
    elif fault=='queue':result['rows'][0]['queue'].pop()
    elif fault=='future':result['rows'][0]['queue'][0]['native_success']=True
    else:result['P3_admission']=True
    with pytest.raises((ValueError,KeyError,IndexError)):m.validate_child_result(child,tasks,result,traces,'independent')


@pytest.mark.parametrize('fault',['outcome','root','history','horizon_bool','horizon2','task'])
def test_native_bad_child_request_rejected_before_torch(fault,monkeypatch):
    child=c.child();task=c.task();native=object.__new__(m.NativeChildSignal)
    native.task=task;native.child=child;native.device=SimpleNamespace(type='cuda')
    req=m.core.request(task,dict(context=child['context']),operation='compile_finite_horizon_J',horizon=1)
    if fault=='outcome':req['native_success']=True
    elif fault=='root':req['parent_context']=deepcopy(task['root_context']);req['parent_context_sha256']=m.digest(task['root_context'])
    elif fault=='history':req['parent_context']['action_history'][0][0]=0
    elif fault=='horizon_bool':req['horizon']=True
    elif fault=='horizon2':req['horizon']=2
    else:req['binding']['model_sha256']='0'*64
    attempted=[];original_import=builtins.__import__
    def guard(name,*args,**kwargs):
        if name=='torch' or name.startswith('torch.'):attempted.append(name);raise AssertionError('no torch permitted')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    with pytest.raises(ValueError):native.signal(req,None)
    assert attempted==[]


def test_valid_nonzero_history_child_request_without_any_model():
    child=c.child();task=c.task();native=object.__new__(m.NativeChildSignal);native.task=task;native.child=child
    req=m.core.request(task,dict(context=child['context']),operation='compile_finite_horizon_J',horizon=3)
    assert native.checked_request(req)==3
    assert np.asarray(req['parent_context']['action_history']).any()
    assert req['parent_context_sha256']!=m.digest(task['root_context'])


@pytest.mark.parametrize('sharing,groups',[('independent',46),('shared_equal',23)])
def test_profile_preserves_unreached_and_exact_paid_caps(sharing,groups):
    children=[dict(status='observed_child') for _ in range(23)]+[dict(status='not_reached')]
    p=m.profile(dict(children=children,inputs=dict(sharing=sharing)))
    assert p['acquisitions']==groups and p['method_slots']==48 and p['child_slots']==24
    caps=p['resource_caps'];assert caps['predict_API']==caps['predict_samples']==8*groups
    assert caps['jacobian_API']==2*groups and caps['VJP_directions']==384*groups
    assert caps['schedule_decisions']==46 and caps['encoder_calls']==caps['projector_calls']==0
    assert p['wall_seconds']==900 and p['output_bytes']==512*1024**2


def test_failure_keeps_two_methods_and_nonrefundable_native_attempts():
    child=c.child();tasks=[c.task(),c.task(1)];sinks=[]
    def factory(actual,group):
        sink=Sink();sinks.append(sink);return RamChild(tasks[0],actual,sink,fault=True)
    result=m.contract.plan_child(child,tasks,factory,permit_ram=True)
    assert len(result['rows'])==len(result['acquisitions'])==2 and all(r['queue'] is None for r in result['rows'])
    for a in result['acquisitions']:
        receipt=a['resource_receipt'];m.core.audit_ledger(receipt)
        assert receipt['attempted']['predict_API']==receipt['completed']['predict_API']==2
        assert receipt['attempted']['jacobian_API']==1 and receipt['completed']['jacobian_API']==0
        assert receipt['attempted']['signal_compile_calls']==1 and receipt['completed']['signal_compile_calls']==0


def test_run_authorization_before_source_verify_or_torch(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(m,'verify',lambda *a:calls.append(a))
    with pytest.raises(ValueError):m.run(str(tmp_path/'not_read'),'0'*64,str(tmp_path/'not_created'),None)
    assert calls==[] and not (tmp_path/'not_created').exists()


def test_cold_import_no_neural_or_physics():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(x in sys.modules for x in ("torch","gym","gymnasium","pymunk"))'
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.core.ROOT,check=True,capture_output=True,timeout=15)


def test_real_contract_external_sha_saved_profile_not_a_signal_run():
    path=m.core.ROOT/'results/agent_p3_child_scheduler_contract_20261028_v1_prepare_MAIN01.json'
    expected='8669fb219c6a17e05d5c1b8dc557b401651a2bf3fc759529d64231e5d317f7fd'
    assert m.sha(path)==expected
    p=m.base.read(path,4*1024**2)
    assert len(p['children'])==24 and len(p['jobs'])==48 and p['new_child_signals'] is None
    assert m.profile(p)['observed_children']==23
    assert all(len(t['binding']['root_sha256'])==64 for t in p['tasks'])
    assert all(t['binding']['model_sha256']=='48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607' for t in p['tasks'])


def test_store_chain_new_output_and_tamper(tmp_path,monkeypatch):
    limits=m.profile(dict(children=[],inputs=dict(sharing='independent')))
    store=m.base.Store(tmp_path/'RAM',limits);store.event(dict(RAM='only'));store.close()
    events,tail=m.journal(tmp_path/'RAM/events.jsonl');assert events==[dict(RAM='only')] and tail==store.tail
    with pytest.raises(FileExistsError):m.base.Store(tmp_path/'RAM',limits)
    path=tmp_path/'RAM/events.jsonl';content=path.read_bytes();path.write_bytes(content.replace(b'only',b'FAIL'))
    with pytest.raises(ValueError):m.journal(path)
    path.write_bytes(content[:-1])
    with pytest.raises(ValueError):m.journal(path)
    root=tmp_path/'repo';(root/'results').mkdir(parents=True);monkeypatch.setattr(m,'ROOT',root)
    out=root/'results'/(m.NAME+'_RAM.json');m.write_new(str(out),dict(RAM='only'))
    with pytest.raises(ValueError):m.write_new(str(out),{})


def make_saved(tmp_path):
    child,tasks,result,traces=deepcopy(fixture())
    children=[c.child(e,b) for e in m.contract.EPISODES for b in m.contract.STEPS]
    for ch in children:
        if ch['child_id']==child['child_id']:continue
        ch['status']='not_reached'
        for k in ('context','context_sha256','raw_history10','executed_prefix','observation','original_compiler_context_sha256'):ch[k]=None
        ch['ancestry']['original_request_sha256']=None
    prepared=dict(children=children,tasks=tasks,inputs=dict(sharing='independent'))
    p=dict(child_contract=prepared,profile=m.profile(prepared),sources={m.__file__:m.sha(m.__file__)},expected_model_hash=m.digest('RAM_ONLY_MODEL_DECLARATION'))
    rows=m.roster(prepared);row=next(r for r in rows if r['child_id']==child['child_id']);row.update(status='returned',result=result)
    output=tmp_path/'RAM_ONLY_saved_not_GPU';store=m.base.Store(output,p['profile'])
    binding=dict(prepared=str(tmp_path/'RAM_prepared.json'),prepare_sha256=m.digest('RAM_prepare'))
    store.put('binding.json',binding)
    store.event(dict(kind='denominator',rows=m.roster(prepared),methods=48,source_manifest_sha256=m.digest(p['sources'])))
    model=dict(original_state_dict=p['expected_model_hash'],all_buffers=m.digest('RAM_ONLY_BUFFER_DECLARATION'))
    store.event(dict(kind='model_before',hashes=model))
    for group,events in traces.items():
        for event in events:store.event(dict(kind='child_trace',group=group,payload=event))
    store.event(dict(kind='child_return',row=row));store.event(dict(kind='model_after',hashes=model))
    store.event(dict(kind='sources_after',sha256=m.digest(p['sources'])));store.close()
    report=dict(schema=m.NAME+'.report',status='paid_child_signals_saved_not_admitted',**binding,rows=rows,
        fixed_denominator=48,child_denominator=24,sharing='independent',resources=m.total_resources(rows),errors=[],
        model_before=model,model_after=model,sources_verified=True,cleanup_ok=True,source_manifest_sha256=m.digest(p['sources']),
        host=dict(device='cuda:0',index=0,name='RAM_NOT_GPU',torch_version='RAM_ONLY',torch_cuda='RAM_ONLY',
            allocation_sum=4.,visible_nvidia_nodes=['RAM_NOT_GPU']),wall_seconds=1.,CUDA_peak_bytes=[0,0],
        journal_tail=store.tail,artifact_sha256={n:m.sha(output/n) for n in ('binding.json','events.jsonl')},**m.FLAGS)
    store.put('report.json',report)
    return p,output,m.sha(output/'report.json')


def test_strict_consume_saved_RAM_artifacts_with_explicit_mocked_prepare(tmp_path,monkeypatch):
    p,output,h=make_saved(tmp_path);calls=[]
    monkeypatch.setattr(m,'verify',lambda *a:(calls.append(a) or p));monkeypatch.setattr(m,'pins',lambda:{})
    r=m.consume(str(output),h)
    assert len(calls)==1 and r['fixed_denominator']==48 and r['not_reached_methods']==46
    assert r['resources']['completed']['predict_API']==16 and r['resources']['completed']['VJP_directions']==768
    assert r['measured_search_budget_refusals']==0 and r['all_observed_methods_budget_refused'] is False
    assert all(v['denominator']==24 and v['observed']==1 and v['not_reached']==23 and v['refused']==0 for v in r['budget_refusals_by_mode'].values())
    assert r['saved_verification']==dict(array_SVD_calls=4,neural_calls=0,physics_calls=0)
    assert all(r[k] is False for k in m.FLAGS)
    calls.clear()
    with pytest.raises(ValueError):m.consume(str(output),'0'*64)
    assert calls==[]


def test_dependency_pin_failure_before_contract_verify(monkeypatch):
    calls=[];monkeypatch.setattr(m,'verify_contract',lambda *a:calls.append(a))
    monkeypatch.setattr(m,'pins',lambda:(_ for _ in ()).throw(ValueError('frozen source mismatch')))
    with pytest.raises(ValueError):m.material('/unopened','a'*64)
    assert calls==[]
