"""Explicit RAM arrays/metadata and fake transitions only; no neural/physics."""
from copy import deepcopy
import builtins
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v1 as m
import agent_p3_local_signal_cuda_20261028_v1_test as signal_helpers
import agent_p3_native_program_20261028_v1_test as program_helpers
import test_p1_moving_live_feedback_physical_20261027_v1 as physical_helpers


class Events:
    def __init__(self):self.events=[]
    def event(self,e):self.events.append(deepcopy(e))


def request(mode='adaptive',slot=0,boundary=0):
    base=signal_helpers.task(slot);base['goal']=[1.,.5]+[0.]*190
    base['binding']['goal_sha256']=m.digest(base['goal'])
    raw10=m.old.kernel.moving.warmup_commands();context=deepcopy(base['root_context'])
    context['action_history']=m.moving.normalized(base,raw10)
    identity=dict(episode_id=f's{slot}_{mode}',source_slot=slot,mode=mode,boundary=boundary,raw10=raw10.tolist(),
        observation=dict(boundary=boundary,png_sha256='a'*64,pixel_sha256='b'*64,encoding_sha256='c'*64,
            observed_index=3,encoded_index=5,observed_latent_sha256=m.old.packet_digest(context['latent_history'][-1])))
    p=dict(schema=m.NAME+'.node',base_task_sha256=m.digest(base),task=m.local_task(base,context,identity),**identity)
    return base,p


class RamBackend(program_helpers.RamReplay):
    kind='RAM_FIXTURE'
    def signal(self,req,ledger):
        # Pure synthetic saved arrays. Populate a declared test response span;
        # NEVER claimed to be a model derivative or a research asset.
        begin=len(self.store.events);h=req['horizon']
        result=signal_helpers.RamSignal(self.task,self.store).signal(req,ledger)
        j=np.zeros((192,10*h),np.float32);j[0,0]=1
        if h==3:j[1,1]=1
        direction=0
        for e in self.store.events[begin:]:
            if e['kind']=='direction':direction=e['direction']
            elif e['kind']=='raw_backward':e['arrays']['value']=m.signal.blob(j[direction])
            elif e['kind']=='io_return' and 'grad:output' in e['arrays']:e['arrays']['grad:output']=m.signal.blob(j[direction])
            elif e['kind']=='jacobian':e['arrays']['value']=m.signal.blob(j)
        result['jacobian']=j.tolist();return result


def planned(mode='adaptive'):
    base,p=request(mode);sink=Events();ledger=m.core.Ledger(m.node_caps(),sink=sink.event)
    result=m.plan_node(p,lambda t,s:RamBackend(t,s),sink,ledger,permit_ram=True)
    return base,p,result,sink.events


@pytest.fixture(scope='module')
def complete_plan():return planned()


def test_paid_local_H3_whole_segmented_and_prefix_saved(complete_plan):
    base,p,r,events=complete_plan;m.validate_request(p,base)
    a=m.audit_plan(p,r,events,r['resource_receipt'])
    assert a['horizon']==3 and a['candidates']==4 and len(r['winner']['actions'])==15 and len(r['commands'])==5
    assert r['prefix_path']==r['winner']['whole'][:1]
    assert a['resources']['predict_API']==a['resources']['predict_samples']==33
    assert a['resources']['jacobian_API']==2 and a['resources']['VJP_directions']==384
    assert a['resources']['array_SVD_calls']==3 and a['resources']['solve_calls']==1
    assert a['resources']['replay_calls']==9 and a['resources']['metric_calls']==5
    assert a['resources']['predict_context_tokens']==99 and a['FLOPs'] is None


def test_fixed_H1_12_useful_candidates_same_API_not_same_search_axes():
    _,p,r,e=planned('fixed');a=m.audit_plan(p,r,e,r['resource_receipt'])
    assert a['horizon']==1 and a['candidates']==12 and len(r['commands'])==5
    assert a['resources']['predict_API']==33 and a['resources']['replay_calls']==25 and a['resources']['metric_calls']==13
    assert len({m.digest(v['actions']) for v in r['candidates']})==12


@pytest.mark.parametrize('damage',['state','reward','success','goal7','env','future_rgb','mode','boundary','raw','older','current','model','Q'])
def test_node_whitelist_and_original_task_history(damage):
    base,p=request()
    if damage in ('state','reward','success','goal7','env','future_rgb'):p[damage]=[0]
    elif damage=='mode':p['mode']='I'
    elif damage=='boundary':p['boundary']=20
    elif damage=='raw':p['raw10'][0][0]=.5
    elif damage in ('older','current'):p['task']['root_context']['latent_history'][0 if damage=='older' else -1][0]=1
    elif damage=='model':p['task']['binding']['model_sha256']='d'*64
    else:p['task']['projection'][0][0]=2
    with pytest.raises(ValueError):m.validate_request(p,base)


@pytest.mark.parametrize('damage',['raw','normalizer','ae','projection','boundary','J','grad','missing_io','cost','winner','fullcommit','childJ','counter','direction','queue','extra_io'])
def test_native_saved_mutations_failclosed(complete_plan,damage):
    _,p,r,events=deepcopy(complete_plan)
    if damage=='missing_io':events.pop(next(i for i,e in enumerate(events) if e['kind']=='io_return'))
    elif damage=='extra_io':events.append(deepcopy(next(e for e in events if e['kind']=='io_return')))
    elif damage=='cost':r['candidates'][0]['cost']+=1
    elif damage=='winner':r['winner']['index']=99
    elif damage=='fullcommit':r['commands']=r['winner']['actions']
    elif damage=='childJ':r['responses'][0]['parent_context_sha256']='a'*64
    elif damage=='counter':r['resource_receipt']['completed']['predict_API']-=1
    elif damage=='direction':next(e for e in events if e['kind']=='direction')['direction']=999
    elif damage=='queue':r['queue']['decision']['horizon']=1
    else:
        kind,key={'raw':('program_packet','raw'),'normalizer':('program_packet','mean'),
            'ae':('io_return','action_encoder:output'),'projection':('io_return','pred_proj:output'),
            'boundary':('program_boundary','latent'),'J':('jacobian','value'),'grad':('raw_backward','value')}[damage]
        e=next(e for e in events if e['kind']==kind and key in e['arrays'])
        a=m.signal.unblob(e['arrays'][key]).copy();a.flat[0]+=1;e['arrays'][key]=m.signal.blob(a)
    with pytest.raises((ValueError,KeyError)):m.audit_plan(p,r,events,r['resource_receipt'])


def test_reconfiguration_no_fallback_and_paid_failure():
    _,p=request();p['task']['goal']=[0.,0.,1.]+[0.]*189
    p['task']['binding']['goal_sha256']=m.digest(p['task']['goal'])
    sink=Events();ledger=m.core.Ledger(m.node_caps(),sink=sink.event)
    with pytest.raises(ValueError,match='unsupported reconfiguration'):
        m.plan_node(p,lambda t,s:RamBackend(t,s),sink,ledger,permit_ram=True)
    last=sink.events[-1]['result']
    assert last['commands'] is None and last['candidates']==[]
    assert ledger.completed['predict_API']==8 and ledger.completed['jacobian_API']==2 and ledger.completed['program_compile_calls']==0


def test_search_budget_failure_no_commands_and_no_refund():
    _,p=request('fixed');sink=Events();caps=m.node_caps();caps['predict_API']=8
    ledger=m.core.Ledger(caps,sink=sink.event)
    with pytest.raises(m.core.BudgetExceeded):m.plan_node(p,lambda t,s:RamBackend(t,s),sink,ledger,permit_ram=True)
    result=sink.events[-1]['result']
    assert result['commands'] is None and result['winner'] is None
    assert ledger.completed['predict_API']==8 and ledger.completed['VJP_directions']==384
    assert ledger.attempted['replay_calls']==1 and ledger.completed['replay_calls']==0
    m.audit_resources(result['resource_receipt'],sink.events)


def test_prepare_verify_own_exclusive_metadata_roundtrip(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    material=dict(schema=m.NAME+'.prepare',RAM_only=True)
    monkeypatch.setattr(m,'build',lambda:deepcopy(material))
    path=tmp_path/'results'/(m.NAME+'_prepare_RAM.json');receipt=m.prepare(path)
    assert m.verify(path,receipt['sha256'])==material
    with pytest.raises(ValueError,match='new exclusive'):m.prepare(path)
    path.write_bytes(m.raw(dict(material,RAM_only=False)))
    with pytest.raises(ValueError,match='external prepare SHA'):m.verify(path,receipt['sha256'])


def test_actual_saved_known2_task_exports_match_P3_ABI_readonly():
    # Explicit sealed known2 JSON only, no sourcepack/NPZ/weights/native reset.
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    assert m.sha(m.moving.OLD_PREPARE)==m.moving.OLD_SHA
    assert m.sha(m.signal.UPSTREAM)==m.signal.UPSTREAM_SHA
    previous=m.old.read(m.moving.OLD_PREPARE);up=m.signal.read(m.signal.UPSTREAM)
    assert len(up['tasks'])==2
    for slot in (0,1):
        layer=next(x for x in previous['layers'] if x['source_slot']==slot and x['arm']==m.old.ARMS[0])
        task=enc.load_task(layer['task_export'],layer['task_sha256'])['gn_task']
        actual=m.core.validate_task(up['tasks'][slot])
        for k in ('root_context','goal','projection','action_mean','action_std'):assert task[k]==actual[k]


class RAM(physical_helpers.RAM):
    finalizing=False
    def check(self,size=4096):assert sum(len(v) for v in self.files.values())+size<m.contract()['output_bytes']
    def png_bytes(self,entry):return self.files[str(Path(entry['path']).relative_to(self.path))]


class RamJournal(Events):
    def __init__(self,store,ns,check):
        super().__init__();self.store=store;self.path=store.path/ns/'events.jsonl';self.tail='0'*64
    def event(self,e):
        super().event(e);self.tail=m.digest(self.events)
        self.store.files[str(self.path.relative_to(self.store.path))]=m.raw(self.events)
    def close(self):pass


class FullRamJournal(Events):
    def __init__(self,store,ns,check):
        super().__init__();self.store=store;self.path=store.path/ns/'events.jsonl';self.tail='0'*64;self.lines=[]
    def event(self,e):
        super().event(e)
    def close(self):
        # Assemble the explicit RAM fixture AFTER its synthetic derivative
        # buffers are populated. Production NativeJournal is immediate/fsync.
        for i,e in enumerate(self.events):
            record=dict(index=i,previous=self.tail,payload=deepcopy(e));record['sha256']=m.digest(record)
            self.tail=record['sha256'];self.lines.append(m.raw(record)+b'\n')
        self.store.files[str(self.path.relative_to(self.store.path))]=b''.join(self.lines)


def execute_ram(*,fault=None,at=18,compile_failure=None,visual_failure=None,real_array_plan=False,path='/RAM_LIVE'):
    p,data=physical_helpers.fixture();p['contract']=m.contract();p['tasks']=[]
    p['layers']=[next(x for x in p['layers'] if x['source_slot']==s and x['arm']==m.old.ARMS[0]) for s in (0,1)]
    data['visual_tasks']=data.pop('tasks')
    for s in (0,1):
        t=program_helpers.task(s);g=data['visual_tasks'][s]['gn_task']
        for k in ('root_context','goal','action_mean','action_std'):t[k]=deepcopy(g[k])
        t['binding'].update(context_sha256=m.digest(t['root_context']),goal_sha256=m.digest(t['goal']),
            normalizer_sha256=m.digest([t['action_mean'],t['action_std']]))
        p['tasks'].append(m.core.validate_task(t))
    store=RAM(path);requests=[];envs=[];real_plan=m.plan_node
    def plan(request,factory,sink,ledger,**kwargs):
        requests.append(deepcopy(request));b=request['boundary']
        if b==compile_failure:raise RuntimeError('RAM compile failure')
        if real_array_plan:return real_plan(request,lambda t,s:RamBackend(t,s),sink,ledger,permit_ram=True)
        commands=np.full((5,2),np.float32(.02+(b/5)*.01+(request['mode']=='adaptive')*.01)).tolist()
        result=dict(status='sealed',commands=commands,resource_receipt=ledger.receipt())
        sink.event(dict(kind='RAM_plan',boundary=b));return result
    class Env(physical_helpers.ArrayEnv):
        def __init__(self):super().__init__(fault=fault,at=at)
    def factory(meter):e=m.old.metered_class(Env,meter)();envs.append(e);return e
    encode=m.old.Handler.encode_observation
    def visual(self,request,sink,**runtime):
        if visual_failure is not None and f'_b{visual_failure:02}' in request['episode_id']:raise RuntimeError('RAM visual failure')
        return encode(self,request,sink,**runtime)
    with physical_helpers.visual_mock(store) as (runtime,_),patch.object(m,'NativeJournal',FullRamJournal if real_array_plan else RamJournal),\
         patch.object(m,'plan_node',plan),patch.object(m,'sha',store.hash),patch.object(m.old,'sha',store.hash),\
         patch.object(m.old.Handler,'encode_observation',visual),\
         patch.object(m.old.source,'sample_state',lambda *a:physical_helpers.REQUEST.copy()):
        result=m.execute(p,data,store,m.old.Handler(None),runtime,factory,None,authorization=m.AUTH,
            final_check=lambda:dict(model_unchanged=True),permit_ram=True)
    store.test_p=p;store.test_data=data
    return result,store,envs,requests


def test_true_same_feedback_all20_four_decisions_not_one_prefix():
    r,s,envs,requests=execute_ram()
    assert r['status']=='known_same_feedback_recorded',r['errors']
    assert len(r['rows'])==4 and len(r['nodes'])==16 and r['completed20']==4 and len(requests)==16
    assert all(len(e.commands)==30 and len(e.resets)==2 and e.closed for e in envs)
    assert r['counts']['public_step']==dict(attempted=120,completed=120)
    assert r['counts']['physics_step']==dict(attempted=1216,completed=1216)
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=20,completed=20)
    assert r['counts']['close']==dict(attempted=4,completed=4)
    assert all([e['step'] for e in row['encodings']]==[0,5,10,15,20] for row in r['rows'])
    assert all(n['status']=='sealed' for n in r['nodes'])
    assert not r['P1_admission'] and not r['P3_admission'] and not r['freshness']
    for request in requests:
        b=request['boundary'];row=next(x for x in r['rows'] if x['episode_id']==request['episode_id'])
        expected=np.concatenate((m.old.kernel.moving.warmup_commands(),np.asarray(row['executed_raw'][:b],np.float32).reshape(-1,2)))[-10:]
        assert np.array_equal(request['raw10'],expected)
        if b==15:assert request['task']['root_context']['latent_history']==[[1.]*192]*3
        assert not {'state','snapshot','env','reward','success','goal7'} & request.keys()
    for row in r['rows']:
        events=[e for e in s.events if e.get('episode_id')==row['episode_id']]
        for b in m.BOUNDARIES:
            enc=next(e['index'] for e in events if e['kind']=='boundary_encoded' and e['actual_step']==b)
            seal=next(e['index'] for e in events if e['kind']=='boundary_sealed' and e['boundary']==b)
            step=next(e['index'] for e in events if e['kind']=='authorized_dispatch' and e['step']==b+1)
            assert enc<seal<step


@pytest.mark.parametrize('b',[0,5,10,15])
def test_compiler_failure_stops_no_reuse_or_padding(b):
    r,_,envs,requests=execute_ram(compile_failure=b)
    assert r['status']=='partial_or_failed' and len(r['nodes'])==16 and len(r['rows'])==4
    assert all(len(e.commands)==10+b and e.closed for e in envs)
    assert sum(n['status']=='failed' for n in r['nodes'])==4
    assert sum(n['status']=='not_reached' for n in r['nodes'])==4*(3-b//5)


@pytest.mark.parametrize('fault',['truncate','terminal','throw','substitute','guard','render'])
def test_native_stop_or_engineering_failure_kept_in_full_denominator(fault):
    r,_,envs,requests=execute_ram(fault=fault,at=18)
    assert len(r['rows'])==4 and len(r['nodes'])==16 and r['completed20']==0
    assert all(len(e.commands)==18 and e.closed for e in envs)
    assert all(n['status']=='not_reached' for n in r['nodes'] if n['boundary']>=10)
    assert all(x.get('actual_Q20') is None for x in r['rows'])
    if fault in ('truncate','terminal'):
        assert r['status']=='known_same_feedback_recorded'
        assert all(x['status']=='native_stopped_control' for x in r['rows'])


def test_rgb_failure_prevents_compile_and_dispatch():
    r,_,envs,requests=execute_ram(visual_failure=10)
    assert r['status']=='partial_or_failed' and len(requests)==8
    assert all(len(e.commands)==20 and e.closed for e in envs)


@pytest.mark.parametrize('step',[5,10,15,20])
def test_native_success_at_packet_end_is_not_padded_and_final20_is_retained(step):
    r,_,envs,_=execute_ram(fault='terminal',at=10+step)
    assert all(len(e.commands)==10+step and e.closed for e in envs)
    assert all(x['native_success'] is True and x['terminated'] is True for x in r['rows'])
    assert r['completed20']==(4 if step==20 else 0)
    assert all(x['completed_packets']==step//5 for x in r['rows'])
    assert all(x['full_horizon']==(step==20) for x in r['rows'])


def test_cold_import_no_foreign_neural_or_physics():
    code='''
import builtins,sys
old=builtins.__import__
def guard(name,*a,**k):
    if name.split('.')[0] in ('torch','gymnasium','pymunk','stable_worldmodel'):raise AssertionError(name)
    return old(name,*a,**k)
builtins.__import__=guard
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v1 as m
assert m.contract()['episodes']==4
'''
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr


def test_unauthorized_run_rejected_before_neural_import(monkeypatch):
    original=builtins.__import__;attempts=[]
    def guarded(name,*a,**k):
        if name.split('.')[0]=='torch':attempts.append(name);raise AssertionError(name)
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError,match='MAIN explicit'):m.run(None,None,None,authorization='no')
    assert not attempts


def test_saved_consume_native_stop_full_arrays_roundtrip_and_tamper(tmp_path,monkeypatch):
    output=tmp_path/'RAM_SAVED';output.mkdir()
    r,store,_,_=execute_ram(fault='truncate',at=11,real_array_plan=True,path=str(output))
    assert r['status']=='known_same_feedback_recorded',r['errors']
    p=store.test_p;p['sources']={};p['modelbinding']=m.moving.suffix.modelbinding()
    p['expected_model_hash']=p['modelbinding']['tensor_sha256']
    prepared=str(tmp_path/'RAM_PREPARE.json');prepare_sha='a'*64
    hashes=dict(original_state_dict=p['expected_model_hash'],all_buffers='b'*64)
    precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    r.update(prepared=prepared,prepare_sha256=prepare_sha,source_manifest_sha256=m.digest({}),
        model_before=hashes,model_after=hashes,precision_before=precision,precision_after=precision,
        CUDA_peak_bytes=[0,0],per_episode_neural=m.node_accounting(r['nodes']))
    store.files['execution_binding.json']=m.raw(dict(prepared=prepared,prepare_sha256=prepare_sha))
    for name,payload in store.files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    report=output/'report.json';report.write_bytes(m.raw(r))
    monkeypatch.setattr(m,'verify',lambda *a:p)
    monkeypatch.setattr(m,'source_data',lambda *a:store.test_data)
    result=m.consume(output,m.sha(report),prepared,prepare_sha)
    assert result['complete'] and result['completed20']==0 and len(result['audited'])==4
    assert result['counts']['public_step']['completed']==44
    assert all(row['native_success'] is False for row in r['rows'])
    # A valid transport SHA cannot promote first5/native-stop into full20.
    r['completed20']=4;report.write_bytes(m.raw(r))
    with pytest.raises(ValueError,match='actual all20'):m.consume(output,m.sha(report),prepared,prepare_sha)
