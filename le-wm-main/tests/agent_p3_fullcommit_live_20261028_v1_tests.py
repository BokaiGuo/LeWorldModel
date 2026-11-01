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
from experiments.spectral_reachability import agent_p3_fullcommit_live_20261028_v1 as m
import agent_p3_local_signal_cuda_20261028_v1_test as signal_helpers
import agent_p3_native_program_20261028_v1_test as program_helpers
import test_p1_moving_live_feedback_physical_20261027_v1 as physical_helpers


class Events:
    def __init__(self):self.events=[]
    def event(self,e):self.events.append(deepcopy(e))


def request(mode='adaptive',slot=0,boundary=5):
    base=signal_helpers.task(slot);base['goal']=[1.,.5]+[0.]*190
    base['binding']['goal_sha256']=m.digest(base['goal'])
    raw10=m.old.kernel.moving.warmup_commands();context=deepcopy(base['root_context'])
    context['action_history']=m.moving.normalized(base,raw10)
    identity=dict(episode_id=f's{slot}_{mode}',source_slot=slot,mode=mode,boundary=boundary,raw10=raw10.tolist(),commitment=None,
        observation=dict(boundary=boundary,png_sha256='a'*64,pixel_sha256='b'*64,encoding_sha256='c'*64,
            observed_index=3,encoded_index=5,observed_latent_sha256=m.old.packet_digest(context['latent_history'][-1])))
    p=dict(schema=m.NAME+'.node',base_task_sha256=m.digest(base),task=m.local_task(base,context,{k:v for k,v in identity.items() if k!='commitment'}),**identity)
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


def execute_ram(*,fault=None,at=18,compile_failure=None,visual_failure=None,real_array_plan=False,geometry='long',path='/RAM_LIVE'):
    p,data=physical_helpers.fixture();p['contract']=m.contract();p['tasks']=[]
    p['layers']=[next(x for x in p['layers'] if x['source_slot']==s and x['arm']==m.old.ARMS[0]) for s in (0,1)]
    data['visual_tasks']=data.pop('tasks')
    for s in (0,1):
        t=program_helpers.task(s);g=data['visual_tasks'][s]['gn_task']
        for k in ('root_context','goal','action_mean','action_std'):t[k]=deepcopy(g[k])
        t['binding'].update(context_sha256=m.digest(t['root_context']),goal_sha256=m.digest(t['goal']),
            normalizer_sha256=m.digest([t['action_mean'],t['action_std']]))
        if real_array_plan:
            t['goal']=[1.,.5]+[0.]*190
            t['binding']['goal_sha256']=m.digest(t['goal'])
        p['tasks'].append(m.core.validate_task(t))
    store=RAM(path);requests=[];envs=[];real_plan=m.plan_node
    def plan(request,factory,sink,ledger,**kwargs):
        requests.append(deepcopy(request));b=request['boundary']
        if b==compile_failure:raise RuntimeError('RAM compile failure')
        if real_array_plan:return real_plan(request,lambda t,s:RamBackend(t,s),sink,ledger,permit_ram=True)
        if request['commitment'] is not None:
            result=dict(status='observed_commitment',commands=None,resource_receipt=ledger.receipt())
        else:
            h=3 if request['mode']=='adaptive' and (geometry=='long' or (geometry=='tail' and b>5)) else 1
            if 5*h>20-b:
                result=dict(status='refused',refusal='fullH_does_not_fit',commands=None,resource_receipt=ledger.receipt())
            else:
                commands=np.full((5*h,2),np.float32(.02+(b/5)*.01+(request['mode']=='adaptive')*.01)).tolist()
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


def planned(mode='adaptive',boundary=5,commitment=None,caps=None):
    base,p=request(mode,boundary=boundary);p['commitment']=commitment
    sink=Events();ledger=m.core.Ledger(caps or m.arm_caps(mode),sink=sink.event)
    result=m.plan_node(p,lambda t,s:RamBackend(t,s),sink,ledger,permit_ram=True)
    return base,p,result,sink.events


@pytest.fixture(scope='module')
def complete_plan():return planned()


@pytest.mark.parametrize('mode,h,n,predict',[('fixed',1,4,17),('adaptive',3,4,35)])
def test_fullH_natural_budget_all_native_arrays(mode,h,n,predict):
    base,p,r,events=planned(mode);m.validate_request(p,base)
    a=m.audit_plan(p,r,events,r['resource_receipt'])
    assert a['horizon']==h and a['candidates']==n and len(r['commands'])==5*h
    assert r['commands']==r['winner']['actions'] and r['selected_path']==r['winner']['whole']
    c=r['resource_receipt']['completed']
    assert c==m.costs(h,n,schedule=True)
    assert c['predict_API']==c['predict_samples']==predict and c['predict_context_tokens']==3*predict
    assert c['replay_calls']==2*n+1 and c['metric_calls']==n
    assert c['VJP_directions']==384 and c['array_SVD_calls']==3


@pytest.mark.parametrize('boundary',[10,15])
def test_H3_residual_refuses_after_paid_signal_no_H1_padding(boundary):
    _,p,r,events=planned(boundary=boundary)
    m.audit_plan(p,r,events,r['resource_receipt'])
    assert r['status']=='refused' and r['refusal']=='fullH_does_not_fit'
    assert r['commands'] is None and not r['candidates']
    assert r['resource_receipt']['completed']==m.costs(schedule=True)


@pytest.mark.parametrize('boundary',[10,15])
def test_pending_H3_pays_signal_no_schedule_no_program(boundary):
    pending=dict(start=5,end=20,seal_sha256='e'*64)
    _,p,r,events=planned(boundary=boundary,commitment=pending)
    m.audit_plan(p,r,events,r['resource_receipt'])
    assert r['status']=='observed_commitment' and r['commands'] is None and r['queue'] is None
    assert r['resource_receipt']['completed']==m.costs()


@pytest.mark.parametrize('damage',['state','reward','success','goal7','env','future_rgb','mode','boundary','raw','older','current','model','Q','pending'])
def test_node_whitelist_and_task_identity(damage):
    base,p=request()
    if damage in ('state','reward','success','goal7','env','future_rgb'):p[damage]=[0]
    elif damage=='mode':p['mode']='I'
    elif damage=='boundary':p['boundary']=0
    elif damage=='raw':p['raw10'][0][0]=.5
    elif damage in ('older','current'):p['task']['root_context']['latent_history'][0 if damage=='older' else -1][0]=1
    elif damage=='model':p['task']['binding']['model_sha256']='d'*64
    elif damage=='pending':p['commitment']=dict(start=5,end=20,seal_sha256='e'*64)
    else:p['task']['projection'][0][0]=2
    with pytest.raises(ValueError):m.validate_request(p,base)


@pytest.mark.parametrize('damage',['raw','normalizer','ae','projection','boundary','J','grad','missing_io','cost','winner','first5','childJ','counter','direction','queue','extra_io'])
def test_saved_native_array_mutation_failclosed(complete_plan,damage):
    _,p,r,events=deepcopy(complete_plan)
    if damage=='missing_io':events.pop(next(i for i,e in enumerate(events) if e['kind']=='io_return'))
    elif damage=='extra_io':events.append(deepcopy(next(e for e in events if e['kind']=='io_return')))
    elif damage=='cost':r['candidates'][0]['cost']+=1
    elif damage=='winner':r['winner']['index']=99
    elif damage=='first5':r['commands']=r['winner']['actions'][:5]
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


def test_unsupported_original_decision_retains_paid_signal():
    _,p=request();p['task']['goal']=[0.,0.,1.]+[0.]*189
    p['task']['binding']['goal_sha256']=m.digest(p['task']['goal'])
    sink=Events();ledger=m.core.Ledger(m.node_caps(),sink=sink.event)
    r=m.plan_node(p,lambda t,s:RamBackend(t,s),sink,ledger,permit_ram=True)
    m.audit_plan(p,r,sink.events,r['resource_receipt'])
    assert r['refusal']=='unsupported_original_decision' and ledger.completed['predict_API']==8


def test_failed_dispatch_nonrefundable_and_remaining_arm_cap():
    _,p=request('fixed');sink=Events();caps=m.arm_caps('fixed');caps['predict_API']=8
    ledger=m.core.Ledger(caps,sink=sink.event)
    with pytest.raises(m.core.BudgetExceeded):
        m.plan_node(p,lambda t,s:RamBackend(t,s),sink,ledger,permit_ram=True)
    r=sink.events[-1]['result'];m.audit_resources(r['resource_receipt'],sink.events)
    assert r['commands'] is None and ledger.attempted['replay_calls']==1 and ledger.completed['replay_calls']==0
    slots=[dict(resource_receipt=r['resource_receipt'])]
    assert m.remaining_caps('fixed',slots)['predict_API']==43


def test_original_width8_bridge_retains_math_and_cast_once():
    base,p=request();sink=Events();ledger=m.core.Ledger(m.node_caps());ledger.scope=dict(test=True)
    response=program_helpers.response(p['task'],3)
    b=m.generate(p['task'],response,8,ledger)
    assert len(b['actions'])==8 and b['fractions']==[i/7 for i in range(8)]
    direction=np.asarray(b['solve']['params'],np.float64)
    assert b['actions']==[(f*direction/b['divisor']).astype(np.float32).reshape(15,2).tolist() for f in b['fractions']]
    assert ledger.completed['solve_calls']==ledger.completed['array_SVD_calls']==1
    ledger.scope=dict(test=True)
    original=m.native.generate(p['task'],response,ledger)
    bridge=m.generate(p['task'],response,4,ledger)
    assert original['actions']==bridge['actions']


def test_true_live_fullcommit_four_denominator_same_target20_not_MPC():
    r,store,envs,requests=execute_ram()
    assert r['status']=='known_fullcommit_recorded',r['errors']
    assert r['completed20']==4 and len(requests)==12
    assert all(len(e.commands)==30 and len(e.resets)==2 and e.closed for e in envs)
    assert r['counts']['public_step']==dict(attempted=120,completed=120)
    assert r['counts']['physics_step']==dict(attempted=1216,completed=1216)
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=16,completed=16)
    assert r['counts']['explicit_render']==dict(attempted=128,completed=128)
    assert m.census(store.events,r)['complete']
    for row in r['rows']:
        own=[p for p in requests if p['episode_id']==row['episode_id']]
        assert [e['step'] for e in row['encodings']]==[5,10,15,20]
        assert row['prefix_exact'] and row['completed_packets']==4
        if row['mode']=='adaptive':
            assert len(row['commitments'])==1 and len(row['commitments'][0]['commands'])==15
            assert [p['commitment'] is None for p in own]==[True,False,False]
            assert row['executed_raw'][5:]==row['commitments'][0]['commands']
        else:assert len(row['commitments'])==3
        for p in own:
            b=p['boundary'];raw=np.concatenate((m.old.kernel.moving.warmup_commands(),
                np.asarray(row['executed_raw'][:b],np.float32)))[-10:]
            assert p['raw10']==raw.tolist()
            assert not {'state','snapshot','env','reward','success','goal7'} & p.keys()


def test_tail_refusal_does_not_shorten_H3_or_force_H1():
    r,_,envs,_=execute_ram(geometry='tail')
    assert r['status']=='known_fullcommit_recorded'
    assert r['completed20']==2 and r['equal_actual_execution_horizon'] is False
    for row,e in zip(r['rows'],envs,strict=True):
        assert e.closed
        if row['mode']=='adaptive':
            assert row['control_returned']==10 and len(e.commands)==20
            assert row['status']=='contract_refused' and row['refusal']['boundary']==10
        else:assert row['control_returned']==20


@pytest.mark.parametrize('b',[5,10,15])
def test_compile_failure_stops_commitment_no_padding(b):
    r,_,envs,_=execute_ram(compile_failure=b)
    assert r['status']=='partial_or_failed' and len(r['nodes'])==12
    assert all(len(e.commands)==10+b and e.closed for e in envs)


@pytest.mark.parametrize('fault',['truncate','terminal','throw','substitute','guard','render'])
def test_native_stop_or_engineering_failure_retained(fault):
    r,s,envs,_=execute_ram(fault=fault,at=23)
    assert r['completed20']==0 and all(e.closed for e in envs)
    assert len(r['rows'])==4 and len(r['nodes'])==12
    assert all(len(e.commands)==23 for e in envs)
    if fault in ('truncate','terminal'):assert m.census(s.events,r)['complete']


@pytest.mark.parametrize('step',[5,10,15,20])
def test_stop_at_packet_boundary_no_extra_step(step):
    r,s,envs,_=execute_ram(fault='terminal',at=10+step)
    assert all(len(e.commands)==10+step and e.closed for e in envs)
    assert r['completed20']==(4 if step==20 else 0)
    assert m.census(s.events,r)['complete']


def test_visual_failure_never_dispatches_stale_commitment():
    r,_,envs,requests=execute_ram(visual_failure=10)
    assert r['status']=='partial_or_failed' and len(requests)==4
    assert all(len(e.commands)==20 and e.closed for e in envs)


def test_prepare_verify_exclusive_roundtrip(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    material=dict(schema=m.NAME+'.prepare',RAM_only=True)
    monkeypatch.setattr(m,'build',lambda:deepcopy(material))
    path=tmp_path/'results'/(m.NAME+'_prepare_RAM.json');receipt=m.prepare(path)
    assert m.verify(path,receipt['sha256'])==material
    with pytest.raises(ValueError):m.prepare(path)
    path.write_bytes(m.raw(dict(material,RAM_only=False)))
    with pytest.raises(ValueError):m.verify(path,receipt['sha256'])


def test_cold_import_and_no_unauthorized_production():
    code="""
import builtins
original=builtins.__import__
def guard(name,*a,**kw):
    if name.split('.')[0] in ('torch','gymnasium','pymunk','stable_worldmodel'):raise AssertionError(name)
    return original(name,*a,**kw)
builtins.__import__=guard
from experiments.spectral_reachability import agent_p3_fullcommit_live_20261028_v1 as m
try:m.run(None,None,None,authorization='no')
except ValueError:pass
else:raise AssertionError('authorization')
assert m.contract()['inclusive_predict_caps']==dict(fixed=51,adaptive=75)
"""
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('angle,velocity',[(0.,0.),(np.pi/2,0.),(2*np.pi-.01,13.),(7.,400.)])
def test_native_reward_not_wrapped_angle_or_XY_only(angle,velocity):
    goal=np.zeros(7);state=np.array([0.,0.,0.,0.,angle,velocity,-velocity])
    a=abs(angle);success=bool(min(a,2*np.pi-a)<np.pi/9)
    distance=float(np.linalg.norm(goal-state))
    got=m.native_return_contract(goal,state,-distance,success,False)
    assert got['native7_distance']==m.old.kernel.metric(goal,state)['native_distance']
    assert got['native_success']==m.old.kernel.metric(goal,state)['native_success']
    if velocity or angle:
        with pytest.raises(ValueError):m.native_return_contract(goal,state,0.,success,False)
    with pytest.raises(ValueError):m.native_return_contract(goal,state,-distance,not success,False)


def test_actual_known2_saved_exports_exact_task_ABI_no_neural():
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    assert m.sha(m.moving.OLD_PREPARE)==m.moving.OLD_SHA
    assert m.sha(m.signal.UPSTREAM)==m.signal.UPSTREAM_SHA
    previous=m.old.read(m.moving.OLD_PREPARE);up=m.signal.read(m.signal.UPSTREAM)
    for slot in (0,1):
        layer=next(x for x in previous['layers'] if x['source_slot']==slot and x['arm']==m.old.ARMS[0])
        task=enc.load_task(layer['task_export'],layer['task_sha256'])['gn_task']
        actual=m.core.validate_task(up['tasks'][slot])
        for k in ('root_context','goal','projection','action_mean','action_std'):assert task[k]==actual[k]
        assert len(m.prefix_binding(layer)['commands'])==5


def saved_ram(tmp_path,monkeypatch,*,fault=None,at=16):
    """Full synthetic array hooks + native-shaped RAM transitions, NOT science."""
    import io
    from PIL import Image
    import agent_p3_same_feedback_physical20_20261028_v2_tests as hf
    output=tmp_path/'RAM_SAVED';output.mkdir()
    r,store,_,_=execute_ram(fault=fault,at=at,real_array_plan=True,path=str(output))
    assert r['status']=='known_fullcommit_recorded',[(x['status'],x['first_error']) for x in r['rows']]
    p=store.test_p;p['sources']=hf.image_sources()['sources']
    p['modelbinding']=m.moving.suffix.modelbinding();p['expected_model_hash']=p['modelbinding']['tensor_sha256']
    prepared=str(tmp_path/'RAM_PREPARE.json');prepare_sha='a'*64
    hashes=dict(original_state_dict=p['expected_model_hash'],all_buffers='b'*64)
    precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    r.update(prepared=prepared,prepare_sha256=prepare_sha,source_manifest_sha256=m.digest(p['sources']),
        model_before=hashes,model_after=hashes,precision_before=precision,precision_after=precision,
        CUDA_peak_bytes=[0,0],per_episode_neural=m.node_accounting(r['nodes']))
    store.files['execution_binding.json']=m.raw(dict(prepared=prepared,prepare_sha256=prepare_sha))
    for name,payload in store.files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    stats=m.harden.image_contract(p)['statistics']
    for row in r['rows']:
        for enc in row['encodings']:
            ns=row['episode_id']+f"_b{enc['step']:02}"
            with Image.open(output/(row['episode_id']+f"_plan_{enc['step']:02}.png")) as image:rgb=np.asarray(image).copy()
            a=m.harden.preprocess(rgb,stats);b=io.BytesIO();np.save(b,a,allow_pickle=False)
            (output/(ns+'_encode/preprocessed_pixels.npy')).write_bytes(b.getvalue())
    report=output/'report.json';report.write_bytes(m.raw(r))
    monkeypatch.setattr(m,'verify',lambda *a:p)
    monkeypatch.setattr(m,'source_data',lambda *a:store.test_data)
    return output,r,prepared,prepare_sha


def test_full_saved_consumer_ALL_H3_offsets_paid_feedback_and_native_arrays(tmp_path,monkeypatch):
    output,r,p,h=saved_ram(tmp_path,monkeypatch)
    result=m.consume(output,m.sha(output/'report.json'),p,h)
    assert result['complete'] and result['completed20']==4 and len(result['audited'])==12
    assert result['RGB_preprocessing']['frames']==16 and result['physical_census']['complete']
    for arm in result['per_episode_neural']:
        assert arm['signal_completed']['predict_API']==24
        assert arm['completed']['predict_API']==51
        assert arm['program_completed']['predict_API']==27
    assert sum(a['status']=='observed_commitment' for a in result['audited'])==4
    assert not any(result[k] for k in ('P1_admission','P3_admission','scientific_admission','freshness'))
    # Valid report transport SHA must not certify fabricated full20 or a
    # replacement action after intermediate RGB. No producer file is changed.
    r['rows'][1]['commitments'][0]['commands'][8][0]+=.25
    (output/'s0_adaptive_episode.json').write_bytes(m.raw(r['rows'][1]))
    (output/'report.json').write_bytes(m.raw(r))
    with pytest.raises(ValueError):m.consume(output,m.sha(output/'report.json'),p,h)


def test_saved_consumer_native_stop_preserves_incomplete_fullH(tmp_path,monkeypatch):
    output,r,p,h=saved_ram(tmp_path,monkeypatch,fault='truncate')
    result=m.consume(output,m.sha(output/'report.json'),p,h)
    assert result['complete'] and result['completed20']==0 and len(result['audited'])==4
    assert all(x['control_returned']==6 for x in r['rows'])
    assert result['counts']['public_step']['completed']==64
    r['completed20']=4;(output/'report.json').write_bytes(m.raw(r))
    with pytest.raises(ValueError):m.consume(output,m.sha(output/'report.json'),p,h)

