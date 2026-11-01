"""Known2 shared-paid H1/H3 panels; MAIN-only CUDA/native run, RAM-safe import.

Controlled pipeline ablation, NOT cheap fixed baseline, equal FLOPs or full C3.
Every reached boundary pays both panels; only the selected FIRST5 is released.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p3_effective_schedule_20261029_v1 as frozen

NAME = 'agent_p3_shared_panel_20261029_v1'
ROOT = frozen.ROOT
core, live, signal = frozen.core, frozen.live, frozen.signal
bridge, moving, old, native = frozen.bridge, frozen.moving, frozen.old, frozen.native
harden, frames = frozen.harden, frozen.frames
require, digest, sha, raw = core.require, core.digest, signal.sha, core.canonical
AUTH = 'MAIN_KNOWN2_SHARED_PANEL_CUDA_AND_PHYSICS'
BOUNDARIES = (5, 10, 15)
BANKS = deepcopy(frozen.BANKS)
ARMS = tuple(p+'_'+b for p in ('fixedH1', 'adaptive') for b in BANKS)
DESIGN = ROOT/'innerdocs/agent_p3_resource_matched_followup_20261029_v1.md'
DESIGN_SHA = '03aa37fcdd878ab854a0495f494652c80a917a45544b56722f5f62e3cb9cd194'
PINS = {Path(frozen.__file__): '7cef2600e9844d981f97fdd6c385bff0c3ae9bdf55713ac8c92cebd28b003c23',
        Path(live.__file__): 'b3e1c321883f47e9030777312989d2a318c699dfa207f53c11006294059e6188',
        Path(signal.__file__): '6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b',
        DESIGN: DESIGN_SHA}
FLAGS = dict(P1_admission=False, P3_admission=False, C3=False, scientific_admission=False,
             freshness=False, independent_confirmation=False, equal_FLOPs=False, equal_wall=False)


def split_arm(arm):
    require(type(arm) is str and arm in ARMS, 'fixed policy x bank, no new roots/arms')
    return arm.split('_', 1)


def contract():
    return dict(schema=NAME+'.contract', arms=list(ARMS), known_roots=[0, 1], episodes=8,
        node_slots=24, panel_slots=48, candidate_slots=192, boundaries=list(BOUNDARIES),
        raw_target=20, warmup=10, prefix=5, execute_packet=5, planning_horizons=[1, 3],
        fractions={k:list(v) for k,v in BANKS.items()}, candidates_per_panel=4,
        feedback='fresh actual RGB rolling3 / actual last10 raw at every reached5/10/15',
        panel_order=[1, 3], signals='fresh H1/H3; independent nominal and AD primal; 192VJP each',
        selection='frozen runtime_signal + core.choose HORIZON ONLY, fixedH1 or adaptive; not original width/depth execution',
        policy_scope='one rebased actual-node progress extension, not imagined child or depth search',
        unsupported='pay both panels then refuse nonprogress/noncontinuation; other episodes continue, no H1 fallback',
        original_policy_fields='preserve complete candidates/depth/type; actual panel width4 independently fixed',
        horizon_contract=dict(physical_end=20, H1_forecast_ends=[10,15,20], H3_forecast_ends=[20,25,30],
            allow_lookahead_past_physical_end=True, remaining_fit_gate=False,
            truncate_prediction=False, release='selected full forecast FIRST5 only; discard rest'),
        tie_break='(terminal original model-Q, candidate_index)', action_map=native.rule(),
        no_resample=True, no_padding=True, no_refund=True, no_retry=True,
        budget_match='shared_paid_pipeline_axes_if_complete', natural_cheap_baseline=False,
        complete20_predict_per_episode=132, total_predict_if_full20=1056,
        total_J_if_all_boundaries=48, total_VJP_if_all_boundaries=9216,
        equal_predict_if_full20=True, equal_execution_target=True, equal_total_compute=False, FLOPs=None,
        physical_caps=deepcopy(frozen.contract()['physical_caps']), visual_caps=dict(encoder=32,projector=32),
        wall_seconds=7200, output_bytes=2*1024**3, final_reserve_bytes=256*1024**2,
        disk_floor_bytes=200*1024**2, cuda_allocated_bytes=3*1024**3, cuda_reserved_bytes=4*1024**3,
        TF32=False, autocast=False, SDP='MATH', **FLAGS)


def costs(arm, boundary):
    split_arm(arm);require(type(boundary) is int and boundary in BOUNDARIES, 'reached boundary5/10/15')
    c=live.predict_axes(44)
    c.update(jacobian_API=2, VJP_directions=384, grad_API=384, raw_backward_callbacks=384,
        array_SVD_calls=4, solve_calls=2, signal_compile_calls=2, program_compile_calls=2,
        replay_calls=18, metric_calls=8, schedule_decisions=1)
    return c


def caps(arm):
    return {k:sum(costs(arm,b)[k] for b in BOUNDARIES) for k in core.AXES}


def accounting():
    return dict(contract=contract(), per_episode=[dict(arm=a,cap=caps(a),
        nodes=[dict(boundary=b,exact_if_completed=costs(a,b)) for b in BOUNDARIES],
        signal_predict=dict(nominal=12,AD_primal=12), program_predict=108,
        native_public_target=30,encoder_target=4,projector_target=4,
        attempted=None,completed=None,wall_ns=None,FLOPs=None) for a in ARMS],
        note='inclusive nested modules, not additive FLOPs/wall; earlystop actual may differ')


def roster():
    return [dict(episode_id=f's{s}_{a}',source_slot=s,mode=a,status='not_started',
        native_success=None,actual_control_returned=None) for s in (0,1) for a in ARMS]


def node_roster():
    return [dict(episode_id=r['episode_id'],source_slot=r['source_slot'],mode=r['mode'],
        boundary=b,status='not_reached',request=None,receipt=None,error=None,resource_receipt=None)
        for r in roster() for b in BOUNDARIES]


def native_roster():
    rows=[]
    for r in roster():
        row=deepcopy(bridge.template.roster()[0]);row.update(r)
        row.update(prefix_exact=False,packets=[],schedule_nodes=[],refusal=None,actual_Q20=None)
        rows.append(row)
    return rows


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def build():
    for path,h in PINS.items():require(sha(path)==h, 'sealed design/source SHA: '+str(path))
    # Rebuild original tasks/model/source authority, not an outcome-based arm selector.
    p=frozen.build();sources=dict(p['sources'])
    for path in own_files()+list(PINS):sources[str(path.resolve())]=sha(path)
    return dict(schema=NAME+'.prepare', contract=contract(), accounting=accounting(), sources=sources,
        tasks=p['tasks'],layers=p['layers'],policy=p['policy'],modelbinding=p['modelbinding'],
        expected_model_hash=p['expected_model_hash'],source_manifest=p['source_manifest'],
        authority=dict(original=p['authority'],design_sha256=DESIGN_SHA,
            original_prepare_schema=p['schema'],new_runtime_schema=True),roster=roster(),nodes=node_roster())


def exclusive(path):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists(),
        'new exclusive own output path')
    return p


def write_metadata(path, value, cap):
    p=exclusive(path);payload=raw(value);require(len(payload)<=cap, 'bounded metadata')
    with p.open('xb') as f:
        require(f.write(payload)==len(payload), 'complete metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(p.parent)
    return dict(path=str(p),sha256=sha(p))


def prepare(output):return write_metadata(output,build(),16*1024**2)


def verify(path, expected):
    core.hash_value(expected);p=harden.regular(path,16*1024**2)
    require(sha(p)==expected,'external prepare SHA')
    value=signal.read(p,16*1024**2)
    require(value==build() and sha(p)==expected,'full prepare reconstruction')
    return value


def make_request(base, context, *, slot, arm, boundary, raw10, observation):
    split_arm(arm)
    identity=dict(episode_id=f's{slot}_{arm}',source_slot=slot,mode=arm,boundary=boundary,
        raw10=np.asarray(raw10,np.float32).tolist(),observation=deepcopy(observation))
    return dict(schema=NAME+'.node',base_task_sha256=digest(base),
        task=live.local_task(base,context,identity),**identity)


def validate_request(req, base):
    core.keys(req,('schema','episode_id','source_slot','mode','boundary','raw10','observation','task','base_task_sha256'))
    split_arm(req['mode'])
    require(req['schema']==NAME+'.node' and type(req['source_slot']) is int and req['source_slot'] in (0,1)
        and req['episode_id']==f"s{req['source_slot']}_{req['mode']}" and type(req['boundary']) is int
        and req['boundary'] in BOUNDARIES,'own fixed node identity')
    t=core.validate_task(req['task']);identity={k:deepcopy(req[k]) for k in
        ('episode_id','source_slot','mode','boundary','raw10','observation')}
    require(req['base_task_sha256']==digest(base) and t==live.local_task(base,t['root_context'],identity),
        'same original task/goal/Q/model and actual node')
    require(len(t['goal'])==192 and np.array_equal(t['projection'],np.eye(192)), 'original identity192 Q')
    a=core.finite(req['raw10'],(10,2),True)
    require(np.all(abs(a)<=1) and t['root_context']['action_history']==moving.normalized(t,a), 'actual last10 normalized once')
    ob=req['observation'];core.keys(ob,('boundary','png_sha256','pixel_sha256','encoding_sha256',
        'observed_index','encoded_index','observed_latent_sha256'))
    require(ob['boundary']==req['boundary'] and type(ob['observed_index']) is int and type(ob['encoded_index']) is int
        and 0<=ob['observed_index']<ob['encoded_index'],'actual observation before encoding')
    for k in ('png_sha256','pixel_sha256','encoding_sha256','observed_latent_sha256'):core.hash_value(ob[k])
    require(ob['observed_latent_sha256']==old.packet_digest(t['root_context']['latent_history'][-1]), 'actual current latent')
    return t


def policy_decision(task, responses, diagnostic, arm):
    mode,_=split_arm(arm)
    # Rebased actual observed node, not a recursive predicted child; original rule unchanged.
    node,signals,receipt=signal.runtime_signal(task,responses,diagnostic)
    d=core.choose(task,node,signals,'fixed' if mode=='fixedH1' else 'adaptive')
    supported=(d['program_type']==d['extension_type']=='progress' and d['continue_expanding'] is True
        and d['horizon'] in (1,3))
    return dict(decision=d,signal_receipt=receipt,supported=supported,
        applied_fields=['horizon'],actual_candidates_per_panel=4,original_width_and_depth_executed=False,
        refusal=None if supported else 'unsupported_original_policy_type_or_continuation')


def empty_panels():
    return [dict(horizon=h,status='not_started',bank=None,winner=None,selected_path=None,
        candidates=[dict(index=i,status='not_started',actions=None,whole=None,segmented=None,cost=None) for i in range(4)])
        for h in (1,3)]


def remaining_caps(arm, results):
    c=caps(arm)
    for r in results:
        rr=r.get('resource_receipt')
        if rr is None:continue
        core.audit_ledger(rr)
        for k in c:c[k]-=rr['attempted'][k]
    require(all(v>=0 for v in c.values()), 'nonrefundable resource remainder')
    return c


def plan_node(req, base, factory, store, ledger, *, permit_ram=False):
    task=validate_request(req,base);s=req['source_slot'];_,bank=split_arm(req['mode'])
    backend=factory(task,store)
    require(getattr(backend,'kind',None)=='HOST_CUDA' or
        (permit_ram is True and getattr(backend,'kind',None)=='RAM_FIXTURE'),'HOST CUDA or explicit RAM fixture')
    require(not ledger.events,'fresh node ledger')
    r=dict(schema=NAME+'.node_result',request_sha256=digest(req),status='attempted',responses=[],diagnostic=None,
        panels=empty_panels(),policy=None,commands=None,selected_horizon=None,refusal=None,error=None,resource_receipt=None)
    store.event(dict(kind=NAME+'.node_request',request=deepcopy(req)))
    try:
        for h in (1,3):
            ledger.scope=dict(slot=s,horizon=h)
            request=core.request(task,dict(context=task['root_context']),operation='compile_finite_horizon_J',horizon=h)
            response=ledger.call('signal_compile',lambda request=request:backend.signal(request,ledger))
            r['responses'].append(response);store.event(dict(kind='local_signal',response=response))
        ledger.scope=dict(slot=s,phase='diagnostic')
        r['diagnostic']=signal.spectra(task,r['responses'],svd=lambda a:ledger.call('array_svd',lambda:np.linalg.svd(a,full_matrices=False)))
        # Complete BOTH paid panels even if policy later refuses. Never use costs to choose H.
        for panel,response in zip(r['panels'],r['responses'],strict=True):
            h=panel['horizon'];panel['status']='attempted'
            ledger.scope=dict(slot=s,mode=req['mode'],horizon=h,phase='program')
            panel['bank']=ledger.call('program_compile',lambda response=response:frozen.generate(task,response,bank,ledger))
            for row in panel['candidates']:
                row.update(status='attempted',actions=panel['bank']['actions'][row['index']])
                ledger.scope=dict(slot=s,mode=req['mode'],horizon=h,index=row['index'],phase='program')
                for segmented,key in ((False,'whole'),(True,'segmented')):
                    row[key]=ledger.call('replay',lambda segmented=segmented:backend.replay(task['root_context'],row['actions'],segmented,ledger))
                require(row['whole']==row['segmented'],'all candidate full native replay paths exact')
                terminal=core.checked_path(task,task['root_context'],row['actions'],row['whole'])
                row['cost']=ledger.call('metric',lambda:core.model_cost(task,terminal));row['status']='returned'
            winner=min(panel['candidates'],key=lambda row:(row['cost'],row['index']))
            ledger.scope=dict(slot=s,mode=req['mode'],horizon=h,index=winner['index'],phase='selected')
            path=ledger.call('replay',lambda:backend.replay(task['root_context'],winner['actions'],False,ledger))
            require(path==winner['whole'],'independent winner replay for BOTH panels')
            panel.update(status='returned',winner=deepcopy(winner),selected_path=path)
        ledger.scope=dict(slot=s,phase='schedule')
        r['policy']=ledger.call('schedule',lambda:policy_decision(task,r['responses'],r['diagnostic'],req['mode']))
        require(ledger.attempted==ledger.completed==costs(req['mode'],req['boundary']),'exact full dual-panel axes')
        if not r['policy']['supported']:
            r.update(status='refused',refusal=r['policy']['refusal'])
        else:
            h=r['policy']['decision']['horizon'];panel=next(p for p in r['panels'] if p['horizon']==h)
            r.update(status='sealed',selected_horizon=h,commands=deepcopy(panel['winner']['actions'][:5]))
        return r
    except BaseException as exc:
        r.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        r['resource_receipt']=ledger.receipt();store.event(dict(kind=NAME+'.node_final',result=deepcopy(r)))


class Session:
    def __init__(self, base, slot, arm):
        split_arm(arm);require(type(slot) is int and slot in (0,1),'known2 only')
        self.base=deepcopy(base);self.slot=slot;self.arm=arm;self.next_boundary=5
        self.pending=None;self.stopped=False;self.requests=[];self.results=[]

    def plan(self,context,raw10,observation,factory,store,*,permit_ram=False):
        require(not self.stopped and self.pending is None and self.next_boundary in BOUNDARIES,'observe only after actual ACK')
        req=make_request(self.base,context,slot=self.slot,arm=self.arm,boundary=self.next_boundary,raw10=raw10,observation=observation)
        self.requests.append(req);ledger=core.Ledger(remaining_caps(self.arm,self.results),sink=store.event)
        try:r=plan_node(req,self.base,factory,store,ledger,permit_ram=permit_ram)
        except BaseException:
            self.stopped=True;self.results.append(dict(status='failed',resource_receipt=ledger.receipt()));raise
        self.results.append(r)
        if r['status']=='refused':self.stopped=True;return req,r,None
        self.pending=np.asarray(r['commands'],np.float32);return req,r,self.pending.copy()

    def acknowledge(self,returned,*,terminated=False,truncated=False):
        require(self.pending is not None and not self.stopped,'one outstanding FIRST5 packet')
        a=np.asarray(returned,np.float32)
        require(a.ndim==2 and a.shape[1]==2 and 0<=len(a)<=5 and a.tobytes()==self.pending[:len(a)].tobytes()
            and (len(a)==5 or terminated or truncated),'exact actual ACK; no early short padding')
        self.next_boundary+=len(a);self.pending=None;self.stopped=bool(terminated or truncated)


def audit_node(req, r, events, base):
    """Exact saved arithmetic/IO census, never a neural recomputation or physics proof."""
    t=validate_request(req,base);rr=r['resource_receipt'];s=req['source_slot'];arm=req['mode']
    require(r['schema']==NAME+'.node_result' and r['request_sha256']==digest(req) and r['error'] is None,
        'own successful node identity')
    bridge.audit_resources(rr,events)
    allowed={'resource_attempt','resource_return','io_input','io_return','origin','packet','boundary',
        'nominal','AD_terminal','jacobian','direction','raw_backward','program_packet','program_boundary',
        'local_signal',NAME+'.node_request',NAME+'.node_final'}
    require(all(e['kind'] in allowed for e in events) and events[0]==dict(kind=NAME+'.node_request',request=req)
        and events[-1]==dict(kind=NAME+'.node_final',result=r), 'full known native event vocabulary/bookends')
    require(sum(e['kind']==NAME+'.node_request' for e in events)==sum(e['kind']==NAME+'.node_final' for e in events)==1,
        'one exact node invocation')
    io={'predict':'latent','action_encoder':'0','predictor':'0','pred_proj':'0'}
    calls=[op for op in rr['events'] if op['resource'] in (*io,'grad')]
    require(sum(e['kind'] in ('io_input','io_return') for e in events)==2*len(calls),'complete native IO census')
    for e in events:
        for b in e.get('arrays',{}).values():require(signal.unblob(b).dtype==np.float32,'float32 native buffers')
    for op in rr['events']:
        linked=[e for e in events if e.get('operation_index')==op['index']]
        if op['resource'] in (*io,'grad'):
            require([e['kind'] for e in linked]==['io_input','io_return'] and all(e['scope']==op['scope'] for e in linked),
                'exact paid native input/return links')
            if op['shape'] is not None:
                require(linked[0]['arrays'][op['resource']+':'+io[op['resource']]]['shape']==op['shape'],'paid actual input shape')
        elif op['resource']=='raw_backward':require([e['kind'] for e in linked]==['raw_backward'],'paid raw callback')
        else:require(not linked,'no invented native link')
    require([e['response'] for e in events if e['kind']=='local_signal']==r['responses'] and
        [a['horizon'] for a in r['responses']]==[1,3],'fresh ordered H1/H3 pair')
    for response in r['responses']:
        h=response['horizon'];local=[e for e in events if e.get('scope')==dict(slot=s,horizon=h)]
        require(response['binding_sha256']==digest(t['binding']) and response['parent_context_sha256']==digest(t['root_context'])
            and response['producer_sha256']==sha(signal.__file__),'fresh current-node derivative source')
        n=core.finite(response['nominal'],(192,),True);j=core.finite(response['jacobian'],(192,10*h),True)
        for phase in ('nominal','AD'):
            v=bridge.audit_trajectory(t,local,phase,h,np.zeros((5*h,2),np.float32),None,program=False)
            require(np.array_equal(v,n),'independent nominal/AD native endpoint')
        def values(kind):return [signal.unblob(e['arrays']['value']) for e in local if e['kind']==kind]
        for kind,v in (('nominal',n),('AD_terminal',n),('jacobian',j)):
            require(len(values(kind))==1 and np.array_equal(values(kind)[0],v),'complete native '+kind)
        require([e['direction'] for e in local if e['kind']=='direction']==list(range(192)),'all VJP directions')
        gi=[e for e in local if e['kind']=='io_input' and 'grad:basis' in e['arrays']]
        go=[e for e in local if e['kind']=='io_return' and 'grad:output' in e['arrays']];rw=values('raw_backward')
        require(len(gi)==len(go)==len(rw)==192,'all scalar gradients/raw returns')
        for i,(a,b,c) in enumerate(zip(gi,go,rw,strict=True)):
            core.keys(a['arrays'],('grad:basis','grad:output','grad:raw'));core.keys(b['arrays'],('grad:output',))
            require(np.array_equal(signal.unblob(a['arrays']['grad:basis']),np.eye(192,dtype=np.float32)[i])
                and np.array_equal(signal.unblob(a['arrays']['grad:output']),n)
                and np.array_equal(signal.unblob(a['arrays']['grad:raw']),np.zeros(10*h,np.float32))
                and np.array_equal(signal.unblob(b['arrays']['grad:output']),j[i]) and np.array_equal(c,j[i]),'all raw derivative rows')
    require(signal.spectra(t,r['responses'])==r['diagnostic'],'exact saved signal diagnostic')
    require([p['horizon'] for p in r['panels']]==[1,3],'all and only two panels')
    _,bank=split_arm(arm)
    for panel,response in zip(r['panels'],r['responses'],strict=True):
        h=panel['horizon'];dummy=core.Ledger(caps(arm));dummy.scope=dict(saved_recompute=True)
        rebuilt=frozen.generate(t,response,bank,dummy)
        require(panel['status']=='returned' and panel['bank']==rebuilt and len(panel['candidates'])==4,'both complete fixed banks')
        for i,row in enumerate(panel['candidates']):
            require(row['index']==i and row['status']=='returned' and row['actions']==rebuilt['actions'][i]
                and row['whole']==row['segmented'],'every full candidate identity')
            local=[e for e in events if e.get('scope')==dict(slot=s,mode=arm,horizon=h,index=i,phase='program')]
            for phase in ('whole','segmented'):bridge.audit_trajectory(t,local,phase,h,row['actions'],row[phase],program=True)
            terminal=core.checked_path(t,t['root_context'],row['actions'],row['whole'])
            require(row['cost']==core.model_cost(t,terminal),'original model-Q on full endpoint')
        winner=min(panel['candidates'],key=lambda x:(x['cost'],x['index']))
        require(panel['winner']==winner and panel['selected_path']==winner['whole'],'stable complete panel winner')
        local=[e for e in events if e.get('scope')==dict(slot=s,mode=arm,horizon=h,index=winner['index'],phase='selected')]
        bridge.audit_trajectory(t,local,'whole',h,winner['actions'],panel['selected_path'],program=True)
    policy=policy_decision(t,r['responses'],r['diagnostic'],arm)
    require(r['policy']==policy,'entire original policy retained; horizon-only application')
    if not policy['supported']:
        require(r['status']=='refused' and r['selected_horizon'] is r['commands'] is None and r['refusal']==policy['refusal'],
            'explicit unsupported policy refusal, no fallback')
    else:
        h=policy['decision']['horizon'];panel=next(p for p in r['panels'] if p['horizon']==h)
        require(r['status']=='sealed' and r['selected_horizon']==h and r['refusal'] is None
            and r['commands']==panel['winner']['actions'][:5], 'only FIRST5 of independently replayed winner')
    require(rr['attempted']==rr['completed']==costs(arm,req['boundary']),'all44 predictions and separate axes')
    return dict(node_verified=True,resources=rr['completed'],whole_job_verified=False,**FLAGS)


class Store(bridge.Store):
    def __init__(self,path):
        self.path=exclusive(path);self.finalizing=False
        require(shutil.disk_usage(self.path.parent).free>=contract()['disk_floor_bytes']+contract()['output_bytes'],
            '2GiB output allowance plus200MiB floor')
        self.path.mkdir();signal.fsync_dir(self.path.parent);self.index=0;self.previous=None
    def check(self,size=4096):
        used=sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file())
        cap=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
        require(used+size<=cap and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,
            'own whole-output reserve/floor cap')


class Meter(bridge.Meter):
    def __init__(self,store,check):
        old.Meter.__init__(self,store,check);self.caps=contract()['physical_caps']|contract()['visual_caps']
        self.counts={k:dict(attempted=0,completed=0) for k in self.caps}


def episode(env, ref, data, layer, base, row, slots, visual, runtime, meter, store, planner_factory, *, permit_ram=False):
    if not moving.setup(env,ref,data['plan'],row,meter,store):return
    row['prefix']=bridge.prefix_binding(layer);store.put(row['episode_id']+'_prefix_commit.json',row['prefix'])
    obs=moving.packet(env,row['prefix']['commands'],start=0,row=row,ref=ref,meter=meter,store=store)
    if obs is None:return
    row['completed_packets']+=1;row['prefix_exact']=True
    context=deepcopy(base['root_context']);session=Session(base,row['source_slot'],row['mode'])
    for b in BOUNDARIES:
        slot=next(n for n in slots if n['episode_id']==row['episode_id'] and n['boundary']==b);slot['status']='attempted'
        meter.region='encoding'
        z,observation=moving.encode_boundary(visual,runtime,data['visual_tasks'][row['source_slot']],layer,row,
            obs,env.unwrapped,meter,store,b)
        raw10=np.concatenate((ref['raw_history'],np.asarray(row['executed_raw'],np.float32).reshape(-1,2)))[-10:]
        context=moving.roll(context,z,raw10,base)
        ns=row['episode_id']+f'_b{b:02}';journal=live.NativeJournal(store,ns,runtime['check_limits'])
        state=old.state_seal(env.unwrapped);meter.region='compiling'
        try:
            req,result,commands=session.plan(context,raw10,observation,planner_factory,journal,permit_ram=permit_ram)
            require(old.state_seal(env.unwrapped)==state,'planner cannot mutate evaluator state/goal')
        finally:
            journal.close()
            node=dict(request=session.requests[-1] if session.requests else None,
                result=session.results[-1] if session.results else None,
                native_journal_sha256=sha(journal.path),native_journal_tail=journal.tail)
            row['schedule_nodes'].append(node);store.put(ns+'_node.json',node)
            slot.update(request=node['request'],receipt=node['result'],native_journal_sha256=node['native_journal_sha256'],
                native_journal_tail=node['native_journal_tail'],resource_receipt=node['result']['resource_receipt'] if node['result'] else None)
        store.put(ns+'_request.json',req)
        seal=dict(schema=NAME+'.dispatch',episode_id=row['episode_id'],boundary=b,request_sha256=digest(req),
            result_sha256=digest(result),commands=result['commands'],commands_sha256=digest(result['commands']),
            native_journal_sha256=node['native_journal_sha256'],status=result['status'],selected_horizon=result['selected_horizon'])
        seal['seal_sha256']=digest(seal);slot.update(status=result['status'],seal=seal)
        store.put(ns+'_seal.json',seal);store.emit('boundary_sealed',episode_id=row['episode_id'],boundary=b,seal_sha256=seal['seal_sha256'])
        if result['status']=='refused':
            row.update(status='contract_refused',refusal=dict(boundary=b,reason=result['refusal']));return
        row['packets'].append(dict(start=b,end=b+5,seal_sha256=seal['seal_sha256'],commands=result['commands']))
        obs=moving.packet(env,commands,start=b,row=row,ref=ref,meter=meter,store=store)
        returned=np.asarray(row['executed_raw'][b:],np.float32).reshape(-1,2)
        session.acknowledge(returned,terminated=row['terminated'],truncated=row['truncated'])
        if row['control_returned']==b+5:row['completed_packets']+=1
        if obs is None:
            if row['control_returned']==20:obs=row['last_observation'];break
            return
    require(row['control_returned']==20,'physical ends20, no padding or lookahead execution')
    z,_=moving.encode_boundary(visual,runtime,data['visual_tasks'][row['source_slot']],layer,row,
        obs,env.unwrapped,meter,store,20)
    row.update(status='completed20',full_horizon=True,
        actual_Q20=float(np.mean((z.astype(float)-np.asarray(base['goal']))**2)))


def node_accounting(nodes):
    out=[]
    for r in roster():
        ns=[n for n in nodes if n['episode_id']==r['episode_id']]
        rs=[n['resource_receipt'] for n in ns if n['resource_receipt'] is not None]
        attempted={k:sum(v['attempted'][k] for v in rs) for k in core.AXES}
        completed={k:sum(v['completed'][k] for v in rs) for k in core.AXES}
        by_phase={phase:{k:sum(op['delta'][k] for v in rs for op in v['events'] if op['status']=='returned'
            and (op['scope'].get('phase') in ('program','selected'))== (phase=='program')) for k in core.AXES}
            for phase in ('signal_and_policy','program')}
        out.append(dict(episode_id=r['episode_id'],source_slot=r['source_slot'],mode=r['mode'],caps=caps(r['mode']),attempted=attempted,completed=completed,
            phases=by_phase,returned_nodes=sum(n['status'] in ('sealed','refused') for n in ns),
            inclusive_operation_wall_ns={k:sum(v['inclusive_operation_wall_ns'].get(k,0) for v in rs)
                for k in sorted({k for v in rs for k in v['inclusive_operation_wall_ns']})},
            wall_semantics='nested inclusive, not additive wall/FLOPs',FLOPs=None,cuda_ns=None,backward_engine_calls=None,
            scalar_backward_calls=None,AD_primal_predict_API=None,Jacobian_elements=None,
            supplemental_unknown_reason='not distinct counters in original core ledger; no unknown-to-zero imputation'))
    return out


def execute(p,data,store,visual,runtime,factory,planner_factory,*,authorization,final_check,permit_ram=False):
    require(authorization==AUTH and p['contract']==contract(),'MAIN authorization and own contract')
    require(set(runtime)=={'model','device','check_limits'} and getattr(runtime['device'],'type',None)=='cuda','HOST CUDA only')
    rows=native_roster();slots=node_roster();meter=Meter(store,runtime['check_limits'])
    report=dict(schema=NAME+'.report',status='running',rows=rows,nodes=slots,errors=[],counts=meter.counts,
        pending=meter.pending,contract=contract(),fixed_denominator=8,node_denominator=24,panel_denominator=48,
        candidate_denominator=192,known_tasks=2,**FLAGS)
    data['live_report']=report;store.put('roster.json',dict(rows=rows,nodes=slots));store.emit('run_started',contract=contract())
    for row in rows:
        env=None;meter.episode=row['episode_id'];meter.region='setup'
        try:
            store.emit('initialization_intent',episode_id=row['episode_id'],
                **bridge.initialization_intent(data['references'][row['source_slot']],data['plan']))
            env=meter.call('constructor',lambda:factory(meter));body=env.unwrapped
            require(body._live_meter is meter and body._live_constructor_complete is True and body.with_target is False
                and body.render_action is False and body.render_size==224 and body.relative is True
                and body.action_scale==100 and body.control_hz==10 and body.dt==.01,'original metered native env')
            row['status']='running';s=row['source_slot']
            episode(env,data['references'][s],data,p['layers'][s],p['tasks'][s],row,slots,visual,runtime,meter,store,
                planner_factory,permit_ram=permit_ram)
        except BaseException as exc:
            old.kernel.retain(row,exc,'episode')
            for n in slots:
                if n['episode_id']==row['episode_id'] and n['status']=='attempted':n.update(status='failed',error=repr(exc))
        finally:
            meter.region='closing'
            if env is not None:
                prior=store.finalizing;store.finalizing=True;entered=False
                def close():
                    nonlocal entered
                    entered=True;return env.close()
                try:meter.call('close',close)
                except BaseException as exc:
                    old.kernel.retain(row,exc,'close')
                    if not entered:
                        try:env.close()
                        except BaseException as secondary:old.kernel.retain(row,secondary,'emergency_close')
                finally:store.finalizing=prior
            store.put(row['episode_id']+'_episode.json',row);store.emit('episode_closed',receipt=row)
    try:
        report['final_check']=final_check();require(report['final_check']['model_unchanged'] is True,'frozen model unchanged')
        require(not meter.pending and not meter.secondary,'no unresolved physical/visual attempts')
        c=meter.counts
        require(c['public_step']['completed']==sum(r['warmup_returned']+r['control_returned'] for r in rows)
            and c['set_state']['completed']==2*c['reset']['completed']
            and c['physics_step']['completed']==2*c['reset']['completed']+10*c['public_step']['completed']
            and c['goal_setter']['completed']==c['reset']['completed']+c['goal_bind']['completed']
            and c['explicit_render']['completed']==sum(r['PNG'] for r in rows),'full native operation identities')
    except BaseException as exc:report['errors'].append(dict(stage='final_checks',error=repr(exc)))
    report['errors'].extend(meter.secondary)
    clean=not report['errors'] and all(r['first_error'] is None and r['status'] in
        ('completed20','contract_refused','native_stopped_warmup','native_stopped_control') for r in rows)
    report.update(status='shared_panel_recorded' if clean else 'partial_or_failed',counts=meter.counts,pending=meter.pending,
        completed20=sum(r['full_horizon'] for r in rows),refused=sum(r['status']=='contract_refused' for r in rows),
        equal_actual_execution_horizon=all(len({r['control_returned'] for r in rows if r['source_slot']==s})==1 for s in (0,1)))
    store.finalizing=True;store.emit('run_finished',status=report['status'],counts=meter.counts,nodes_sha256=digest(slots))
    report.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    return report


def run(prepared,expected,output,*,authorization):
    require(authorization==AUTH,'MAIN explicit known2 HOST CUDA and physics authorization')
    started=time.monotonic();p=verify(prepared,expected);output=exclusive(output)
    data=bridge.source_data(p)
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    data['visual_tasks']={s:dict(gn_task=enc.load_task(l['task_export'],l['task_sha256'])['gn_task'],model_bindings=enc.MODELS)
        for s,l in enumerate(p['layers'])}
    store=Store(output);store.put('execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    store.put('startup_roster.json',dict(rows=roster(),nodes=node_roster()))
    model=None;before=None;host=None;report=None;errors=[]
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def check():
            require(time.monotonic()-started<contract()['wall_seconds'] and
                torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and
                torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'whole wall/CUDA cap')
            store.check()
            require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                and not torch.is_autocast_enabled('cuda'),'TF32/autocast disabled')
        check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);check()
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original model weights')
        for module in model.modules():
            require(str(Path(inspect.getfile(type(module))).resolve()) in p['sources'],'pinned installed model class')
        def final():
            enc.low.frozen(model);after=signal.model_hash(model);require(before==after,'all model tensors/buffers unchanged')
            for f,h in p['sources'].items():require(sha(f)==h,'source/weight after SHA: '+f)
            return dict(model_unchanged=True,before=before,after=after,source_manifest_sha256=digest(p['sources']))
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            def precision():
                return dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
                    autocast=torch.is_autocast_enabled('cuda'),math=torch.backends.cuda.math_sdp_enabled(),
                    flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                    cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())
            numeric_before=precision()
            report=execute(p,data,store,old.Handler(None),dict(model=model,device=device,check_limits=check),old.native_factory,
                lambda t,s:native.NativeProgram(model,device,t,s),authorization=AUTH,final_check=final)
            report.update(precision_before=numeric_before,precision_after=precision())
            require(numeric_before==report['precision_after']==precision_contract(),'MATH guards before/after')
        torch.cuda.synchronize(device);check()
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        if report is None:
            report=data.get('live_report') or dict(schema=NAME+'.report',status='partial_or_failed',rows=native_roster(),
                nodes=node_roster(),errors=[],contract=contract(),fixed_denominator=8,node_denominator=24,
                panel_denominator=48,candidate_denominator=192,known_tasks=2,counts={},pending={})
        report['errors'].extend(errors)
        if errors:report['status']='partial_or_failed'
        report.update(prepared=str(Path(prepared).absolute()),prepare_sha256=expected,host=host,**FLAGS,
            source_manifest_sha256=digest(p['sources']),budget_match=contract()['budget_match'],
            per_episode_neural=node_accounting(report['nodes']),CUDA_peak_bytes=None)
        if model is not None:
            try:
                report.update(model_before=before,model_after=signal.model_hash(model),
                    CUDA_peak_bytes=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)])
                require(before==report['model_after'],'cleanup model unchanged')
            except BaseException as exc:
                report['errors'].append(dict(stage='cleanup_model',error=repr(exc)));report['status']='partial_or_failed'
        report['wall_seconds']=time.monotonic()-started
        if report['wall_seconds']>=contract()['wall_seconds']:
            report['errors'].append(dict(stage='wall',error='whole job wall cap'));report['status']='partial_or_failed'
        store.put('report.json',report,final=True)
    return dict(output=str(output),report_sha256=sha(output/'report.json'),status=report['status'])


def precision_contract():
    return dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)


def load_events(output,r):
    path=harden.regular(Path(output)/'journal.jsonl',contract()['output_bytes'])
    require(sha(path)==r['journal_sha256'],'own physical journal bytes')
    payload=path.read_bytes();require(payload.endswith(b'\n'),'complete journal line')
    events=[json.loads(line,object_pairs_hook=signal.unique_pairs) for line in payload.splitlines()];previous=None
    for i,e in enumerate(events):
        raw(e)
        require(type(e) is dict and e.get('kind') in bridge.FIELDS and set(e)==bridge.COMMON|bridge.FIELDS[e['kind']],
            'known exact physical event kind/fields')
        require(type(e['index']) is int and e['index']==i and e['previous_sha256']==previous and
            e['event_sha256']==old.source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'physical hash chain')
        previous=e['event_sha256']
    require(events and previous==r['journal_chain_tail'] and events[0]['kind']=='run_started'
        and events[-1]['kind']=='run_finished','physical terminal bookends')
    return events


def census(events,r):
    """Own8 census: no schema relabel and no unknown events, including partial prefixes."""
    ids=[x['episode_id'] for x in roster()];rows=r['rows'];nodes=r['nodes']
    require([x['episode_id'] for x in rows]==ids and [(x['episode_id'],x['boundary']) for x in nodes]==
        [(x['episode_id'],x['boundary']) for x in node_roster()],'all8/24 denominator')
    require(events[0]['contract']==contract() and events[-1]['counts']==r['counts'] and
        (events[-1]['status']==r['status'] or (r['status']=='partial_or_failed' and r['errors'] and
            events[-1]['status'] in ('shared_panel_recorded','partial_or_failed')))
        and events[-1]['nodes_sha256']==digest(nodes),'exact run bookends')
    resources=contract()['physical_caps']|contract()['visual_caps'];totals={k:dict(attempted=0,completed=0) for k in resources}
    per={eid:{k:dict(attempted=0,completed=0) for k in resources} for eid in ids}
    grouped={eid:[] for eid in ids};pending={};stack=[];serial=-1;active=0
    regions={eid:Counter() for eid in ids}
    for e in events[1:-1]:
        kind=e['kind']
        require(kind in bridge.FIELDS and set(e)==bridge.COMMON|bridge.FIELDS[kind]
            and kind not in ('run_started','run_finished'),'known exact nonbookend event')
        require(active<8,'no event after all episode closes');eid=ids[active]
        if kind=='episode_closed':
            require(e['receipt']==rows[active],'ordered unique durable close')
            grouped[eid].append(e);active+=1;continue
        if kind=='handler_event':
            require(e['namespace'] in [eid+f'_b{b:02}_encode' for b in (5,10,15,20)],'own encoding namespace')
        else:require(e['episode_id']==eid,'active episode identity')
        grouped[eid].append(e)
        if kind not in ('operation_attempted','operation_completed'):continue
        k=e['resource'];ident=e['operation_id'];units=e['units']
        require(k in resources and type(units) is int and units==1 and type(ident) is int
            and e['region'] in bridge.REGIONS[k],'single native unit and region')
        op={k:e[k] for k in bridge.FIELDS[kind]}
        if kind=='operation_attempted':
            require(ident==serial+1,'consecutive operation ID');serial=ident;pending[ident]=op;stack.append(ident);axis='attempted'
        else:
            require(stack and stack[-1]==ident and pending.pop(ident,None)==op,'nested matching return')
            stack.pop();axis='completed'
        totals[k][axis]+=1;per[eid][k][axis]+=1;regions[eid][k,e['region'],axis]+=1
    require(totals==r['counts'] and {str(k):v for k,v in pending.items()}=={str(k):v for k,v in r['pending'].items()},
        'all-resource counts/pending')
    require(all(0<=v['completed']<=v['attempted']<=resources[k] for k,v in totals.items()),'all caps')
    clean=r['status']=='shared_panel_recorded' and not r['errors'] and not r['pending']
    if not clean:return dict(complete=False,counts=totals,event_counts=dict(Counter(e['kind'] for e in events)))
    require(active==8 and not stack and not pending,'all episodes closed')
    for row in rows:
        eid=row['episode_id'];es=grouped[eid];by=defaultdict(list)
        for e in es:by[e['kind']].append(e)
        warm=row['warmup_returned'];control=row['control_returned'];enc=row['encodings']
        require(type(warm) is int and 0<=warm<=10 and type(control) is int and 0<=control<=20,'bounded returned steps')
        observations=[('warmup',b) for b in range(warm+1)]
        if row['initial_exact']:observations += [('plan',b) for b in range(control+1)]
        require([(e['receipt']['phase'],e['receipt']['step']) for e in by['fresh_observation']]==observations
            and [(e['phase'],e['step']) for e in by['guard_passed']]==observations and row['PNG']==len(observations),
            'every observation/guard/PNG')
        for ob,g in zip(by['fresh_observation'],by['guard_passed'],strict=True):
            require(ob['index']==ob['receipt']['observed_event_index']<g['index'] and
                type(g['canonical_agent_pixels']) is int and g['canonical_agent_pixels']>0,'positive guard follows actual observation')
        for kind,n in (('warmup_return',warm),('control_return',control),('authorized_dispatch',control)):
            require([e['step'] for e in by[kind]]==list(range(1,n+1)),'every returned/dispatched step')
        reached=[n for n in nodes if n['episode_id']==eid and n['status'] in ('sealed','refused')]
        require([e['actual_step'] for e in by['boundary_encoded']]==[x['step'] for x in enc] and
            [e['boundary'] for e in by['boundary_sealed']]==[n['boundary'] for n in reached],'every encoding/sealed node')
        require(row['checkpoints']==[e['receipt'] for e in by['fresh_observation'] if e['receipt']['phase']=='plan'
            and e['receipt']['step'] in (0,5,10,15,20)],'all native checkpoints')
        want=dict(constructor=1,reset=2,set_state=4,goal_render=2,goal_setter=2+int(row['initial_exact']),
            goal_bind=int(row['initial_exact']),public_step=warm+control,physics_step=4+10*(warm+control),
            explicit_render=len(observations),close=1,encoder=len(enc),projector=len(enc))
        require(per[eid]=={k:dict(attempted=v,completed=v) for k,v in want.items()},'independent complete native counts')
        reg={('constructor','setup'):1,('reset','setup'):2,('set_state','setup'):4,('goal_render','setup'):2,
            ('goal_setter','setup'):2,('goal_setter','goal_bind'):int(row['initial_exact']),('goal_bind','goal_bind'):int(row['initial_exact']),
            ('public_step','warmup'):warm,('physics_step','setup'):4,('physics_step','warmup'):10*warm,
            ('explicit_render','setup'):1,('explicit_render','warmup'):warm,('explicit_render','goal_bind'):int(row['initial_exact']),
            ('encoder','encoding'):len(enc),('projector','encoding'):len(enc),('close','closing'):1}
        for region,n in (('prefix',min(control,5)),('suffix',max(control-5,0))):
            reg.update({('public_step',region):n,('physics_step',region):10*n,('explicit_render',region):n})
        require(regions[eid]==Counter({(k,region,axis):n for (k,region),n in reg.items()
            for axis in ('attempted','completed') if n}),'full region census')
        require(len(by['initialization_intent'])==1 and by['initialization_intent'][0]['index']<by['operation_attempted'][0]['index'],
            'initialization before constructor')
        handlers=[]
        for encoded,event in zip(enc,by['boundary_encoded'],strict=True):
            ns=eid+f"_b{encoded['step']:02}_encode";local=[e for e in by['handler_event'] if e['namespace']==ns]
            require(len(local)==4 and event['encoding_sha256']==encoded['receipt_sha256'],'all visual entries/receipt')
            for i,(resource,action) in enumerate((('encoder','attempt'),('encoder','return'),('projector','attempt'),('projector','return'))):
                require(local[i]['event']==dict(kind='native_event',native=dict(kind='module_'+action,
                    entry=dict(index=i//2,image=0,resource=resource,completed=action=='return'))),'exact visual native ABI')
            ops=[e for e in es if e['kind'] in ('operation_attempted','operation_completed') and
                e['resource'] in ('encoder','projector') and local[0]['index']<e['index']<event['index']]
            require([(e['kind'],e['resource']) for e in ops]==[('operation_attempted','encoder'),('operation_completed','encoder'),
                ('operation_attempted','projector'),('operation_completed','projector')],'metered visual lifecycle')
            require(all(local[i]['index']<ops[i]['index']<(local[i+1]['index'] if i<3 else event['index']) for i in range(4)),
                'exact visual lifecycle interleave');handlers.extend(e['index'] for e in local)
        require(handlers==[e['index'] for e in by['handler_event']],'no stray visual event')
        for event,node in zip(by['boundary_sealed'],reached,strict=True):require(event['seal_sha256']==node['seal']['seal_sha256'],'all seal events')
        wanted=dict(initialization_intent=1,operation_attempted=sum(want.values()),operation_completed=sum(want.values()),
            fresh_observation=len(observations),guard_passed=len(observations),warmup_return=warm,
            authorized_dispatch=control,control_return=control,handler_event=4*len(enc),boundary_encoded=len(enc),
            boundary_sealed=len(reached),episode_closed=1)
        require(Counter(e['kind'] for e in es)==Counter({k:v for k,v in wanted.items() if v}),'full event-kind census including unknown/stray rejection')
    return dict(complete=True,counts=totals,per_episode=per,event_counts=dict(Counter(e['kind'] for e in events)))


def audit_encoding(output,p,row,encoded,ob):
    """Saved current PNG/receipt/latent closure; preprocessing checked by harden.audit_rgb."""
    eid=row['episode_id'];b=encoded['step'];ns=eid+f'_b{b:02}'
    er=harden.read_json(output/(ns+'_encoding.json'),1024**2);response=er['response'];request=er['request']
    require(old.source.digest(er)==encoded['receipt_sha256'] and er['actual_step']==b,'visual receipt digest')
    original=harden.regular(ob['path'],1024**2);alias=harden.regular(output/(ns+'_plan_05.png'),1024**2)
    require(original==output/(eid+f'_plan_{b:02}.png') and original.read_bytes()==alias.read_bytes()
        and sha(original)==sha(alias)==ob['file_sha256'] and er['mapping']['actual_step']==b
        and er['mapping']['original_png']=={k:ob[k] for k in ('path','file_sha256','pixel_sha256')},'current PNG not stale frame')
    require(encoded['latent']==response['latent'] and response['episode_id']==ns and
        response['request_sha256']==request['request_sha256']==old.source.digest({k:v for k,v in request.items() if k!='request_sha256'})
        and request['schema']==old.NAME+'.encode_request' and request['episode_id']==ns and request['observed_step']==5
        and request['task_export']==p['layers'][row['source_slot']]['task_export'] and
        request['png']==dict(path=str(alias),file_sha256=ob['file_sha256'],pixel_sha256=ob['pixel_sha256']), 'exact original encoder request ABI')
    require(response['source_verified'] is response['model_unchanged'] is True and response['device_type']=='cuda'
        and response['model_hash_before']==response['model_hash_after']==p['modelbinding']['tensor_sha256'],'frozen actual encoder')
    z=bridge.load_array(output,response['latent']);old.native(z);return z


def audit_episode(output,p,row,nodes,events,source):
    bridge.terminal_disposition(row)
    eid=row['episode_id'];s=row['source_slot'];base=p['tasks'][s];ref=source['references'][s]
    es=[e for e in events if e.get('episode_id')==eid]
    require(old.read(output/(eid+'_episode.json'))==row,'durable episode identity')
    intent=[e for e in es if e['kind']=='initialization_intent']
    require(len(intent)==1 and {k:v for k,v in intent[0].items() if k not in bridge.COMMON|{'episode_id'}}==
        bridge.initialization_intent(ref,source['plan']),'exact original recipe/root/history')
    warm=[e['receipt'] for e in es if e['kind']=='fresh_observation' and e['receipt']['phase']=='warmup']
    ws=[e for e in es if e['kind']=='warmup_return']
    for i,e in enumerate(ws):
        require(np.asarray(e['command'],np.float32).tobytes()==ref['raw_history'][i].tobytes() and
            type(e['terminated']) is type(e['truncated']) is bool and
            (i==len(ws)-1 or not(e['terminated'] or e['truncated'])),'warmup exact rawbits and stop')
    for k,b in enumerate((0,5,10)):
        if b>row['warmup_returned']:continue
        require(warm[b]['pixel_sha256']==old.array_sha(ref['history'][k]) and np.array_equal(warm[b]['state'],ref['states'][k])
            and warm[b]['snapshot']==ref['dynamics'][k],'original known warmup identity')
    obs=[e['receipt'] for e in es if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan']
    dispatch=[e for e in es if e['kind']=='authorized_dispatch'];returns=[e for e in es if e['kind']=='control_return']
    require(len(dispatch)==len(returns)==len(row['executed_raw'])==row['control_returned'],'actual full action denominator')
    for ob in warm+obs:frames.png(ob,output)
    if row['initial_exact']:
        require(row['warmup_returned']==10 and row['prefix']==bridge.prefix_binding(p['layers'][s]) and
            old.read(output/(eid+'_prefix_commit.json'))==row['prefix'],'fixed original prefix authority')
        require([e['command'] for e in dispatch[:5]]==row['prefix']['commands'][:min(5,len(dispatch))],'same prefix5')
    require(row['full_horizon']==(row['status']=='completed20'),'no first5 relabeled full20')
    if row['full_horizon']:require([e['step'] for e in row['encodings']]==[5,10,15,20],'all20 four real encoder inputs')
    cached=deepcopy(base['root_context']);audited=[];own=[n for n in nodes if n['episode_id']==eid]
    expected_schedule=[];expected_packets=[]
    for encoded in row['encodings']:
        b=encoded['step'];ns=eid+f'_b{b:02}';ob=next(o for o in obs if o['step']==b)
        z=audit_encoding(output,p,row,encoded,ob)
        if b==20:
            require(row['full_horizon'] and row['actual_Q20']==float(np.mean((z.astype(float)-np.asarray(base['goal']))**2)),
                'actual terminal model-Q separate from lookahead');continue
        require(b in BOUNDARIES,'fixed feedback boundaries')
        raw10=np.concatenate((ref['raw_history'],np.asarray(row['executed_raw'][:b],np.float32).reshape(-1,2)))[-10:]
        cached=moving.roll(cached,z,raw10,base);n=next(n for n in own if n['boundary']==b)
        req=n['request'];r=n['receipt'];validate_request(req,base)
        require(req['task']['root_context']==cached and req['raw10']==raw10.tolist(),'actual rolling history; no old root J')
        er=old.read(output/(ns+'_encoding.json'))
        expected_ob=dict(boundary=b,png_sha256=ob['file_sha256'],pixel_sha256=ob['pixel_sha256'],
            encoding_sha256=encoded['receipt_sha256'],observed_index=ob['observed_event_index'],
            encoded_index=next(e['index'] for e in es if e['kind']=='boundary_encoded' and e['actual_step']==b),
            observed_latent_sha256=old.packet_digest(z.tolist()))
        require(req['observation']==expected_ob and old.read(output/(ns+'_request.json'))==req,'model request exact RGB origin')
        node=dict(request=req,result=r,native_journal_sha256=n['native_journal_sha256'],native_journal_tail=n['native_journal_tail'])
        require(old.read(output/(ns+'_node.json'))==node,'durable node full record');expected_schedule.append(node)
        jp=harden.regular(output/ns/'events.jsonl',contract()['output_bytes']);require(sha(jp)==n['native_journal_sha256'],'native journal bytes')
        ne,tail=signal.journal(jp);require(tail==n['native_journal_tail'],'complete native journal tail')
        audited.append(dict(episode_id=eid,boundary=b,**audit_node(req,r,ne,base)))
        seal=n['seal'];expected=dict(schema=NAME+'.dispatch',episode_id=eid,boundary=b,request_sha256=digest(req),
            result_sha256=digest(r),commands=r['commands'],commands_sha256=digest(r['commands']),
            native_journal_sha256=n['native_journal_sha256'],status=r['status'],selected_horizon=r['selected_horizon'])
        expected['seal_sha256']=digest(expected)
        require(seal==expected and old.read(output/(ns+'_seal.json'))==seal and n['status']==r['status'],'own exact FIRST5 seal')
        sealed=next(e['index'] for e in es if e['kind']=='boundary_sealed' and e['boundary']==b)
        require(ob['observed_event_index']<expected_ob['encoded_index']<sealed,'actual observe/encode/compile/seal order')
        selected=[e for e in dispatch if e['packet_boundary']==b]
        if r['status']=='refused':
            require(not selected and row['status']=='contract_refused' and row['control_returned']==b and
                row['refusal']==dict(boundary=b,reason=r['refusal']),'no action after unsupported-policy refusal')
        else:
            require([e['step'] for e in selected]==list(range(b+1,b+1+len(selected))) and len(selected)<=5
                and [e['command'] for e in selected]==r['commands'][:len(selected)] and all(e['index']>sealed for e in selected),
                'only chosen FIRST5; no execution of extra H3 forecast')
            expected_packets.append(dict(start=b,end=b+5,seal_sha256=seal['seal_sha256'],commands=r['commands']))
    require(row['schedule_nodes']==expected_schedule and row['packets']==expected_packets,'all and only paid nodes/released packets')
    reached=[n for n in own if n['status'] in ('sealed','refused')]
    require([n['boundary'] for n in reached]==[e['step'] for e in row['encodings'] if e['step']!=20], 'no ghost node/no encoded node omission')
    for n in own:
        if n['status']=='not_reached':require(n==next(x for x in node_roster() if x['episode_id']==eid and x['boundary']==n['boundary']),
            'unreached remains exact null, not zero resource completion')
    for e,ret,ob,command in zip(dispatch,returns,obs[1:],row['executed_raw'],strict=True):
        require(e['step']==ret['step']==ob['step'] and e['command']==ret['command']==command
            and e['index']<ret['index'] and len(ob['state'])==7,'actual action/state7 identity')
        require(np.asarray(e['command'],np.float32).tobytes()==np.asarray(command,np.float32).tobytes()==
            np.asarray(ret['command'],np.float32).tobytes() and np.array_equal(ob['goal'],ref['goal'])
            and ob['goal_rgb_sha256']==old.array_sha(ref['goal_rgb']) and
            np.asarray(ob['goal_pose'],np.float64).tobytes()==np.asarray(warm[-1]['goal_pose'],np.float64).tobytes(), 'rawbit and unchanged native goal')
        checked=bridge.native_return_contract(ref['goal'],ob['state'],ret['reward'],ret['terminated'],ret['truncated'])
        metric=old.kernel.metric(ref['goal'],ob['state'])
        require(checked['native7_distance']==metric['native_distance'] and checked['native_success']==metric['native_success'], 'native reward/velocity/angle evaluation')
        calls=[x for x in es if x['kind']=='operation_attempted' and x['resource']=='public_step' and e['index']<x['index']<ret['index']]
        require(len(calls)==1,'one public attempt per dispatch')
        done=next(x for x in es if x['kind']=='operation_completed' and x['operation_id']==calls[0]['operation_id'])
        require(done['index']<ob['observed_event_index']<ret['index'] and
            sum(x['kind']=='operation_attempted' and x.get('resource')=='physics_step' and calls[0]['index']<x['index']<done['index'] for x in es)==10,
            'all ten native integration substeps before actual observation')
        if e['step']<row['control_returned']:require(not ret['terminated'] and not ret['truncated'],'no dispatch after native stop')
    if returns:
        metric=old.kernel.metric(ref['goal'],obs[-1]['state'])
        require(row['final_metrics']==metric and row['native_success']==metric['native_success'] and row['final_state']==obs[-1]['state']
            and row['last_observation']==obs[-1] and row['terminated']==returns[-1]['terminated'] and row['truncated']==returns[-1]['truncated'],
            'actual terminal or native earlystop preserved')
    return audited


def consume(output,report_sha,prepared,prepare_sha):
    """MAIN-only production strict entry; tests may use explicit synthetic saved fixtures."""
    output=Path(output).absolute();p=verify(prepared,prepare_sha)
    rp=harden.regular(output/'report.json',contract()['final_reserve_bytes']);require(sha(rp)==report_sha,'external report SHA')
    r=harden.read_json(rp,contract()['final_reserve_bytes'])
    require(r['schema']==NAME+'.report' and r['contract']==contract() and
        (r['fixed_denominator'],r['node_denominator'],r['panel_denominator'],r['candidate_denominator'],r['known_tasks'])==(8,24,48,192,2),
        'own8/24/48/192 all denominator')
    require([(x['episode_id'],x['source_slot'],x['mode']) for x in r['rows']]==
        [(x['episode_id'],x['source_slot'],x['mode']) for x in roster()] and
        [(x['episode_id'],x['boundary']) for x in r['nodes']]==[(x['episode_id'],x['boundary']) for x in node_roster()], 'complete roster order')
    require(r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha and
        old.read(output/'execution_binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'execution prepare binding')
    for k,v in FLAGS.items():require(r[k] is v,'engineering-only flags')
    if 'journal_sha256' not in r:
        require(r['status']=='partial_or_failed','startup failure never success')
        return dict(schema=NAME+'.consume',status='startup_partial_preserved',complete=False,
            prepared=str(Path(prepared).absolute()),report_sha256=report_sha,prepare_sha256=prepare_sha,fixed_denominator=8,node_denominator=24,
            panel_denominator=48,candidate_denominator=192,**FLAGS)
    events=load_events(output,r);counts=census(events,r)
    for n in r['nodes']:
        if n.get('resource_receipt') is None:continue
        prior=[x for x in r['nodes'] if x['episode_id']==n['episode_id'] and x['boundary']<n['boundary']]
        require(n['resource_receipt']['caps']==remaining_caps(n['mode'],prior),'exact nonrefundable predispatch remainder')
        jp=harden.regular(output/(n['episode_id']+f"_b{n['boundary']:02}")/'events.jsonl',contract()['output_bytes'])
        require(sha(jp)==n['native_journal_sha256'],'partial/full native journal bytes')
        es,tail=signal.journal(jp);require(tail==n['native_journal_tail'],'native chain tail')
        bridge.audit_resources(n['resource_receipt'],es)
    if not counts['complete']:
        return dict(schema=NAME+'.consume',status='saved_partial_preserved',complete=False,report_sha256=report_sha,
            prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha,counts=counts,rows=r['rows'],nodes=r['nodes'],fixed_denominator=8,node_denominator=24,
            panel_denominator=48,candidate_denominator=192,**FLAGS)
    rgb=harden.audit_rgb(output,r,p);source=bridge.source_data(p);audited=[]
    for row in r['rows']:audited.extend(audit_episode(output,p,row,r['nodes'],events,source))
    for s in (0,1):
        pair=[row for row in r['rows'] if row['source_slot']==s and row['prefix_exact']]
        ends=[next(e['receipt'] for e in events if e['kind']=='fresh_observation' and e.get('episode_id')==row['episode_id']
            and e['receipt']['phase']=='plan' and e['receipt']['step']==5) for row in pair]
        require(all(ob[k]==ends[0][k] for ob in ends[1:] for k in ('pixel_sha256','state','snapshot','goal','goal_rgb_sha256','goal_pose')),
            'independently initialized same-root prefix across all arms')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'] and
        r['source_manifest_sha256']==digest(p['sources']) and r['final_check']==dict(model_unchanged=True,
            before=r['model_before'],after=r['model_after'],source_manifest_sha256=digest(p['sources'])),'whole model/source unchanged')
    require(r['precision_before']==r['precision_after']==precision_contract(),'actual numeric guards before/after')
    require(type(r['wall_seconds']) in (int,float) and 0<=r['wall_seconds']<contract()['wall_seconds'] and
        type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=contract()[k]
        for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)), 'actual wall and CUDA memory caps')
    require(r['per_episode_neural']==node_accounting(r['nodes']) and r['budget_match']==contract()['budget_match'],'complete axis accounting')
    require(r['completed20']==sum(row['full_horizon'] for row in r['rows']) and
        r['refused']==sum(row['status']=='contract_refused' for row in r['rows']) and
        r['equal_actual_execution_horizon']==all(len({row['control_returned'] for row in r['rows'] if row['source_slot']==s})==1 for s in (0,1)),
        'actual20, refusals and equal execution measured, not assumed')
    for f,h in p['sources'].items():require(sha(f)==h,'all closure sources unchanged after strict')
    require(sha(rp)==report_sha,'report unchanged through saved strict')
    all20=all(row['full_horizon'] for row in r['rows'])
    return dict(schema=NAME+'.consume',status='saved_shared_panel_verified_NOT_admitted',complete=True,
        prepared=str(Path(prepared).absolute()),report_sha256=report_sha,prepare_sha256=prepare_sha,fixed_denominator=8,node_denominator=24,
        panel_denominator=48,candidate_denominator=192,completed20=r['completed20'],refused=r['refused'],
        all20=all20,equal_actual_execution_horizon=r['equal_actual_execution_horizon'],
        equal_paid_axes_full20=all20,refused_rows_preserved=[dict(episode_id=row['episode_id'],refusal=row['refusal'])
            for row in r['rows'] if row['status']=='contract_refused'],
        audited=audited,physical_census=counts,RGB_preprocessing=rgb,per_episode_neural=r['per_episode_neural'],
        native_metrics=[row['final_metrics'] for row in r['rows']],budget_match=contract()['budget_match'],
        natural_cheap_baseline=False,independent_neural_replay=False,**FLAGS)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','accounting','run','consume','strict'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt'):parser.add_argument('--'+k)
    parser.add_argument('--authorize-known-cuda-and-physics',action='store_true',help='MAIN only; real HOST CUDA/native execution')
    args=parser.parse_args(argv)
    required={'prepare':('output',),'verify':('prepared','prepare_sha'),'accounting':(),
        'run':('output','prepared','prepare_sha'),'consume':('output','prepared','prepare_sha','report_sha'),
        'strict':('output','prepared','prepare_sha','report_sha')}
    for k in required[args.command]:require(getattr(args,k) is not None,'required CLI --'+k.replace('_','-'))
    require(not args.receipt or args.command in ('consume','strict'),'receipt only for saved consumption')
    if args.command=='prepare':result=prepare(args.output)
    elif args.command=='verify':result=dict(verified=True,schema=verify(args.prepared,args.prepare_sha)['schema'])
    elif args.command=='accounting':result=accounting()
    elif args.command=='run':
        require(args.authorize_known_cuda_and_physics,'MAIN explicit run authorization')
        result=run(args.prepared,args.prepare_sha,args.output,authorization=AUTH)
    else:
        result=consume(args.output,args.report_sha,args.prepared,args.prepare_sha)
        if args.receipt:
            receipt=write_metadata(args.receipt,result,contract()['final_reserve_bytes'])
            result=dict(**receipt,status=result['status'],complete=result['complete'])
    print(raw(result).decode())
    if args.command=='run' and result['status']!='shared_panel_recorded':raise SystemExit(2)


if __name__=='__main__':main()
