"""Outcome-free 2x2 schedule/bank intervention; real compile/replay node + native episode.

Importing this module does not import torch or construct an environment. Only
explicit MAIN-authorized run loads HOST CUDA/native physics. The own eight-arm
launcher/strict never relabels a historical dispatcher or science receipt.
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
from experiments.spectral_reachability import agent_p3_commitment_controls_20261029_v1 as previous
from experiments.spectral_reachability import agent_p3_commitment_controls_20261029_v2 as bridge

NAME = 'agent_p3_effective_schedule_20261029_v1'
ROOT = previous.ROOT
live, core, signal = previous.live, previous.core, previous.signal
require, digest, sha = previous.require, previous.digest, previous.sha
moving, old, native = bridge.moving, bridge.old, live.native
harden, frames = bridge.harden, bridge.frames
raw = core.canonical
AUTH = 'MAIN_KNOWN2_EFFECTIVE_SCHEDULE_V1_CUDA_AND_PHYSICS'
SCHEDULES = ('commit15', 'feedback5')
BANKS = dict(endpoints=(0., 1/3, 2/3, 1.), midpoints=(1/8, 3/8, 5/8, 7/8))
ARMS = tuple(s+'_'+b for s in SCHEDULES for b in BANKS)
SEEN_REPORT = ROOT/'results/agent_p3_commitment_controls_20261029_v2_MAIN01/report.json'
SEEN_REPORT_SHA = '3e02de2f957253830a9122e62926563ab6ff80b8eb3c61207d4d1c367bbe044f'
SEEN_STRICT = ROOT/'results/agent_p3_commitment_controls_20261029_v2_consume_MAIN01.json'
SEEN_STRICT_SHA = 'aa5c34faabae5205dbeb7dd216fa5ff0631d16239eae25de06bf2b9a2084e94a'


def split_arm(arm):
    require(type(arm) is str and arm in ARMS, 'predeclared schedule x fraction bank')
    return arm.split('_', 1)


def contract():
    return dict(schema=NAME+'.contract', arms=list(ARMS), known_roots=[0, 1],
        episodes=8, node_slots=24, boundaries=[5, 10, 15], raw_target=20, warmup=10,
        prefix=5, schedules=dict(commit15='select H3 at5; retain through10/15',
            feedback5='select fresh H1 at5/10/15'), fractions={k:list(v) for k,v in BANKS.items()},
        candidates=4, signal='pay fresh H1+H3 and spectra at EVERY reached boundary',
        scheduler='fixed ex ante; no chi/gain/outcome-dependent branching',
        action_map=live.native.rule(), incumbent_retention=False,
        bank_intervention='fraction supports disjoint; endpoint bank includes zero, midpoint bank does not',
        raw_support='may overlap under degeneracy/float32; record, never repair or resample',
        feedback='actual RGB three-latent rolling + last10 actually returned raw, normalized once',
        success='unchanged native earlystop; no pad/truncate/reselect',
        complete20_predict_per_episode=51, total_predict_if_full20=408,
        total_J_if_all_boundaries=48, total_VJP_if_all_boundaries=9216,
        budget_match='notmatched', equal_predict_if_full20=True, equal_total_compute=False,
        equal_FLOPs=False, equal_wall=False, FLOPs=None,
        historical_roots_seen=True, independent_confirmation=False, freshness=False,
        P1_admission=False, P3_admission=False, C3=False, scientific_admission=False,
        scope='known-root diagnostic scheduler AND candidate-support factorial, not adaptive-policy superiority',
        whole_job_launcher_included=True, whole_job_strict_included=True,
        physical_caps=dict(constructor=8,reset=24,set_state=48,goal_render=24,
            goal_setter=32,goal_bind=8,public_step=240,physics_step=2448,explicit_render=256,close=8),
        visual_caps=dict(encoder=32,projector=32), wall_seconds=3600,output_bytes=1024**3,
        final_reserve_bytes=128*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        TF32=False,autocast=False,SDP='MATH')


def costs(arm, boundary):
    schedule, _ = split_arm(arm)
    require(type(boundary) is int and boundary in live.BOUNDARIES, 'fixed boundary')
    if schedule == 'commit15':
        return live.costs(3, 4, schedule=True) if boundary == 5 else live.costs()
    return live.costs(1, 4, schedule=True)


def caps(arm):
    return {k:sum(costs(arm,b)[k] for b in live.BOUNDARIES) for k in core.AXES}


def accounting():
    return dict(contract=contract(), per_episode=[dict(arm=a, cap=caps(a),
        nodes=[dict(boundary=b, exact_if_completed=costs(a,b)) for b in live.BOUNDARIES],
        attempted=None, completed=None, failed_attempts_refunded=False,
        native_public_target=30, encoder_target=4, projector_target=4,
        actual_control=None, wall_ns=None, FLOPs=None) for a in ARMS],
        physical_and_visual_caps=contract()['physical_caps']|contract()['visual_caps'],
        before_native_stop_costs_are_not_imputed=True)


def roster():
    return [dict(episode_id=f's{s}_{a}', source_slot=s, mode=a,
        status='not_started', native_success=None, actual_control_returned=None,
        nodes=[dict(boundary=b,status='not_reached') for b in live.BOUNDARIES])
        for s in (0,1) for a in ARMS]


def native_roster():
    """Executable moving.packet row ABI, distinct from the prepare-only census."""
    rows=[]
    for entry in roster():
        row=deepcopy(bridge.template.roster()[0])
        row.update(episode_id=entry['episode_id'],source_slot=entry['source_slot'],mode=entry['mode'],
            prefix_exact=False,commitments=[],schedule_nodes=[],refusal=None,actual_Q20=None)
        rows.append(row)
    return rows


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'),
        ROOT/'innerdocs'/(NAME+'.md')]


def build():
    # Original tasks/model/normalizer are fully rebuilt by the original authority.
    # The newer historical receipt is accepted explicitly, NOT re-consumed here.
    accepted = previous.build()
    for path, h in ((SEEN_REPORT, SEEN_REPORT_SHA), (SEEN_STRICT, SEEN_STRICT_SHA)):
        require(sha(path) == h, 'pinned prior known-root evidence bytes')
    seen = signal.read(SEEN_STRICT)
    require(seen['complete'] is True, 'historical engineering receipt, not science authority')
    files = own_files()+[Path(previous.__file__), Path(bridge.__file__)]
    sources = dict(accepted['bindings'])
    for path in files+[SEEN_REPORT, SEEN_STRICT]:sources[str(path.resolve())] = sha(path)
    return dict(schema=NAME+'.prepare', contract=contract(), accounting=accounting(),
        sources=sources, tasks=accepted['tasks'], layers=accepted['layers'],
        policy=accepted['policy'], modelbinding=accepted['modelbinding'],
        expected_model_hash=accepted['expected_model_hash'], source_manifest=accepted['source_manifest'],
        authority=dict(original_fullcommit=accepted['authority'], seen_report_sha256=SEEN_REPORT_SHA,
            seen_receipt_sha256=SEEN_STRICT_SHA, seen_receipt_reconstructed_here=False,
            no_native_labels_in_scheduler=True), roster=roster())


def verify(path, expected):
    core.hash_value(expected)
    require(sha(path)==expected, 'external prepare SHA')
    p=signal.read(path)
    require(p==build() and sha(path)==expected, 'complete source/task/protocol reconstruction')
    return p


def prepare(output):
    path=Path(output).absolute()
    require(path.parent==ROOT/'results' and path.name.startswith(NAME) and
        path.resolve()==path and not path.exists(), 'exclusive new prepare')
    value=core.canonical(build());require(len(value)<=16*1024**2, '16MiB prepare cap')
    with path.open('xb') as f:
        require(f.write(value)==len(value), 'whole metadata write')
        f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent)
    return dict(path=str(path), sha256=sha(path), whole_job_launchable=True)


def make_request(base, context, *, slot, arm, boundary, raw10, observation, commitment=None):
    split_arm(arm)
    identity=dict(episode_id=f's{slot}_{arm}',source_slot=slot,mode=arm,
        boundary=boundary,raw10=np.asarray(raw10,np.float32).tolist(),observation=deepcopy(observation))
    return dict(schema=NAME+'.node',base_task_sha256=digest(base),
        task=live.local_task(base,context,identity),commitment=deepcopy(commitment),**identity)


def validate_request(req, base):
    core.keys(req, ('schema','episode_id','source_slot','mode','boundary','raw10',
        'observation','task','commitment','base_task_sha256'))
    schedule,_=split_arm(req['mode'])
    require(req['schema']==NAME+'.node' and type(req['source_slot']) is int and
        req['source_slot'] in (0,1) and req['episode_id']==f"s{req['source_slot']}_{req['mode']}",
        'own exact episode and source identity')
    require(type(req['boundary']) is int and req['boundary'] in live.BOUNDARIES,
        'predeclared actual boundary')
    require((req['commitment'] is not None)==(schedule=='commit15' and req['boundary']>5),
        'fixed schedule cannot silently drop a commitment or replay H3 late')
    t=core.validate_task(req['task'])
    identity={k:deepcopy(req[k]) for k in
        ('episode_id','source_slot','mode','boundary','raw10','observation')}
    require(req['base_task_sha256']==digest(base) and
        t==live.local_task(base,t['root_context'],identity), 'original task/current RGB context')
    # Pure structural validation view only; never dispatched or receipted as old.
    view=deepcopy(req);view.update(schema=live.NAME+'.node',mode='fixed',
        episode_id=f"s{req['source_slot']}_fixed")
    live.validate_request(view)
    if req['commitment'] is not None:
        require(req['commitment']['start']==5 and req['commitment']['end']==20,
            'only original full15 commitment')
    return t


def generate(task, response, bank, ledger):
    require(bank in BANKS, 'fixed nonnested bank')
    solved=live.native.generate(task,response,ledger)
    direction=np.asarray(solved['solve']['params'], np.float64)
    all_actions={name:[(f*direction/solved['divisor']).astype(np.float32)
        .reshape(5*response['horizon'],2).tolist() for f in fractions]
        for name,fractions in BANKS.items()}
    bits={name:[np.asarray(a,np.float32).tobytes() for a in actions]
        for name,actions in all_actions.items()}
    overlap=[dict(endpoint_index=i,midpoint_index=j) for i,a in enumerate(bits['endpoints'])
        for j,b in enumerate(bits['midpoints']) if a==b]
    return dict(solve=solved['solve'],divisor=solved['divisor'],fractions=list(BANKS[bank]),
        bank=bank,actions=all_actions[bank],response_sha256=digest(response),
        generator_sha256=sha(live.native.__file__),fraction_support_disjoint=True,
        counterfactual_raw_support_overlap=overlap,
        support_note='same-node/ridge arithmetic only; no extra predictions or outcomes')


def remaining_caps(arm, results):
    c=caps(arm)
    for r in results:
        rr=r.get('resource_receipt')
        if rr is None:continue  # fixed unreached node census, not a refunded attempted node
        core.audit_ledger(rr)
        for k in c:c[k]-=rr['attempted'][k]
    require(all(v>=0 for v in c.values()), 'nonrefundable arm remainder')
    return c


def plan_node(req, base, factory, store, ledger, *, permit_ram=False):
    """Actual frozen native compilation and all-candidate whole/segmented replay."""
    t=validate_request(req,base);arm=req['mode'];schedule,bank=split_arm(arm)
    backend=factory(t,store)
    require(getattr(backend,'kind',None)=='HOST_CUDA' or
        (permit_ram is True and getattr(backend,'kind',None)=='RAM_FIXTURE'), 'HOST CUDA only; explicit synthetic fixtures')
    require(not ledger.events and all(0<=ledger.caps[k]<=caps(arm)[k] for k in core.AXES),
        'fresh bounded ledger')
    store.event(dict(kind=NAME+'.node_request',request=deepcopy(req)))
    out=dict(schema=NAME+'.node_result',request_sha256=digest(req),status='attempted',
        responses=[],diagnostic=None,decision=None,bank=None,candidates=[],winner=None,
        commands=None,selected_path=None,error=None,resource_receipt=None)
    try:
        for h in (1,3):
            ledger.scope=dict(slot=req['source_slot'],horizon=h)
            request=core.request(t,dict(context=t['root_context']),operation='compile_finite_horizon_J',horizon=h)
            response=ledger.call('signal_compile',lambda request=request:backend.signal(request,ledger))
            out['responses'].append(response);store.event(dict(kind='local_signal',response=response))
        ledger.scope=dict(slot=req['source_slot'],phase='diagnostic')
        out['diagnostic']=signal.spectra(t,out['responses'],svd=lambda a:
            ledger.call('array_svd',lambda:np.linalg.svd(a,full_matrices=False)))
        if req['commitment'] is not None:
            require(ledger.attempted==ledger.completed==costs(arm,req['boundary']), 'paid pending signals')
            out['status']='observed_commitment';return out
        h=3 if schedule=='commit15' else 1
        ledger.scope=dict(slot=req['source_slot'],phase='schedule')
        out['decision']=ledger.call('schedule',lambda:dict(horizon=h,candidates=4,bank=bank,
            schedule=schedule,source=NAME+'.fixed_no_signal_branch'))
        out['candidates']=[dict(index=i,status='not_started',actions=None,whole=None,
            segmented=None,cost=None) for i in range(4)]
        ledger.scope=dict(slot=req['source_slot'],mode=arm,horizon=h,phase='program')
        response=next(r for r in out['responses'] if r['horizon']==h)
        out['bank']=ledger.call('program_compile',lambda:generate(t,response,bank,ledger))
        for row in out['candidates']:
            row.update(status='attempted',actions=out['bank']['actions'][row['index']])
            ledger.scope=dict(slot=req['source_slot'],mode=arm,index=row['index'],horizon=h,phase='program')
            for segmented,key in ((False,'whole'),(True,'segmented')):
                row[key]=ledger.call('replay',lambda segmented=segmented:
                    backend.replay(t['root_context'],row['actions'],segmented,ledger))
            require(row['whole']==row['segmented'], 'full candidate exact segmented replay')
            terminal=core.checked_path(t,t['root_context'],row['actions'],row['whole'])
            row['cost']=ledger.call('metric',lambda:core.model_cost(t,terminal));row['status']='returned'
        winner=min(out['candidates'],key=lambda r:(r['cost'],r['index']))
        ledger.scope=dict(slot=req['source_slot'],mode=arm,index=winner['index'],horizon=h,phase='selected')
        path=ledger.call('replay',lambda:backend.replay(t['root_context'],winner['actions'],False,ledger))
        require(path==winner['whole'], 'independent selected replay')
        require(ledger.attempted==ledger.completed==costs(arm,req['boundary']), 'exact full node axes')
        out.update(status='sealed',winner=deepcopy(winner),commands=deepcopy(winner['actions']),selected_path=path)
    except BaseException as exc:
        out.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        out['resource_receipt']=ledger.receipt()
        store.event(dict(kind=NAME+'.node_final',result=deepcopy(out)))
    return out


class Session:
    """One arm's actual feedback boundary/packet ACK gate; no evaluator input."""
    def __init__(self,base,slot,arm):
        require(type(slot) is int and slot in (0,1), 'fixed known root')
        split_arm(arm);self.base=deepcopy(base);self.slot=slot;self.arm=arm
        self.next_boundary=5;self.active=None;self.results=[];self.requests=[];self.stopped=False

    def plan(self,context,raw10,observation,factory,store,*,permit_ram=False):
        require(not self.stopped and self.next_boundary in live.BOUNDARIES and
            (self.active is None or self.active.pending is None), 'observe once after actual ACK')
        pending=None if self.active is None else self.active.active(self.next_boundary)
        req=make_request(self.base,context,slot=self.slot,arm=self.arm,boundary=self.next_boundary,
            raw10=raw10,observation=observation,commitment=pending)
        ledger=core.Ledger(remaining_caps(self.arm,self.results),sink=store.event)
        self.requests.append(req)
        try:r=plan_node(req,self.base,factory,store,ledger,permit_ram=permit_ram)
        except BaseException:
            self.stopped=True
            self.results.append(dict(status='failed',resource_receipt=ledger.receipt()))
            raise
        self.results.append(r)
        if self.active is None:self.active=previous.Commitment(req,r)
        commands=self.active.packet(self.next_boundary)
        return req,r,commands

    def acknowledge(self,returned,*,terminated=False,truncated=False):
        require(self.active is not None and not self.stopped, 'one outstanding actual packet')
        before=self.active.returned
        self.active.acknowledge(returned,terminated=terminated,truncated=truncated)
        self.next_boundary+=self.active.returned-before;self.stopped=terminated or truncated
        if self.active.active(self.next_boundary) is None:self.active=None


def audit_node(req, result, events, base):
    """Saved NODE validation only, not a native/RGB whole-job certification.

    Reuses policy-free native module/trajectory arithmetic, not an old producer
    receipt or an old scheduler. The caller must authenticate journal bytes via
    signal.journal and its externally sealed SHA/tail before calling this.
    """
    t=validate_request(req,base);rr=result['resource_receipt'];arm=req['mode']
    require(result['schema']==NAME+'.node_result' and result['request_sha256']==digest(req)
        and result['error'] is None, 'own nonfailure result/request identity')
    bridge.audit_resources(rr,events)
    allowed={'resource_attempt','resource_return','io_input','io_return','origin','packet','boundary',
        'nominal','AD_terminal','jacobian','direction','raw_backward','program_packet','program_boundary',
        'local_signal',NAME+'.node_request',NAME+'.node_final'}
    require(all(e['kind'] in allowed for e in events) and
        events[0]==dict(kind=NAME+'.node_request',request=req) and
        events[-1]==dict(kind=NAME+'.node_final',result=result), 'complete node event vocabulary and endpoints')
    require(sum(e['kind']==NAME+'.node_request' for e in events)==
        sum(e['kind']==NAME+'.node_final' for e in events)==1, 'exact single dispatch')
    for e in events:
        for b in e.get('arrays',{}).values():require(signal.unblob(b).dtype==np.float32, 'native float32 arrays')
    io={'predict':'latent','action_encoder':'0','predictor':'0','pred_proj':'0'}
    calls=[op for op in rr['events'] if op['resource'] in (*io,'grad')]
    require(sum(e['kind'] in ('io_input','io_return') for e in events)==2*len(calls), 'complete actual module IO census')
    for op in rr['events']:
        linked=[e for e in events if e.get('operation_index')==op['index']]
        if op['resource'] in (*io,'grad'):
            require([e['kind'] for e in linked]==['io_input','io_return'] and
                all(e['scope']==op['scope'] for e in linked), 'exact linked module input/return')
            if op['shape'] is not None:
                require(linked[0]['arrays'][op['resource']+':'+io[op['resource']]]['shape']==op['shape'], 'paid input dimensions')
        elif op['resource']=='raw_backward':require([e['kind'] for e in linked]==['raw_backward'], 'paid raw callback')
        else:require(not linked, 'no invented module IO')
    require([e['response'] for e in events if e['kind']=='local_signal']==result['responses'] and
        [r['horizon'] for r in result['responses']]==[1,3], 'all and only fresh H1/H3 signals')
    for response in result['responses']:
        h=response['horizon'];scope=dict(slot=req['source_slot'],horizon=h)
        local=[e for e in events if e.get('scope')==scope]
        require(response['binding_sha256']==digest(t['binding']) and
            response['parent_context_sha256']==digest(t['root_context']) and
            response['producer_sha256']==sha(signal.__file__), 'current actual node derivative binding')
        n=core.finite(response['nominal'],(192,),True)
        j=core.finite(response['jacobian'],(192,10*h),True)
        for phase in ('nominal','AD'):
            terminal=bridge.audit_trajectory(t,local,phase,h,np.zeros((5*h,2),np.float32),None,program=False)
            require(np.array_equal(terminal,n), 'nominal and AD actual endpoint')
        def values(kind):return [signal.unblob(e['arrays']['value']) for e in local if e['kind']==kind]
        require(len(values('nominal'))==len(values('AD_terminal'))==len(values('jacobian'))==1 and
            np.array_equal(values('nominal')[0],n) and np.array_equal(values('AD_terminal')[0],n) and
            np.array_equal(values('jacobian')[0],j), 'complete native nominal and Jacobian arrays')
        require([e['direction'] for e in local if e['kind']=='direction']==list(range(192)), 'all scalar VJP directions')
        gi=[e for e in local if e['kind']=='io_input' and 'grad:basis' in e['arrays']]
        go=[e for e in local if e['kind']=='io_return' and 'grad:output' in e['arrays']]
        rw=values('raw_backward');require(len(gi)==len(go)==len(rw)==192, 'full gradient/raw callbacks')
        for i,(a,b,c) in enumerate(zip(gi,go,rw,strict=True)):
            core.keys(a['arrays'],('grad:basis','grad:output','grad:raw'));core.keys(b['arrays'],('grad:output',))
            require(np.array_equal(signal.unblob(a['arrays']['grad:basis']),np.eye(192,dtype=np.float32)[i]) and
                np.array_equal(signal.unblob(a['arrays']['grad:output']),n) and
                np.array_equal(signal.unblob(a['arrays']['grad:raw']),np.zeros(10*h,np.float32)) and
                np.array_equal(signal.unblob(b['arrays']['grad:output']),j[i]) and np.array_equal(c,j[i]), 'all raw derivative rows')
    require(signal.spectra(t,result['responses'])==result['diagnostic'], 'diagnostic recomputation, not selection input')
    if req['commitment'] is not None:
        require(result['status']=='observed_commitment' and result['decision'] is None and
            result['bank'] is None and result['candidates']==[] and result['winner'] is None and
            result['selected_path'] is None and result['commands'] is None, 'no replacement of actual commitment')
    else:
        schedule,bank=split_arm(arm);h=3 if schedule=='commit15' else 1
        require(result['decision']==dict(horizon=h,candidates=4,bank=bank,schedule=schedule,
            source=NAME+'.fixed_no_signal_branch'), 'ex ante fixed schedule')
        dummy=core.Ledger(caps(arm));dummy.scope=dict(saved_recompute=True)
        rebuilt=generate(t,next(r for r in result['responses'] if r['horizon']==h),bank,dummy)
        require(result['bank']==rebuilt and len(result['candidates'])==4, 'all four fixed bank candidates')
        for i,row in enumerate(result['candidates']):
            require(row['index']==i and row['status']=='returned' and row['actions']==rebuilt['actions'][i]
                and row['whole']==row['segmented'], 'literal candidate full path')
            scope=dict(slot=req['source_slot'],mode=arm,index=i,horizon=h,phase='program')
            local=[e for e in events if e.get('scope')==scope]
            for phase in ('whole','segmented'):
                bridge.audit_trajectory(t,local,phase,h,row['actions'],row[phase],program=True)
            terminal=core.checked_path(t,t['root_context'],row['actions'],row['whole'])
            require(row['cost']==core.model_cost(t,terminal), 'original terminal Q')
        winner=min(result['candidates'],key=lambda r:(r['cost'],r['index']))
        require(result['status']=='sealed' and result['winner']==winner and result['commands']==winner['actions']
            and result['selected_path']==winner['whole'], 'stable model-only selected program')
        local=[e for e in events if e.get('scope')==dict(slot=req['source_slot'],mode=arm,
            index=winner['index'],horizon=h,phase='selected')]
        bridge.audit_trajectory(t,local,'whole',h,result['commands'],result['selected_path'],program=True)
    require(rr['attempted']==rr['completed']==costs(arm,req['boundary']), 'every completed node cost axis')
    return dict(node_verified=True,resources=rr['completed'],whole_job_verified=False,
        P1_admission=False,P3_admission=False,C3=False)


def episode(env, ref, data, layer, base, row, visual, runtime, meter, store, factory, *, permit_ram=False, slots=None):
    """Real RGB/native hookup, no global patch, old episode, or receipt relabeling.

    Called by our eight-arm execute; direct callers must own the same enclosing
    constructor/close, caps, model/precision/source and whole-job strict checks.
    """
    require(row['mode'] in ARMS and row['episode_id']==f"s{row['source_slot']}_{row['mode']}", 'own arm row')
    moving=bridge.moving
    if not moving.setup(env,ref,data['plan'],row,meter,store):return
    prefix=bridge.prefix_binding(layer);row['prefix']=prefix
    store.put(row['episode_id']+'_prefix_commit.json',prefix)
    obs=moving.packet(env,prefix['commands'],start=0,row=row,ref=ref,meter=meter,store=store)
    if obs is None:return
    row['completed_packets']+=1;row['prefix_exact']=True
    match={k:obs[k] for k in ('pixel_sha256','state','snapshot','goal','goal_rgb_sha256','goal_pose')}
    paired=data.setdefault(NAME+'.paired_prefix',{})
    source=row['source_slot']
    if source not in paired:paired[source]=deepcopy(match)
    else:require(paired[source]==match, 'independently initialized same root and prefix')
    context=deepcopy(base['root_context']);session=Session(base,source,row['mode'])
    row['schedule_nodes']=[]
    for boundary in live.BOUNDARIES:
        slot=next(x for x in slots if x['episode_id']==row['episode_id'] and x['boundary']==boundary) if slots is not None else None
        if slot is not None:slot['status']='attempted'
        meter.region='encoding'
        z,observation=moving.encode_boundary(visual,runtime,data['visual_tasks'][source],layer,row,
            obs,env.unwrapped,meter,store,boundary)
        raw10=np.concatenate((ref['raw_history'],np.asarray(row['executed_raw'],np.float32).reshape(-1,2)))[-10:]
        context=moving.roll(context,z,raw10,base)
        ns=row['episode_id']+f'_b{boundary:02}'
        journal=live.NativeJournal(store,ns,runtime['check_limits']);state=bridge.old.state_seal(env.unwrapped)
        meter.region='compiling'
        try:
            req,result,commands=session.plan(context,raw10,observation,factory,journal,permit_ram=permit_ram)
            require(bridge.old.state_seal(env.unwrapped)==state, 'planner cannot mutate native state or goal')
        finally:
            journal.close()
            node=dict(request=session.requests[-1] if session.requests else None,
                result=session.results[-1] if session.results else None,
                native_journal_sha256=sha(journal.path),native_journal_tail=journal.tail)
            row['schedule_nodes'].append(node);store.put(ns+'_node.json',node)
            if slot is not None:
                slot.update(request=node['request'],receipt=node['result'],
                    native_journal_sha256=node['native_journal_sha256'],native_journal_tail=node['native_journal_tail'],
                    resource_receipt=node['result']['resource_receipt'] if node['result'] is not None else None)
        store.put(ns+'_request.json',req)
        seal=dict(schema=NAME+'.dispatch',episode_id=row['episode_id'],boundary=boundary,
            request_sha256=digest(req),plan_sha256=digest(result),commands=result['commands'],
            commands_sha256=digest(result['commands']),native_journal_sha256=node['native_journal_sha256'],
            status=result['status'],commitment=req['commitment'],
            commitment_sha256=session.active.seal)
        seal['seal_sha256']=digest(seal)
        if slot is not None:slot.update(status=result['status'],seal=seal)
        store.put(ns+'_seal.json',seal)
        store.emit('boundary_sealed',episode_id=row['episode_id'],boundary=boundary,seal_sha256=seal['seal_sha256'])
        if result['status']=='sealed':
            row['commitments'].append(dict(start=boundary,end=boundary+len(result['commands']),
                seal_sha256=session.active.seal,commands=deepcopy(result['commands'])))
        obs=moving.packet(env,commands,start=boundary,row=row,ref=ref,meter=meter,store=store)
        returned=np.asarray(row['executed_raw'][boundary:],np.float32).reshape(-1,2)
        # Actual native stop flags, never a model/outcome-based replan input.
        session.acknowledge(returned,terminated=row['terminated'],truncated=row['truncated'])
        if row['control_returned']==boundary+5:row['completed_packets']+=1
        if obs is None:
            if row['control_returned']==20:obs=row['last_observation'];break
            return
    require(row['control_returned']==20, 'whole target20, no padding')
    terminal,_=moving.encode_boundary(visual,runtime,data['visual_tasks'][source],layer,row,
        obs,env.unwrapped,meter,store,20)
    row.update(full_horizon=True,status='completed20',actual_Q20=core.model_cost(base,
        dict(context,latent_history=context['latent_history'][1:]+[terminal.tolist()])))


# Whole-job integration: frozen policy-free native/RGB primitives only.
BOUNDARIES=live.BOUNDARIES
MODES=ARMS
load_array=bridge.load_array
prefix_binding=bridge.prefix_binding
initialization_intent=bridge.initialization_intent
native_return_contract=bridge.native_return_contract
terminal_disposition=bridge.terminal_disposition
audit_resources=bridge.audit_resources
FIELDS,COMMON,REGIONS=bridge.FIELDS,bridge.COMMON,bridge.REGIONS


def node_roster():
    return [dict(episode_id=r['episode_id'],source_slot=r['source_slot'],mode=r['mode'],
        boundary=b,status='not_reached',request=None,receipt=None,error=None,resource_receipt=None)
        for r in roster() for b in BOUNDARIES]


class Store(bridge.Store):
    def __init__(self,path):
        self.path=exclusive(path);self.finalizing=False
        require(shutil.disk_usage(self.path.parent).free>=contract()['disk_floor_bytes']+contract()['output_bytes'],
            'whole1GiB output plus200MiB floor')
        self.path.mkdir();signal.fsync_dir(self.path.parent);self.index=0;self.previous=None
    def check(self,size=4096):
        used=sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file())
        cap=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
        require(used+size<=cap and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,
            'own eight-arm whole-output/final-reserve/disk-floor caps')


class Meter(bridge.Meter):
    def __init__(self,store,check):
        old.Meter.__init__(self,store,check)
        self.caps=contract()['physical_caps']|contract()['visual_caps']
        self.counts={k:dict(attempted=0,completed=0) for k in self.caps}


def exclusive(path):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and
        p.resolve()==p and not p.exists(),'new exclusive effective-schedule result')
    return p


def source_data(p):
    return bridge.source_data(p)


def node_accounting(nodes):
    out=[]
    for s in (0,1):
        for mode in MODES:
            selected=[n for n in nodes if (n['source_slot'],n['mode'])==(s,mode)]
            receipts=[n['resource_receipt'] for n in selected if n.get('resource_receipt') is not None]
            signal_completed={k:sum(op['delta'][k] for r in receipts for op in r['events']
                if op['status']=='returned' and op['scope'].get('phase') not in ('program','selected')
                and 'index' not in op['scope']) for k in core.AXES}
            completed={k:sum(r['completed'][k] for r in receipts) for k in core.AXES}
            out.append(dict(source_slot=s,mode=mode,requested_nodes=3,receipts=len(receipts),
                sealed_nodes=sum(n['status']=='sealed' for n in selected),
                attempted={k:sum(r['attempted'][k] for r in receipts) for k in core.AXES},
                completed=completed,signal_completed=signal_completed,
                program_completed={k:completed[k]-signal_completed[k] for k in core.AXES},
                caps=caps(mode),FLOPs=None,CUDA_ns=None,backward_engine_calls=None,
                inclusive_operation_wall_ns={resource:sum(r['inclusive_operation_wall_ns'].get(resource,0)
                    for r in receipts) for resource in sorted({k for r in receipts for k in r['inclusive_operation_wall_ns']})},
                wall_semantics='nested inclusive; never summed into runtime or FLOPs'))
    return out

def execute(p,data,store,visual,runtime,factory,planner_factory,*,authorization,final_check,permit_ram=False):
    require(authorization==AUTH and p['contract']==contract(),'MAIN explicit known2 authorization/contract')
    require(set(runtime)=={'model','device','check_limits'} and getattr(runtime['device'],'type',None)=='cuda',
        'explicit single-process HOST CUDA runtime')
    rows=native_roster();slots=node_roster();meter=Meter(store,runtime['check_limits']);errors=[]
    report=dict(schema=NAME+'.report',status='running',rows=rows,nodes=slots,errors=errors,contract=contract(),
        fixed_denominator=8,node_denominator=24,known_tasks=2,P1_admission=False,P3_admission=False,
        scientific_admission=False,freshness=False)
    report.update(counts=meter.counts,pending=meter.pending)
    data['live_report']=report
    store.put('roster.json',dict(rows=rows,nodes=slots));store.emit('run_started',contract=contract())
    for row in rows:
        env=None;meter.episode=row['episode_id'];meter.region='setup'
        try:
            store.emit('initialization_intent',episode_id=row['episode_id'],
                **initialization_intent(data['references'][row['source_slot']],data['plan']))
            env=meter.call('constructor',lambda:factory(meter));body=env.unwrapped
            require(body._live_meter is meter and body._live_constructor_complete is True and body.with_target is False
                and body.render_action is False and body.render_size==224 and body.relative is True
                and body.action_scale==100 and body.control_hz==10 and body.dt==.01,'original metered native environment')
            row['status']='running';s=row['source_slot']
            episode(env,data['references'][s],data,p['layers'][s],p['tasks'][s],row,visual,runtime,
                meter,store,planner_factory,permit_ram=permit_ram,slots=slots)
        except BaseException as exc:
            old.kernel.retain(row,exc,'episode')
            for slot in slots:
                if slot['episode_id']==row['episode_id'] and slot['status']=='attempted':
                    slot.update(status='failed',error=repr(exc))
        finally:
            meter.region='closing'
            if env is not None:
                prior=store.finalizing;store.finalizing=True;close_entered=False
                def close():
                    nonlocal close_entered
                    close_entered=True;return env.close()
                try:meter.call('close',close)
                except BaseException as exc:
                    old.kernel.retain(row,exc,'close')
                    if not close_entered:
                        try:env.close()
                        except BaseException as secondary:old.kernel.retain(row,secondary,'emergency_close')
                finally:store.finalizing=prior
            store.put(row['episode_id']+'_episode.json',row);store.emit('episode_closed',receipt=row)
    try:
        report['final_check']=final_check();require(report['final_check']['model_unchanged'] is True,'whole-job model unchanged')
        require(not meter.pending and not meter.secondary,'all physical/visual attempts returned')
        c=meter.counts
        require(c['public_step']['completed']==sum(r['warmup_returned']+r['control_returned'] for r in rows)
            and c['set_state']['completed']==2*c['reset']['completed'] and
            c['physics_step']['completed']==2*c['reset']['completed']+10*c['public_step']['completed'] and
            c['goal_setter']['completed']==c['reset']['completed']+c['goal_bind']['completed'] and
            c['explicit_render']['completed']==sum(r['PNG'] for r in rows),'actual native operation identities')
    except BaseException as exc:errors.append(dict(stage='final_checks',error=repr(exc)))
    errors.extend(meter.secondary)
    clean=not errors and all(r['first_error'] is None and r['status'] in
        ('completed20','contract_refused','native_stopped_warmup','native_stopped_control') for r in rows)
    report.update(status='known_fullcommit_recorded' if clean else 'partial_or_failed',counts=meter.counts,
        pending=meter.pending,completed20=sum(r['full_horizon'] for r in rows),
        equal_actual_execution_horizon=all(len({r['control_returned'] for r in rows if r['source_slot']==s})==1 for s in (0,1)),
        wall_seconds=time.monotonic()-data.get('job_started',meter.started),budget_match='notmatched',equal_FLOPs=False,equal_wall=False)
    store.finalizing=True
    store.emit('run_finished',status=report['status'],counts=meter.counts,nodes_sha256=digest(slots))
    report.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    return report

def run(prepared,expected,output,*,authorization):
    require(authorization==AUTH,'MAIN explicit known2 CUDA + native physics authorization')
    started=time.monotonic();p=verify(prepared,expected);output=exclusive(output)
    data=source_data(p)
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    data['visual_tasks']={s:dict(gn_task=enc.load_task(l['task_export'],l['task_sha256'])['gn_task'],model_bindings=enc.MODELS)
        for s,l in enumerate(p['layers'])};data['job_started']=started
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
                torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'whole-job wall/CUDA cap');store.check()
            require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32 and
                not torch.is_autocast_enabled('cuda'),'actual TF32/autocast disabled')
        check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);check()
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original frozen model')
        for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['sources'],'presealed native model class')
        def final():
            enc.low.frozen(model)
            after=signal.model_hash(model)
            require(before==after,'frozen state and all persistent/nonpersistent buffers')
            for f,h in p['sources'].items():require(sha(f)==h,'source/weights/config after SHA: '+f)
            return dict(model_unchanged=True,before=before,after=after,source_manifest_sha256=digest(p['sources']))
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            def precision():
                return dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
                    autocast=torch.is_autocast_enabled('cuda'),math=torch.backends.cuda.math_sdp_enabled(),
                    flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                    cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())
            numeric_before=precision()
            report=execute(p,data,store,old.Handler(None),dict(model=model,device=device,check_limits=check),
                old.native_factory,lambda t,s:native.NativeProgram(model,device,t,s),authorization=AUTH,final_check=final)
            report['precision_before']=numeric_before;report['precision_after']=precision()
            require(numeric_before==report['precision_after']==dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,
                math=True,flash=False,efficient=False,cudnn_sdp=False),'actual before/after MATH precision guards')
        torch.cuda.synchronize(device);check()
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        if report is None:
            report=data.get('live_report') or dict(schema=NAME+'.report',status='partial_or_failed',rows=roster(),nodes=node_roster(),
                errors=[],fixed_denominator=8,node_denominator=24,known_tasks=2,contract=contract())
        report['errors'].extend(errors)
        if errors:report['status']='partial_or_failed'
        report.update(prepared=str(Path(prepared).absolute()),prepare_sha256=expected,host=host,
            wall_seconds=time.monotonic()-started,P1_admission=False,P3_admission=False,freshness=False,
            scientific_admission=False,source_manifest_sha256=digest(p['sources']))
        report['per_episode_neural']=node_accounting(report['nodes']);report['CUDA_peak_bytes']=None
        if model is not None:
            try:
                after=signal.model_hash(model);report['model_before']=before;report['model_after']=after
                require(before==after,'final frozen model buffers')
                report['CUDA_peak_bytes']=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
            except BaseException as exc:
                report['errors'].append(dict(stage='cleanup_model',error=repr(exc)));report['status']='partial_or_failed'
        report['wall_seconds']=time.monotonic()-started
        if report['wall_seconds']>=contract()['wall_seconds']:
            report['errors'].append(dict(stage='wall',error='whole-job engineering wall exceeded'));report['status']='partial_or_failed'
        store.put('report.json',report,final=True)
    return dict(output=str(output),report_sha256=sha(output/'report.json'),status=report['status'])


def load_events(output,report):
    path=harden.regular(Path(output)/'journal.jsonl',contract()['output_bytes'])
    require(sha(path)==report['journal_sha256'],'own physical journal SHA')
    payload=path.read_bytes();require(payload.endswith(b'\n'),'complete physical line')
    events=[json.loads(line,object_pairs_hook=signal.unique_pairs) for line in payload.splitlines()]
    previous=None
    for i,e in enumerate(events):
        raw(e)
        require(type(e) is dict and e.get('kind') in FIELDS and set(e)==COMMON|FIELDS[e['kind']],
            'known exact own physical ABI')
        require(type(e['index']) is int and e['index']==i and e['previous_sha256']==previous and
            e['event_sha256']==old.source.digest({k:v for k,v in e.items() if k!='event_sha256'}),
            'own full physical chain')
        previous=e['event_sha256']
    require(events and previous==report['journal_chain_tail'] and events[0]['kind']=='run_started'
        and events[-1]['kind']=='run_finished','complete own terminal journal')
    return events

def census(events, report):
    """Account for EVERY event, every resource, every episode, including prefixes.

    Clean native-stop/refusal is complete engineering evidence, not completed20. Partial
    prefixes retain attempts/pending and are never granted complete census.
    """
    rows = report['rows']; nodes = report['nodes']; ids = [r['episode_id'] for r in roster()]
    require([r['episode_id'] for r in rows] == ids, 'fixed eight episode denominator')
    require([(n['episode_id'], n['boundary']) for n in nodes] ==
            [(n['episode_id'], n['boundary']) for n in node_roster()], 'fixed twenty-four node denominator')
    clean = report['status'] == 'known_fullcommit_recorded' and not report['errors'] and not report['pending']
    require(events[0]['contract'] == contract() and events[-1]['status'] == report['status']
            and events[-1]['counts'] == report['counts'] and events[-1]['nodes_sha256'] == digest(nodes),
            'run bookends bind original contract/counts/nodes')
    require(Counter(e['kind'] for e in events)['run_started'] == 1 and
            Counter(e['kind'] for e in events)['run_finished'] == 1, 'unique run bookends')
    resources = contract()['physical_caps'] | contract()['visual_caps']
    totals = {k: dict(attempted=0, completed=0) for k in resources}
    per = {eid: {k: dict(attempted=0, completed=0) for k in resources} for eid in ids}
    grouped = {eid: [] for eid in ids}; pending = {}; serial = -1
    active = 0; stack = []; region_counts = {eid: Counter() for eid in ids}
    for e in events[1:-1]:
        kind = e['kind']
        require(kind in FIELDS and set(e) == COMMON | FIELDS[kind], 'known exact physical event schema')
        if kind == 'episode_closed':
            require(active < 8 and e['receipt'] == rows[active], 'exact ordered episode close, no duplicate')
            grouped[ids[active]].append(e); active += 1; continue
        require(active < 8, 'no event after eight episode closes')
        eid = ids[active]
        if kind == 'handler_event':
            require(type(e['namespace']) is str and e['namespace'] in
                    [eid + f'_b{b:02}_encode' for b in (5, 10, 15, 20)], 'own current encoder namespace')
            native_event = e['event']
            require(type(native_event) is dict and set(native_event) == {'kind', 'native'} and
                    native_event['kind'] == 'native_event', 'known visual handler event')
            value = native_event['native']
            require(type(value) is dict and set(value) == {'kind', 'entry'} and
                    value['kind'] in ('module_attempt', 'module_return'), 'known visual native event')
            entry = value['entry']
            require(type(entry) is dict and set(entry) == {'index', 'image', 'resource', 'completed'} and
                    type(entry['completed']) is bool and entry['completed'] == (value['kind'] == 'module_return') and
                    type(entry['image']) is int and entry['image'] == 0 and type(entry['index']) is int and
                    (entry['index'], entry['resource']) in ((0, 'encoder'), (1, 'projector')),
                    'exact visual native entry')
        else:
            require(e.get('episode_id') == eid, 'event belongs to active ordered episode')
        grouped[eid].append(e)
        if kind not in ('operation_attempted', 'operation_completed'):
            continue
        k = e['resource']; units = e['units']; ident = e['operation_id']
        require(k in resources and type(units) is int and units == 1 and type(ident) is int and
                e['region'] in REGIONS[k], 'native single-call units and legal resource region')
        operation = {k: e[k] for k in FIELDS[kind]}
        if kind == 'operation_attempted':
            require(ident == serial + 1, 'consecutive unique physical operation IDs')
            serial = ident; pending[ident] = operation; stack.append(ident); axis = 'attempted'
        else:
            require(stack and stack[-1] == ident and pending.pop(ident, None) == operation,
                    'nested native operation matching return')
            stack.pop(); axis = 'completed'
        totals[k][axis] += units; per[eid][k][axis] += units
        region_counts[eid][k, e['region'], axis] += units
    require(totals == report['counts'] and {str(k): v for k, v in pending.items()} ==
            {str(k): v for k, v in report['pending'].items()}, 'all-resource physical census and pending')
    for k, c in totals.items():
        require(0 <= c['completed'] <= c['attempted'] <= resources[k], 'all physical/visual caps')
    if not clean:
        return dict(complete=False, event_counts=dict(Counter(e['kind'] for e in events)),
                    counts=totals, per_episode=per, reason='partial prefix; full census not certified')
    require(active == 8 and not stack and not pending, 'all eight episodes closed with no pending operations')
    for row in rows:
        eid = row['episode_id']; es = grouped[eid]; by = defaultdict(list)
        for e in es: by[e['kind']].append(e)
        require(not by['invalid_render'], 'clean run has no invalid RGB event')
        require(len(by['initialization_intent'])==1 and
            by['initialization_intent'][0]['index']<by['operation_attempted'][0]['index'],
            'recipe sealed before each independent native constructor')
        warm, control, enc = row['warmup_returned'], row['control_returned'], row['encodings']
        require(type(warm) is int and 0 <= warm <= 10 and type(control) is int and 0 <= control <= 20,
                'bounded actual returned steps')
        expected_observations = [('warmup', b) for b in range(warm + 1)]
        if row['initial_exact']: expected_observations += [('plan', b) for b in range(control + 1)]
        observations = by['fresh_observation']
        require([(e['receipt']['phase'], e['receipt']['step']) for e in observations] == expected_observations
                and row['PNG'] == len(observations), 'all actual observations and PNG census')
        require([(e['phase'], e['step']) for e in by['guard_passed']] == expected_observations,
                'one guard for every actual observation')
        for observation, guard in zip(observations, by['guard_passed'], strict=True):
            require(observation['index'] < guard['index'] and observation['receipt']['observed_event_index'] ==
                    observation['index'] and type(guard['canonical_agent_pixels']) is int and
                    guard['canonical_agent_pixels'] > 0, 'observation/guard order and real positive guard')
        for kind, n in (('warmup_return', warm), ('control_return', control), ('authorized_dispatch', control)):
            require([e['step'] for e in by[kind]] == list(range(1, n + 1)), 'all returned/authorized step census')
        require([e['actual_step'] for e in by['boundary_encoded']] == [e['step'] for e in enc],
                'all and only current encoding events')
        sealed = [n for n in nodes if n['episode_id'] == eid and n['status'] in ('sealed','observed_commitment','refused')]
        require([e['boundary'] for e in by['boundary_sealed']] == [n['boundary'] for n in sealed],
                'all and only sealed node events')
        require([c['step'] for c in row['checkpoints']] ==
                [b for phase, b in expected_observations if phase == 'plan' and b in (0, 5, 10, 15, 20)],
                'complete checkpoint census')
        require(row['checkpoints'] == [e['receipt'] for e in observations if e['receipt']['phase'] == 'plan'
                and e['receipt']['step'] in (0, 5, 10, 15, 20)], 'checkpoint receipts equal actual observations')
        for event, node in zip(by['boundary_sealed'], sealed, strict=True):
            require(event['seal_sha256'] == node['seal']['seal_sha256'], 'all seal events bind actual sealed node')
        # Pinned native constructor has no implicit reset. Each explicit reset
        # invokes two setters, one goal render and two integration substeps.
        want = dict(constructor=1, reset=2, set_state=4, goal_render=2,
                    goal_setter=2 + int(row['initial_exact']), goal_bind=int(row['initial_exact']),
                    public_step=warm + control, physics_step=4 + 10 * (warm + control),
                    explicit_render=len(observations), close=1, encoder=len(enc), projector=len(enc))
        require(per[eid] == {k: dict(attempted=v, completed=v) for k, v in want.items()},
                'independent per-episode ALL resource census')
        regions = {('constructor', 'setup'): 1, ('reset', 'setup'): 2, ('set_state', 'setup'): 4,
            ('goal_render', 'setup'): 2, ('goal_setter', 'setup'): 2,
            ('goal_setter', 'goal_bind'): int(row['initial_exact']), ('goal_bind', 'goal_bind'): int(row['initial_exact']),
            ('public_step', 'warmup'): warm, ('physics_step', 'setup'): 4, ('physics_step', 'warmup'): 10 * warm,
            ('explicit_render', 'setup'): 1, ('explicit_render', 'warmup'): warm,
            ('explicit_render', 'goal_bind'): int(row['initial_exact']), ('close', 'closing'): 1,
            ('encoder', 'encoding'): len(enc), ('projector', 'encoding'): len(enc)}
        for region, steps in (('prefix', min(control, 5)), ('suffix', max(control - 5, 0))):
            regions.update({('public_step', region): steps, ('physics_step', region): 10 * steps,
                            ('explicit_render', region): steps})
        require(region_counts[eid] == Counter({(k, region, axis): n for (k, region), n in regions.items()
                for axis in ('attempted', 'completed') if n}), 'all-resource per-region native census')
        expected_handlers = []
        for encoded, event in zip(enc, by['boundary_encoded'], strict=True):
            b = encoded['step']; ns = eid + f'_b{b:02}_encode'
            require(event['encoding_sha256'] == encoded['receipt_sha256'], 'encoding event binds exact receipt')
            local = [h for h in by['handler_event'] if h['namespace'] == ns]
            require(len(local) == 4, 'exact four visual handler events per encoding')
            for i, (resource, action) in enumerate((('encoder', 'attempt'), ('encoder', 'return'),
                                                   ('projector', 'attempt'), ('projector', 'return'))):
                h = local[i]; native = dict(kind='module_' + action, entry=dict(
                    completed=action == 'return', image=0, index=i // 2, resource=resource))
                require(h['event'] == dict(kind='native_event', native=native) and h['index'] < event['index'],
                        'exact visual handler native lifecycle')
                expected_handlers.append(h['index'])
            ops = [op for op in es if op['kind'] in ('operation_attempted', 'operation_completed') and
                   op['resource'] in ('encoder', 'projector') and local[0]['index'] < op['index'] < event['index']]
            require([(op['kind'], op['resource']) for op in ops] ==
                    [('operation_attempted', 'encoder'), ('operation_completed', 'encoder'),
                     ('operation_attempted', 'projector'), ('operation_completed', 'projector')],
                    'each visual handler has exact metered native calls')
            require(all(local[i]['index'] < ops[i]['index'] <
                    (local[i + 1]['index'] if i < 3 else event['index']) for i in range(4)),
                    'visual handler and metering exact interleave')
        require(expected_handlers == [h['index'] for h in by['handler_event']], 'no stray visual handler events')
        # Every kind has an exact expected census; operations include ALL axes.
        wanted_kinds = dict(initialization_intent=1,operation_attempted=sum(want.values()), operation_completed=sum(want.values()),
            fresh_observation=len(observations), guard_passed=len(observations), warmup_return=warm,
            authorized_dispatch=control, control_return=control, handler_event=4 * len(enc),
            boundary_encoded=len(enc), boundary_sealed=len(sealed), episode_closed=1)
        require(Counter(e['kind'] for e in es) == Counter({k: v for k, v in wanted_kinds.items() if v}),
                'full physical event-kind census')
    return dict(complete=True, event_counts=dict(Counter(e['kind'] for e in events)), counts=totals, per_episode=per,
                per_region={eid: [dict(resource=k, region=region, axis=axis, units=n)
                    for (k, region, axis), n in sorted(counts.items())] for eid, counts in region_counts.items()})

def consume(output,report_sha,prepared,prepare_sha):
    output=Path(output).absolute();p=verify(prepared,prepare_sha)
    require(sha(output/'report.json')==report_sha,'external report byte SHA')
    r=signal.read(output/'report.json',contract()['final_reserve_bytes']);rows=r['rows'];nodes=r['nodes']
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['fixed_denominator']==8 and
        r['node_denominator']==24 and r['known_tasks']==2 and
        [(x['episode_id'],x['source_slot'],x['mode']) for x in rows]==
        [(x['episode_id'],x['source_slot'],x['mode']) for x in roster()] and
        [(x['episode_id'],x['boundary']) for x in nodes]==[(x['episode_id'],x['boundary']) for x in node_roster()],
        'all8 arms and24 slots including failed/unreached')
    require(r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha and
        old.read(output/'execution_binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'execution prepare binding')
    for key in ('P1_admission','P3_admission','freshness','scientific_admission'):require(r[key] is False,'no scientific/fresh release')
    # Startup failure is evidence only, never a successful engineering run.
    if 'journal_sha256' not in r:
        require(r['status']=='partial_or_failed','no invented terminal journal')
        return dict(schema=NAME+'.consume',status='startup_or_incomplete_preserved',
            fixed_denominator=8,node_denominator=24,complete=False,
            P1_admission=False,P3_admission=False,C3=False,freshness=False,scientific_admission=False)
    jp=output/'journal.jsonl';require(jp.stat().st_size<=contract()['output_bytes'] and sha(jp)==r['journal_sha256'],'physical journal bytes')
    payload=jp.read_bytes();require(payload.endswith(b'\n'),'complete terminal event')
    events=[json.loads(line,object_pairs_hook=signal.unique_pairs) for line in payload.splitlines()]
    counts={k:dict(attempted=0,completed=0) for k in contract()['physical_caps']|contract()['visual_caps']}
    previous=None;pending={};serial=-1
    for i,e in enumerate(events):
        require(e['index']==i and e['previous_sha256']==previous and
            e['event_sha256']==old.source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'physical hash chain')
        previous=e['event_sha256']
        if e['kind'] not in ('operation_attempted','operation_completed'):continue
        op={k:e[k] for k in ('operation_id','episode_id','resource','units','region')};ident=op['operation_id'];k=op['resource']
        require(k in counts and type(op['units']) is int and op['units']>0,'actual operation units')
        if e['kind']=='operation_attempted':
            require(type(ident) is int and ident>serial,'unique operation attempt');serial=ident;pending[ident]=op
            counts[k]['attempted']+=op['units']
        else:
            require(pending.pop(ident,None)==op,'matching actual native return');counts[k]['completed']+=op['units']
    require(previous==r['journal_chain_tail'] and events[-1]['kind']=='run_finished' and
        events[-1]['nodes_sha256']==digest(nodes) and counts==r['counts'] and
        {str(k):v for k,v in pending.items()}=={str(k):v for k,v in r['pending'].items()},'terminal actual operation census')
    for k,v in counts.items():
        require(v['completed']<=v['attempted']<=(contract()['physical_caps']|contract()['visual_caps'])[k],
            'actual physical/visual engineering caps')
    for row in rows:require(old.read(output/(row['episode_id']+'_episode.json'))==row,'durable episode')
    for slot in nodes:
        if 'native_journal_sha256' not in slot:continue
        prior=[x for x in nodes if x['episode_id']==slot['episode_id'] and x['boundary']<slot['boundary']]
        require(slot['resource_receipt']['caps']==remaining_caps(slot['mode'],prior),
            'exact predispatch arm remainder, failed work never refunded')
        path=output/(slot['episode_id']+f"_b{slot['boundary']:02}")/'events.jsonl'
        require(sha(path)==slot['native_journal_sha256'],'native journal bytes including partial work')
        es,tail=signal.journal(path);require(tail==slot['native_journal_tail'],'native journal prefix/terminal tail')
        audit_resources(slot['resource_receipt'],es)
    strict_events=load_events(output,r)
    physical_census=census(strict_events,r)
    if r['status']!='known_fullcommit_recorded' or r['errors'] or r['pending']:
        return dict(schema=NAME+'.consume',status='saved_partial_preserved',complete=False,
            fixed_denominator=8,node_denominator=24,counts=counts,rows=rows,nodes=nodes,
            neural_paths_complete=False,P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False)
    rgb_audit=harden.audit_rgb(output,r,p)
    source=source_data(p);references=source['references']
    audited=[];latent_by_episode={}
    for row in rows:
        terminal_disposition(row)
        require(old.read(output/(row['episode_id']+'_episode.json'))==row,'durable episode')
        s=row['source_slot'];base=p['tasks'][s];eid=row['episode_id'];same=[e for e in events if e.get('episode_id')==eid]
        ref=references[s]
        intents=[e for e in same if e['kind']=='initialization_intent']
        require(len(intents)==1 and {k:v for k,v in intents[0].items() if k not in COMMON|{'episode_id'}}==
            initialization_intent(ref,source['plan']),'exact independent reset/place/warm recipe, no poststate')
        warm=[e['receipt'] for e in same if e['kind']=='fresh_observation' and e['receipt']['phase']=='warmup']
        require([x['step'] for x in warm]==list(range(row['warmup_returned']+1)),'complete actual warmup trajectory')
        ws=[e for e in same if e['kind']=='warmup_return']
        for k,e in enumerate(ws):
            require(np.asarray(e['command'],np.float32).tobytes()==ref['raw_history'][k].tobytes()
                and type(e['terminated']) is type(e['truncated']) is bool and
                (k==len(ws)-1 or not(e['terminated'] or e['truncated'])),
                'actual original warmup raw bits and no return after earlystop')
        if row['initial_exact']:require(row['warmup_returned']==10,'same root requires completed warmup10')
        for k,b in enumerate((0,5,10)):
            if b>row['warmup_returned']:continue
            ob=warm[b]
            require(ob['pixel_sha256']==old.array_sha(ref['history'][k]) and
                np.array_equal(ob['state'],ref['states'][k]) and ob['snapshot']==ref['dynamics'][k],
                'original known reset/warmup RGB/state/dynamics including native earlystop')
        observed=[e['receipt'] for e in same if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan']
        require([x['step'] for x in observed]==list(range(row['control_returned']+1)) if row['initial_exact'] else not observed,
            'full actual control observation trajectory')
        dispatch=[e for e in same if e['kind']=='authorized_dispatch'];returns=[e for e in same if e['kind']=='control_return']
        require(len(dispatch)==len(returns)==len(row['executed_raw'])==row['control_returned'],'no unreturned action fabricated')
        require(row['full_horizon']==(row['status']=='completed20') and (not row['full_horizon'] or row['control_returned']==20),
            'completed20 is actual full horizon, not first5')
        if row['full_horizon']:require([e['step'] for e in row['encodings']]==[5,10,15,20],'all four real encodings')
        for ob in warm+observed:frames.png(ob,output)
        if row['initial_exact']:
            require(row.get('prefix')==prefix_binding(p['layers'][s]) and
                old.read(output/(eid+'_prefix_commit.json'))==row['prefix'],'original fixed GN prefix binding')
            require([e['command'] for e in dispatch[:5]]==row['prefix']['commands'][:min(5,len(dispatch))],
                'original raw prefix5, not another plan or previous outcome')
        cached=deepcopy(base['root_context']);last_boundary=None;latent_by_episode[eid]={}
        for encoded in row['encodings']:
            b=encoded['step'];ns=eid+f'_b{b:02}';er=old.read(output/(ns+'_encoding.json'));mapping=er['mapping'];response=er['response']
            require(old.source.digest(er)==encoded['receipt_sha256'],
                'original visual receipt digest')
            ob=next(x for x in observed if x['step']==b)
            original=Path(ob['path']);alias=output/(ns+'_plan_05.png')
            require(original.parent==output and original.name==eid+f'_plan_{b:02}.png' and
                original.resolve()==original and alias.resolve()==alias and
                original.stat().st_size<=1024**2 and sha(original)==sha(alias)==ob['file_sha256'] and
                original.read_bytes()==alias.read_bytes() and mapping['actual_step']==b and
                mapping['original_png']=={k:ob[k] for k in ('path','file_sha256','pixel_sha256')},'current actual frame, no stale copy')
            from PIL import Image
            with Image.open(original) as image:rgb=np.asarray(image).copy()
            require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8 and old.array_sha(rgb)==ob['pixel_sha256'],'actual RGB pixel identity')
            z=load_array(output,response['latent']);old.native(z);latent_by_episode[eid][b]=z
            enc_request=er['request']
            require(encoded['latent']==response['latent'] and er['actual_step']==b and response['episode_id']==ns and
                response['request_sha256']==enc_request['request_sha256']==
                old.source.digest({k:v for k,v in enc_request.items() if k!='request_sha256'}) and
                enc_request['schema']==old.NAME+'.encode_request' and enc_request['episode_id']==ns and
                enc_request['task_export']==p['layers'][s]['task_export'] and enc_request['observed_step']==5 and
                enc_request['png']==dict(path=str(alias),file_sha256=ob['file_sha256'],pixel_sha256=ob['pixel_sha256']),
                'exact current-frame encoder request/return/artifact binding')
            require(response['source_verified'] is response['model_unchanged'] is True and response['device_type']=='cuda'
                and response['model_hash_before']==response['model_hash_after']==p['modelbinding']['tensor_sha256'],
                'frozen current-frame CUDA encoder binding')
            if b==20:
                require(row['full_horizon'] is True and row['actual_Q20']==float(np.mean((z.astype(float)-np.asarray(base['goal']))**2)),
                    'terminal20 actual Q separate from planning-horizon Q');continue
            actual=np.concatenate((moving.old.kernel.moving.warmup_commands(),np.asarray(row['executed_raw'][:b],np.float32).reshape(-1,2)))[-10:]
            cached=moving.roll(cached,z,actual,base)
            slot=next(x for x in nodes if x['episode_id']==eid and x['boundary']==b)
            require(slot['status'] in ('sealed','observed_commitment','refused'),
                'every clean observed planning boundary has actual paid disposition')
            request=slot['request'];validate_request(request,base)
            require(old.read(output/(ns+'_request.json'))==request,'durable exact model-only request')
            require(request['task']['root_context']==cached and request['raw10']==actual.tolist(),
                'own full actual observed history, never root-J or predicted-child reuse')
            result=slot['receipt'];journal=output/ns/'events.jsonl'
            require(old.read(output/(ns+'_node.json'))==dict(request=request,result=result,
                native_journal_sha256=slot['native_journal_sha256'],native_journal_tail=slot['native_journal_tail']),
                'own durable complete node record')
            require(sha(journal)==slot['native_journal_sha256'],'all native journal bytes')
            native_events,tail=signal.journal(journal);require(tail==slot['native_journal_tail'],'all native chain tail')
            audited.append(dict(episode_id=eid,boundary=b,**audit_node(request,result,native_events,base)))
            seal=slot['seal'];require(old.read(output/(ns+'_seal.json'))==seal and
                seal['seal_sha256']==digest({k:v for k,v in seal.items() if k!='seal_sha256'}) and
                seal['request_sha256']==digest(request) and seal['plan_sha256']==digest(result) and
                seal['commands']==result['commands'] and seal['commands_sha256']==digest(result['commands']),
                'durable request/selected fullH seal')
            require(seal['commitment_sha256']==(digest([request,result]) if result['status']=='sealed'
                else request['commitment']['seal_sha256']), 'actual session commitment digest, not a relabeled dispatch seal')
            require(slot['status']==result['status'] and seal['status']==result['status'] and
                seal['commitment']==request['commitment'],'node disposition and pending identity')
            sealed=next(e['index'] for e in same if e['kind']=='boundary_sealed' and e['boundary']==b)
            encoded_event=next(e['index'] for e in same if e['kind']=='boundary_encoded' and e['actual_step']==b)
            require(ob['observed_event_index']<encoded_event<sealed,'observe+encode+pay+seal causal order')
            selected=[x for x in dispatch if x['packet_boundary']==b]
            if result['status']=='refused':
                require(not selected and row['control_returned']==b and row['status']=='contract_refused' and
                    row['refusal']==dict(boundary=b,reason=result['refusal']),'whole residual refusal/no post-refusal actions')
            else:
                active=next((c for c in row['commitments'] if c['start']<=b<c['end']),None)
                require(active is not None,'one active complete commitment')
                origin=next(n for n in nodes if n['episode_id']==eid and n['boundary']==active['start'])
                require(active==dict(start=active['start'],end=active['start']+len(origin['receipt']['commands']),
                    seal_sha256=origin['seal']['commitment_sha256'],commands=origin['receipt']['commands']),
                    'full immutable sealed command buffer')
                require(request['commitment']==(None if b==active['start'] else
                    {k:active[k] for k in ('start','end','seal_sha256')}),'no replacement after own intermediate signals')
                offset=b-active['start'];commands=active['commands'][offset:offset+5]
                require([x['step'] for x in selected]==list(range(b+1,b+1+len(selected))) and len(selected)<=5 and
                    all(x['index']>sealed for x in selected) and [x['command'] for x in selected]==commands[:len(selected)],
                    'exact fullH commitment offsets, no fresh reselection/truncation')
            last_boundary=b
        for e,ret,ob,command in zip(dispatch,returns,observed[1:],row['executed_raw'],strict=True):
            require(e['step']==ret['step']==ob['step'] and e['command']==ret['command']==command and
                ret['index']>e['index'] and len(ob['state'])==7,'actual step/raw/state7 identity')
            metric=old.kernel.metric(ref['goal'],ob['state'])
            require(np.asarray(e['command'],np.float32).tobytes()==np.asarray(command,np.float32).tobytes()==
                np.asarray(ret['command'],np.float32).tobytes() and
                np.array_equal(ob['goal'],ref['goal']) and ob['goal_rgb_sha256']==old.array_sha(ref['goal_rgb']) and
                np.asarray(ob['goal_pose'],np.float64).tobytes()==np.asarray(warm[-1]['goal_pose'],np.float64).tobytes(),
                'actual raw bits and original goal/goal_pose unchanged')
            native_check=native_return_contract(ref['goal'],ob['state'],ret['reward'],ret['terminated'],ret['truncated'])
            require(native_check['native7_distance']==metric['native_distance'] and
                native_check['native_success']==metric['native_success'],'original metric and native API agree')
            calls=[x for x in same if x['kind']=='operation_attempted' and x['resource']=='public_step'
                and e['index']<x['index']<ret['index']]
            require(len(calls)==1,'one actual public attempt per authorized control')
            done=next(x for x in same if x['kind']=='operation_completed' and x['operation_id']==calls[0]['operation_id'])
            require(done['index']<ob['observed_event_index']<ret['index'] and
                sum(x['kind']=='operation_attempted' and x.get('resource')=='physics_step' and
                    calls[0]['index']<x['index']<done['index'] for x in same)==10,
                'actual public return then observation; all ten Space substeps')
            if e['step']<row['control_returned']:
                require(not ret['terminated'] and not ret['truncated'],'never dispatch after native stop')
        if returns:
            metric=old.kernel.metric(ref['goal'],observed[-1]['state'])
            require(metric==row['final_metrics'] and metric['native_success']==row['native_success'] and
                row['final_state']==observed[-1]['state'] and row['last_observation']==observed[-1] and
                row['terminated']==returns[-1]['terminated'] and row['truncated']==returns[-1]['truncated'],
                'actual terminal20 or native-stop outcome retained')
    for row in rows:
        expected_nodes=[dict(request=n['request'],result=n['receipt'],native_journal_sha256=n['native_journal_sha256'],
            native_journal_tail=n['native_journal_tail']) for n in nodes if n['episode_id']==row['episode_id']
            and n['status'] in ('sealed','observed_commitment')]
        require(row['schedule_nodes']==expected_nodes, 'all and only returned schedule nodes')
        committed=[]
        for n in [x for x in nodes if x['episode_id']==row['episode_id'] and x['status']=='sealed']:
            c=n['receipt']['commands']
            committed.append(dict(start=n['boundary'],end=n['boundary']+len(c),seal_sha256=n['seal']['commitment_sha256'],commands=c))
        require(row['commitments']==committed,'all and only released commitments')
        own=[n for n in nodes if n['episode_id']==row['episode_id']]
        remaining_caps(row['mode'],own)
    for s in (0,1):
        pair=[r0 for r0 in rows if r0['source_slot']==s]
        if all(r0['prefix_exact'] for r0 in pair):
            ends=[next(e['receipt'] for e in events if e['kind']=='fresh_observation' and
                e.get('episode_id')==r0['episode_id'] and e['receipt']['phase']=='plan' and e['receipt']['step']==5) for r0 in pair]
            require(all(ob[k]==ends[0][k] for ob in ends[1:] for k in ('pixel_sha256','state','snapshot','goal','goal_rgb_sha256','goal_pose')),
                'paired fresh native prefix identity')
    clean=r['status']=='known_fullcommit_recorded' and not r['errors'] and not r['pending']
    if clean:
        require(r['budget_match']=='notmatched' and r['equal_FLOPs'] is False and r['equal_wall'] is False,
            'recorded actual budget unmatched; never inferred equal FLOPs')
        require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'] and
            r['final_check']['model_unchanged'] is True and r['source_manifest_sha256']==digest(p['sources']),
            'unchanged actual frozen model/weights/source closure')
        require(r['final_check']['before']==r['model_before'] and r['final_check']['after']==r['model_after'] and
            r['final_check']['source_manifest_sha256']==digest(p['sources']), 'whole-job final model/source check binding')
        require(r['precision_before']==r['precision_after']==dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,
            math=True,flash=False,efficient=False,cudnn_sdp=False),'actual MATH/TF32/autocast guards')
        require(type(r['wall_seconds']) in (int,float) and 0<=r['wall_seconds']<contract()['wall_seconds'] and
            len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=contract()[k] for v,k in
                zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),
            'measured whole-job wall/CUDA memory caps')
        require(r['per_episode_neural']==node_accounting(nodes),'each arm actual API/sample/AD/SVD/wall axes')
        require(r['completed20']==sum(x['full_horizon'] for x in rows) and
            r['equal_actual_execution_horizon']==all(len({r['control_returned'] for r in rows if r['source_slot']==s})==1 for s in (0,1)),
            'actual all20/equal-horizon outcomes not assumed')
        require(counts['public_step']['completed']==sum(x['warmup_returned']+x['control_returned'] for x in rows)
            and counts['physics_step']['completed']==2*counts['reset']['completed']+10*counts['public_step']['completed'],
            'full physical census, not just compiler cost')
        if all(x['full_horizon'] for x in rows):
            require(counts['encoder']==counts['projector']==dict(attempted=32,completed=32) and
                counts['public_step']==dict(attempted=240,completed=240) and
                counts['explicit_render']==dict(attempted=256,completed=256) and
                counts['goal_bind']==counts['close']==dict(attempted=8,completed=8),'full all20 physical/visual census')
    for path,h in p['sources'].items():require(sha(path)==h, 'sources unchanged through full saved consume')
    return dict(schema=NAME+'.consume',status='saved_fullcommit_verified_NOT_admitted' if clean else 'saved_partial_preserved',
        complete=clean,fixed_denominator=8,node_denominator=24,audited=audited,counts=counts,
        completed20=r.get('completed20'),equal_actual_execution_horizon=r.get('equal_actual_execution_horizon'),
        native_metrics=[x['final_metrics'] for x in rows],budget_match='notmatched',equal_FLOPs=False,equal_wall=False,
        per_episode_neural=r['per_episode_neural'],physical_census=physical_census,RGB_preprocessing=rgb_audit,
        P1_admission=False,P3_admission=False,C3=False,freshness=False,scientific_admission=False,independent_neural_replay=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','accounting','run','consume'))
    for key in ('output','prepared','prepare-sha','report-sha','receipt'):parser.add_argument('--'+key)
    parser.add_argument('--authorize-known-cuda-and-physics',action='store_true')
    args=parser.parse_args(argv)
    require(not args.receipt or args.command=='consume', 'receipt writer only for saved consume')
    if args.command=='prepare':result=prepare(args.output)
    elif args.command=='verify':result=dict(verified=True,schema=verify(args.prepared,args.prepare_sha)['schema'])
    elif args.command=='accounting':result=accounting()
    elif args.command=='run':
        require(args.authorize_known_cuda_and_physics, 'MAIN explicit HOST CUDA/native authorization')
        result=run(args.prepared,args.prepare_sha,args.output,authorization=AUTH)
    else:
        result=consume(args.output,args.report_sha,args.prepared,args.prepare_sha)
        if args.receipt:
            path=exclusive(args.receipt);payload=raw(result)
            require(len(payload)<=contract()['final_reserve_bytes'], 'bounded own receipt')
            with path.open('xb') as f:
                require(f.write(payload)==len(payload), 'full saved receipt');f.flush();os.fsync(f.fileno())
            signal.fsync_dir(path.parent)
            result=dict(path=str(path),sha256=sha(path),status=result['status'],complete=result['complete'])
    print(core.canonical(result).decode())
    if args.command=='run' and result['status']!='known_fullcommit_recorded':raise SystemExit(2)


if __name__=='__main__':main()
