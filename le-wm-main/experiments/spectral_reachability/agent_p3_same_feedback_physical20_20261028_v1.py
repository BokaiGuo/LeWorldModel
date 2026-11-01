"""Known2 same-feedback MPC20: MAIN-only HOST CUDA + native physics.

Four actual RGB feedback decisions/episode, execute ONLY first raw5 each time.
No CPU neural fallback, fresh-root access, trained component or admission.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p3_common_budget_20261028_v1 as common
from experiments.spectral_reachability import agent_p4_moving_multiboundary_20261028_v1 as moving

NAME='agent_p3_same_feedback_physical20_20261028_v1'
ROOT=common.ROOT
AUTH='MAIN_KNOWN2_SAME_FEEDBACK_CUDA_AND_PHYSICS'
signal,native,core=common.signal,common.native,common.core
old=moving.old
require,sha,digest,raw=core.require,signal.sha,core.digest,core.canonical
BOUNDARIES=(0,5,10,15)
MODES=('fixed','adaptive')


def contract():
    return dict(boundaries=list(BOUNDARIES),terminal=20,packet_steps=5,episodes=4,known_tasks=2,
        slots=16,execute='winner.actions[:5] only; discard suffix then observe actual RGB and recompile',
        feedback='both modes own actual RGB; older2 own cached observed latents, never model successors',
        original_decision='unmodified runtime chi=.5 gain=.05 policy; unsupported reconfiguration fails closed',
        search_rule=common.rule(),prefix_validation_predict=1,predict_per_complete_node=33,
        nominal_and_AD_predict_per_node=8,search_predict_per_node=24,candidates_by_H={'1':12,'3':4},
        tree_depth='one local extension per actual node; requested depth logged, not a full tree',
        physical_caps=dict(constructor=4,reset=12,set_state=24,goal_render=12,goal_setter=16,
            goal_bind=4,public_step=120,physics_step=1224,explicit_render=128,close=4),
        visual_caps=dict(encoder=20,projector=20),wall_seconds=1800,output_bytes=512*1024**2,
        final_reserve_bytes=64*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        equal_requested_execution_horizon=True,equal_actual_execution_horizon=None,
        same_predict_budget_per_completed_node=True,equal_FLOPs=False,equal_wall=False,
        fixed_is_cost_minimal_H1=False,controlled_dual_signal_pipeline=True,no_padding=True,
        no_retry=True,no_outcome_input=True,with_target=False,TF32=False,SDP='MATH',autocast=False,
        P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False,
        full_C3_demonstrated=False,independent_confirmation=False)


def node_caps():
    c=dict.fromkeys(core.AXES,0)
    c.update(predict_API=33,predict_samples=33,predict_context_tokens=99,
        action_encoder_calls=33,action_encoder_samples=33,action_encoder_context_tokens=99,
        predictor_calls=33,predictor_samples=33,predictor_context_tokens=99,
        pred_proj_calls=33,pred_proj_rows=99,jacobian_API=2,VJP_directions=384,
        grad_API=384,raw_backward_callbacks=384,array_SVD_calls=3,solve_calls=1,
        signal_compile_calls=2,program_compile_calls=1,replay_calls=25,metric_calls=13,schedule_decisions=1)
    return c


def node_roster():
    return [dict(episode_id=f's{s}_{m}',source_slot=s,mode=m,boundary=b,status='not_reached',
        request=None,receipt=None,error=None) for s in (0,1) for m in MODES for b in BOUNDARIES]


def node_accounting(nodes):
    out=[]
    for s in (0,1):
        for mode in MODES:
            selected=[n for n in nodes if (n['source_slot'],n['mode'])==(s,mode)]
            receipts=[n['resource_receipt'] for n in selected if n.get('resource_receipt') is not None]
            out.append(dict(source_slot=s,mode=mode,requested_nodes=4,receipts=len(receipts),
                sealed_nodes=sum(n['status']=='sealed' for n in selected),
                attempted={k:sum(r['attempted'][k] for r in receipts) for k in core.AXES} if receipts else None,
                completed={k:sum(r['completed'][k] for r in receipts) for k in core.AXES} if receipts else None,
                inclusive_operation_wall_ns={resource:sum(r['inclusive_operation_wall_ns'].get(resource,0) for r in receipts)
                    for resource in sorted({k for r in receipts for k in r['inclusive_operation_wall_ns']})},
                wall_semantics='nested inclusive operation wall; not summed into runtime or FLOPs',
                FLOPs=None,CUDA_ns=None,backward_engine_calls=None))
    return out


def roster():
    return [dict(episode_id=f's{s}_{m}',source_slot=s,mode=m,status='not_started',first_error=None,
        secondary_errors=[],warmup_returned=0,control_returned=0,completed_packets=0,executed_raw=[],
        encodings=[],checkpoints=[],PNG=0,initial_exact=False,full_horizon=False,native_success=None,
        final_metrics=None,terminated=None,truncated=None) for s in (0,1) for m in MODES]


class Store(moving.Store):
    def __init__(self,path):
        self.path=Path(path).absolute();self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME) and
            self.path.resolve()==self.path and not self.path.exists(),'new exclusive runtime output')
        require(shutil.disk_usage(self.path.parent).free>=contract()['disk_floor_bytes']+contract()['output_bytes'],
            '512MiB output plus200MiB reserve before start')
        self.path.mkdir();signal.fsync_dir(self.path.parent);self.index=0;self.previous=None
    def check(self,size=4096):
        used=sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file())
        cap=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
        require(used+size<=cap and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,
            'whole-output/final-reserve/disk-floor cap')
    def write(self,name,value,append=False,final=False):
        p=Path(name)
        require(not p.is_absolute() and 1<=len(p.parts)<=2 and all(v not in ('','.','..') for v in p.parts)
            and not (self.path/p).is_symlink() and not (self.path/p).parent.is_symlink(),'own bounded output path')
        before=self.finalizing;self.finalizing=before or final
        try:
            self.check(len(value))
            with (self.path/p).open('ab' if append else 'xb') as f:
                require(f.write(value)==len(value),'full durable write');f.flush();os.fsync(f.fileno())
            signal.fsync_dir((self.path/p).parent);return sha(self.path/p)
        finally:self.finalizing=before


class Meter(old.Meter):
    def __init__(self,store,check):
        super().__init__(store,check)
        self.caps=contract()['physical_caps']|contract()['visual_caps']
        self.counts={k:dict(attempted=0,completed=0) for k in self.caps}
    def check(self):
        if self.region!='closing':self.job_check()
        self.store.check()


class NativeJournal:
    """Model-side output sink: no env/reward/state inputs, immediate native bytes."""
    def __init__(self,store,namespace,check):
        self.path=store.namespace(namespace)/'events.jsonl';self.check=check
        self.store=store;self.stream=self.path.open('xb');self.index=0;self.tail='0'*64
        signal.fsync_dir(self.path.parent)
    def event(self,value):
        self.check()
        record=dict(index=self.index,previous=self.tail,payload=deepcopy(value));record['sha256']=digest(record)
        payload=raw(record)+b'\n';self.store.check(len(payload))
        require(self.stream.write(payload)==len(payload),'full native buffer journal write')
        self.stream.flush();os.fsync(self.stream.fileno());self.tail=record['sha256'];self.index+=1
    def close(self):
        self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();signal.fsync_dir(self.path.parent)


def local_task(base,context,identity):
    """Rebase the reviewed root adapter on THIS paid actual observed node.

    Original root binding remains explicit separately. No predicted edge is
    promoted to observed history; no original-node J is accepted here.
    """
    base=core.validate_task(base);core.context(context)
    out=deepcopy(base);out['root_context']=deepcopy(context)
    out['binding'].update(context_sha256=digest(context),task_id=digest([base['binding']['task_id'],identity]),
        task_sha256=digest([base,context,identity]),task_contract_sha256=digest([NAME,identity]))
    return core.validate_task(out)


def validate_request(p,base=None):
    core.keys(p,('schema','episode_id','source_slot','mode','boundary','base_task_sha256','task','raw10','observation'))
    require(p['schema']==NAME+'.node' and type(p['source_slot']) is int and p['source_slot'] in (0,1)
        and p['mode'] in MODES and p['episode_id']==f"s{p['source_slot']}_{p['mode']}"
        and type(p['boundary']) is int and p['boundary'] in BOUNDARIES,'fixed current own node')
    core.hash_value(p['base_task_sha256']);t=core.validate_task(p['task'])
    require(len(t['goal'])==192 and np.array_equal(t['projection'],np.eye(192)),'frozen identity192 Q')
    a=core.finite(p['raw10'],(10,2),True)
    require((abs(a)<=1).all() and t['root_context']['action_history']==moving.normalized(t,a),
        'own last10 actual raw actions normalized exactly once')
    core.keys(p['observation'],('boundary','png_sha256','pixel_sha256','encoding_sha256','observed_index',
        'encoded_index','observed_latent_sha256'))
    ob=p['observation']
    require(ob['boundary']==p['boundary'] and type(ob['observed_index']) is int and type(ob['encoded_index']) is int
        and 0<=ob['observed_index']<ob['encoded_index'],'observe before actual encoding')
    for k in ('png_sha256','pixel_sha256','encoding_sha256','observed_latent_sha256'):core.hash_value(ob[k])
    require(ob['observed_latent_sha256']==old.packet_digest(t['root_context']['latent_history'][-1]),'own current RGB latent')
    if base is not None:
        require(p['base_task_sha256']==digest(base),'original known task binding')
        identity={k:deepcopy(p[k]) for k in ('episode_id','source_slot','mode','boundary','raw10','observation')}
        require(t==local_task(base,t['root_context'],identity),'same original goal/Q/model/normalizer and current-node binding')
    return t


def plan_node(request,factory,store,ledger,*,permit_ram=False):
    """ACTUAL local signals + useful search + independent selected-prefix replay."""
    t=validate_request(request);s=request['source_slot'];mode=request['mode'];responses=[]
    backend=factory(t,store)
    require(getattr(backend,'kind',None)=='HOST_CUDA' or
        (permit_ram is True and getattr(backend,'kind',None)=='RAM_FIXTURE'),'reviewed native HOST CUDA adapter only')
    result=dict(status='attempted',responses=responses,diagnostic=None,queue=None,bank=None,
        candidates=[],winner=None,prefix_path=None,commands=None,resource_receipt=None)
    try:
        for h in (1,3):
            ledger.scope=dict(slot=s,horizon=h)
            req=core.request(t,dict(context=t['root_context']),operation='compile_finite_horizon_J',horizon=h)
            response=ledger.call('signal_compile',lambda req=req:backend.signal(req,ledger))
            responses.append(response);store.event(dict(kind='local_signal',response=response))
        ledger.scope=dict(slot=s,phase='diagnostic')
        diagnostic=signal.spectra(t,responses,svd=lambda a:ledger.call('array_svd',lambda:np.linalg.svd(a,full_matrices=False)))
        node,signals,receipt=signal.runtime_signal(t,responses,diagnostic);result['diagnostic']=diagnostic
        ledger.scope=dict(slot=s,phase='schedule')
        q=ledger.call('schedule',lambda:signal.decide_root(t,mode,node,signals,receipt));result['queue']=q
        d=q['decision'];h=d['horizon']
        require(d['program_type']==d['extension_type']=='progress' and d['continue_expanding'] is True,
            'unsupported reconfiguration/composition; no progress surrogate or old-action fallback')
        count=24//(2*h);response=next(r for r in responses if r['horizon']==h)
        result['candidates']=[dict(slot=s,mode=mode,index=k,horizon=h,status='not_started',actions=None,
            whole=None,segmented=None,cost=None,error=None) for k in range(count)]
        ledger.scope=dict(slot=s,mode=mode,index=0,horizon=h)
        bank=ledger.call('program_compile',lambda:common.generate(t,response,count,ledger));result['bank']=bank
        for row in result['candidates']:
            k=row['index'];ledger.scope=dict(slot=s,mode=mode,index=k,horizon=h);row['status']='attempted'
            row['actions']=bank['actions'][k]
            row['whole']=ledger.call('replay',lambda:backend.replay(t['root_context'],row['actions'],False,ledger))
            row['segmented']=ledger.call('replay',lambda:backend.replay(t['root_context'],row['actions'],True,ledger))
            require(row['whole']==row['segmented'],'independent full H whole/segmented replay exact')
            terminal=core.checked_path(t,t['root_context'],row['actions'],row['whole'])
            row['cost']=ledger.call('metric',lambda:core.model_cost(t,terminal));row['status']='model_replayed_not_admitted'
        winner=min(result['candidates'],key=lambda r:(r['cost'],r['index']));result['winner']=deepcopy(winner)
        ledger.scope=dict(slot=s,mode=mode,index=winner['index'],horizon=1,operation='selected_prefix_verification')
        path=ledger.call('replay',lambda:backend.replay(t['root_context'],winner['actions'][:5],False,ledger))
        require(path==winner['whole'][:1],'paid selected first5 endpoint equals full plan first boundary')
        ledger.call('metric',lambda:core.model_cost(t,path[0]));result['prefix_path']=path
        result['commands']=deepcopy(winner['actions'][:5]);result['status']='sealed'
        expected=node_caps();expected.update(replay_calls=2*count+1,metric_calls=count+1)
        require(ledger.attempted==ledger.completed==expected,'exact useful per-node33 API/sample full-resource census')
    finally:result['resource_receipt']=ledger.receipt();store.event(dict(kind='local_plan_final',result=result))
    return result


def episode(env,ref,data,layer,base,row,slots,visual,runtime,meter,store,planner_factory,*,permit_ram=False):
    if not moving.setup(env,ref,data['plan'],row,meter,store):return
    context=deepcopy(base['root_context']);obs=row['last_observation']
    for boundary in BOUNDARIES:
        slot=next(x for x in slots if x['episode_id']==row['episode_id'] and x['boundary']==boundary)
        slot['status']='attempted';meter.region='encoding'
        z,observation=moving.encode_boundary(visual,runtime,data['visual_tasks'][row['source_slot']],layer,row,obs,
            env.unwrapped,meter,store,boundary)
        raw10=np.concatenate((ref['raw_history'],np.asarray(row['executed_raw'],np.float32).reshape(-1,2)))[-10:]
        if boundary==0:
            context=deepcopy(context);context['latent_history'][-1]=z.tolist()
            context['action_history']=moving.normalized(base,raw10)
        else:context=moving.roll(context,z,raw10,base)
        identity=dict(episode_id=row['episode_id'],source_slot=row['source_slot'],mode=row['mode'],boundary=boundary,
            raw10=raw10.tolist(),observation=observation)
        request=dict(schema=NAME+'.node',base_task_sha256=digest(base),task=local_task(base,context,identity),**identity)
        validate_request(request,base);slot['request']=request;meter.region='compiling'
        ns=row['episode_id']+f'_b{boundary:02}';store.put(ns+'_request.json',request)
        state=old.state_seal(env.unwrapped);journal=NativeJournal(store,ns,runtime['check_limits'])
        ledger=core.Ledger(node_caps(),sink=journal.event);planned=None
        try:
            planned=plan_node(request,planner_factory,journal,ledger,permit_ram=permit_ram)
            require(old.state_seal(env.unwrapped)==state,'planning cannot mutate live state/goal')
        finally:
            journal.close();slot['native_journal_sha256']=sha(journal.path);slot['native_journal_tail']=journal.tail
            slot['resource_receipt']=planned['resource_receipt'] if planned is not None else ledger.receipt()
        slot['receipt']=planned;slot['status']='sealed'
        seal=dict(schema=NAME+'.dispatch',episode_id=row['episode_id'],boundary=boundary,
            request_sha256=digest(request),plan_sha256=digest(planned),commands=planned['commands'],
            commands_sha256=digest(planned['commands']),native_journal_sha256=slot['native_journal_sha256'])
        seal['seal_sha256']=digest(seal);slot['seal']=seal;store.put(ns+'_seal.json',seal)
        store.emit('boundary_sealed',episode_id=row['episode_id'],boundary=boundary,seal_sha256=seal['seal_sha256'])
        obs=moving.packet(env,planned['commands'],start=boundary,row=row,ref=ref,meter=meter,store=store)
        if obs is None:
            if row['control_returned']==boundary+5:row['completed_packets']+=1
            if row['control_returned']==20:
                # Native success/truncation on the final requested step is
                # still an actual twenty-step outcome, never pad or re-step.
                obs=row['last_observation'];break
            return
        row['completed_packets']+=1
    # terminal encoding is evaluation only; it never controls a fifth packet.
    row['full_horizon']=row['control_returned']==20
    terminal,_=moving.encode_boundary(visual,runtime,data['visual_tasks'][row['source_slot']],layer,row,obs,
        env.unwrapped,meter,store,20)
    row.update(full_horizon=True,status='completed20',actual_Q20=core.model_cost(base,
        dict(context,latent_history=context['latent_history'][1:]+[terminal.tolist()])))


def execute(p,data,store,visual,runtime,factory,planner_factory,*,authorization,final_check,permit_ram=False):
    require(authorization==AUTH and p['contract']==contract(),'MAIN explicit known2 authorization/contract')
    require(set(runtime)=={'model','device','check_limits'} and getattr(runtime['device'],'type',None)=='cuda',
        'explicit single-process HOST CUDA runtime')
    rows=roster();slots=node_roster();meter=Meter(store,runtime['check_limits']);errors=[]
    report=dict(schema=NAME+'.report',status='running',rows=rows,nodes=slots,errors=errors,contract=contract(),
        fixed_denominator=4,node_denominator=16,known_tasks=2,P1_admission=False,P3_admission=False,
        scientific_admission=False,freshness=False)
    report.update(counts=meter.counts,pending=meter.pending)
    data['live_report']=report
    store.put('roster.json',dict(rows=rows,nodes=slots));store.emit('run_started',contract=contract())
    for row in rows:
        env=None;meter.episode=row['episode_id'];meter.region='setup'
        try:
            env=meter.call('constructor',lambda:factory(meter));body=env.unwrapped
            require(body._live_meter is meter and body._live_constructor_complete is True and body.with_target is False
                and body.render_action is False and body.render_size==224 and body.relative is True
                and body.action_scale==100 and body.control_hz==10 and body.dt==.01,'original metered native environment')
            row['status']='running';s=row['source_slot']
            episode(env,data['references'][s],data,p['layers'][s],p['tasks'][s],row,slots,visual,runtime,
                meter,store,planner_factory,permit_ram=permit_ram)
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
        ('completed20','native_stopped_warmup','native_stopped_control') for r in rows)
    report.update(status='known_same_feedback_recorded' if clean else 'partial_or_failed',counts=meter.counts,
        pending=meter.pending,completed20=sum(r['full_horizon'] for r in rows),
        equal_actual_execution_horizon=all(rows[2*s]['control_returned']==rows[2*s+1]['control_returned'] for s in (0,1)),
        wall_seconds=time.monotonic()-data.get('job_started',meter.started),equal_FLOPs=False,equal_wall=False)
    store.finalizing=True
    store.emit('run_finished',status=report['status'],counts=meter.counts,nodes_sha256=digest(slots))
    report.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    return report


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/(NAME+'_tests.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build():
    # Existing actual authorities suffice for known2. No new approval/cert manifest.
    old_p=old.verify(moving.OLD_PREPARE,moving.OLD_SHA);signal_p=signal.prepare_material()
    layers=[deepcopy(next(x for x in old_p['layers'] if x['source_slot']==s and x['arm']==old.ARMS[0])) for s in (0,1)]
    tasks=signal_p['tasks']
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    for s,layer in enumerate(layers):
        interface=enc.load_task(layer['task_export'],layer['task_sha256']);t=interface['gn_task']
        require(tasks[s]['root_context']==t['root_context'] and tasks[s]['goal']==t['goal']
            and tasks[s]['projection']==t['projection'] and tasks[s]['action_mean']==t['action_mean']
            and tasks[s]['action_std']==t['action_std'],'same original P3/native moving task, goal, identity Q and ABI')
    files=set(map(Path,old_p['bindings']))|set(map(Path,signal_p['sources']))|set(own_files())|{moving.OLD_PREPARE}
    for module in (moving,common,native,signal,core,native.generator):files.add(Path(module.__file__).resolve())
    return dict(schema=NAME+'.prepare',contract=contract(),layers=layers,tasks=tasks,policy=signal_p['policy'],
        expected_model_hash=signal_p['expected_model_hash'],modelbinding=old_p['modelbinding'],
        source_manifest=old_p['manifest'],sources={str(p.resolve()):sha(p) for p in sorted(files)},
        original_native_prepare=dict(path=str(moving.OLD_PREPARE),sha256=moving.OLD_SHA),
        original_task_signal_history=signal_p['history'],new_node_signals_recomputed=True,
        historical_cost_unknowns=None,known_only=True,P1_admission=False,P3_admission=False,freshness=False)


def exclusive(path):
    p=Path(path).absolute();require(p.parent==ROOT/'results' and p.name.startswith(NAME)
        and p.resolve()==p and not p.exists(),'new exclusive own-stem result');return p


def prepare(output):
    p=build();path=exclusive(output);payload=raw(p)
    require(len(payload)<=16*1024**2 and shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(payload),
        'bounded prepare/disk floor')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full prepare write');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path))


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external prepare SHA')
    p=signal.read(path);require(p==build() and sha(path)==expected,'complete task/source/rule/caps prepare reconstruction');return p


def source_data(p):
    manifest=old.read(p['source_manifest'])
    return old.kernel.load_source(manifest['sourcepack'],old.SOURCE_SHA,manifest['source_prepared'],manifest['source_prepare_sha256'])


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
                errors=[],fixed_denominator=4,node_denominator=16,known_tasks=2,contract=contract())
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


def audit_resources(receipt,events):
    core.audit_ledger(receipt);expected=[];stack=[]
    for op in receipt['events']:
        if op['status']=='blocked_before_dispatch':expected.append(dict(kind='resource_blocked',entry=op));continue
        expected.append(dict(kind='resource_attempt',entry=dict(op,status='attempted',ended_ns=None,error=None)))
        expected.append(dict(kind='resource_return' if op['status']=='returned' else 'resource_failure',entry=op))
    actual=[e for e in events if e['kind'].startswith('resource_')]
    require(sorted(map(raw,actual))==sorted(map(raw,expected)),'native attempt/return/failure ledger bijection')
    for e in actual:
        if e['kind']=='resource_attempt':stack.append(e['entry']['index'])
        elif e['kind']!='resource_blocked':require(stack and stack.pop()==e['entry']['index'],'actual nested native ordering')
    require(not stack,'complete disposition, no fabricated native return')
    return receipt['completed']


def audit_trajectory(task,local,phase,h,actions,path,*,program):
    """Full saved native float32 module wiring, normalization and every endpoint."""
    def arrays(e):
        a={k:signal.unblob(v) for k,v in e['arrays'].items()}
        require(all(v.dtype==np.float32 for v in a.values()),'native buffer precision, no repaired arrays')
        return a
    es=[e for e in local if e.get('phase')==phase]
    packets=[e for e in es if e['kind']==('program_packet' if program else 'packet')]
    boundaries=[e for e in es if e['kind']==('program_boundary' if program else 'boundary')]
    require(len(packets)==len(boundaries)==h,'all paid packet boundaries')
    latent=np.asarray(task['root_context']['latent_history'],np.float32)[None]
    history=np.asarray(task['root_context']['action_history'],np.float32)[None]
    mean,std=(np.asarray(task[k],np.float32) for k in ('action_mean','action_std'))
    if not program:
        origins=[e for e in es if e['kind']=='origin'];require(len(origins)==1,'one signal origin')
        origin=arrays(origins[0]);core.keys(origin,('latent','history','raw_flat','mean','std'))
        for k,v in dict(latent=latent,history=history,raw_flat=np.zeros(10*h,np.float32),mean=mean,std=std).items():
            require(np.array_equal(origin[k],v),'same exact local-node AD/nominal origin '+k)
    for step,(pe,be) in enumerate(zip(packets,boundaries,strict=True)):
        require(pe['step']==be['step']==step,'packet chronology')
        p=arrays(pe);b=arrays(be);core.keys(b,('latent','history'))
        core.keys(p,('latent','history','raw','mean','std','token','outgoing') if program else ('raw','token','outgoing'))
        a=np.asarray(actions[5*step:5*step+5],np.float32)
        token=((a-mean)/std).reshape(10);outgoing=np.concatenate((history,token[None,None]),axis=1)
        expected=dict(raw=a,token=token,outgoing=outgoing)
        if program:expected.update(latent=latent,history=history,mean=mean,std=std)
        for k,v in expected.items():require(np.array_equal(p[k],v),'raw/actual context normalization '+k)
        def get(resource,returned=False):
            kind='io_return' if returned else 'io_input'
            match=[e for e in es if e['kind']==kind and e['step']==step and
                any(k.startswith(resource+':') for k in e['arrays'])]
            require(len(match)==1,'exact single native module buffer per packet: '+resource)
            value=arrays(match[0]);keys={'action_encoder':('0',),'predict':('latent','act_emb'),
                'predictor':('0','1'),'pred_proj':('0',)}[resource]
            core.keys(value,(resource+':output',) if returned else tuple(resource+':'+k for k in keys));return value
        ai,ao=get('action_encoder'),get('action_encoder',True)
        pi,po=get('predict'),get('predict',True)
        ni,no=get('predictor'),get('predictor',True)
        qi,qo=get('pred_proj'),get('pred_proj',True)
        require(np.array_equal(ai['action_encoder:0'],outgoing) and
            np.array_equal(pi['predict:latent'],latent) and np.array_equal(pi['predict:act_emb'],ao['action_encoder:output']) and
            np.array_equal(ni['predictor:0'],latent) and np.array_equal(ni['predictor:1'],ao['action_encoder:output']) and
            np.array_equal(qi['pred_proj:0'],no['predictor:output'].reshape(3,192)) and
            po['predict:output'].shape==(1,3,192) and
            np.array_equal(qo['pred_proj:output'].reshape(1,3,192),po['predict:output']),'AE/predict/predictor/pred_proj exact native wiring')
        latent=np.concatenate((latent[:,1:],po['predict:output'][:,-1:]),axis=1);history=outgoing[:,1:]
        require(np.array_equal(b['latent'],latent) and np.array_equal(b['history'],history),'every exact native endpoint')
        if path is not None:
            require(np.array_equal(path[step]['latent_history'],latent[0]) and
                np.array_equal(path[step]['action_history'],history[0]),'saved replay path exact full3 endpoint')
    return latent[0,-1]


def audit_plan(request,result,events,resources):
    t=validate_request(request);s=request['source_slot'];mode=request['mode']
    core.keys(result,('status','responses','diagnostic','queue','bank','candidates','winner','prefix_path','commands','resource_receipt'))
    audit_resources(resources,events)
    require(resources==result['resource_receipt'],'same node resource receipt')
    allowed={'resource_attempt','resource_return','io_input','io_return','origin','packet','boundary',
        'nominal','AD_terminal','jacobian','direction','raw_backward','program_packet','program_boundary','local_signal','local_plan_final'}
    require(all(e['kind'] in allowed for e in events),'no unknown/evaluator event in model journal')
    io_ops=[op for op in resources['events'] if op['resource'] in ('predict','action_encoder','predictor','pred_proj','grad')]
    require(sum(e['kind'] in ('io_input','io_return') for e in events)==2*len(io_ops),'all and only native/grad IO')
    for e in events:
        if 'arrays' in e:
            for b in e['arrays'].values():require(signal.unblob(b).dtype==np.float32,'original actual native float32')
    # Every actual module/grad call, and only those calls, has exact linked IO.
    for op in resources['events']:
        linked=[e for e in events if e.get('operation_index')==op['index']]
        if op['resource'] in ('predict','action_encoder','predictor','pred_proj','grad'):
            require([e['kind'] for e in linked]==['io_input','io_return'] and
                all(e['scope']==op['scope'] for e in linked),'all paid native IO linked to operation')
            if op['shape'] is not None:
                first={'predict':'latent','action_encoder':'0','predictor':'0','pred_proj':'0'}[op['resource']]
                require(linked[0]['arrays'][op['resource']+':'+first]['shape']==op['shape'],'charged actual shape/sample/token axes')
        elif op['resource']=='raw_backward':require([e['kind'] for e in linked]==['raw_backward'],'paid actual raw callback')
        else:require(not linked,'no invented module/grad IO')
    require([e['response'] for e in events if e['kind']=='local_signal']==result['responses'],'actual signal response journal')
    require([r['horizon'] for r in result['responses']]==[1,3],'complete independent paid H1/H3')
    for response in result['responses']:
        h=response['horizon'];scope=dict(slot=s,horizon=h);local=[e for e in events if e.get('scope')==scope]
        require(response['binding_sha256']==digest(t['binding']) and response['parent_context_sha256']==digest(t['root_context'])
            and response['producer_sha256']==sha(signal.__file__),'fresh same-actual-node signal producer/binding')
        n=core.finite(response['nominal'],(192,),True);j=core.finite(response['jacobian'],(192,10*h),True)
        for phase in ('nominal','AD'):
            terminal=audit_trajectory(t,local,phase,h,np.zeros((5*h,2),np.float32),None,program=False)
            require(np.array_equal(n,terminal),'actual nominal/AD terminal matches saved signal')
        def values(kind):return [signal.unblob(e['arrays']['value']) for e in local if e['kind']==kind]
        require(len(values('nominal'))==len(values('AD_terminal'))==len(values('jacobian'))==1 and
            np.array_equal(values('nominal')[0],n) and np.array_equal(values('AD_terminal')[0],n) and
            np.array_equal(values('jacobian')[0],j),'full J/native response arrays')
        require([e['direction'] for e in local if e['kind']=='direction']==list(range(192)),'all192 VJP directions')
        gi=[e for e in local if e['kind']=='io_input' and 'grad:basis' in e['arrays']]
        go=[e for e in local if e['kind']=='io_return' and 'grad:output' in e['arrays']]
        rw=values('raw_backward');require(len(gi)==len(go)==len(rw)==192,'all gradient/raw callbacks')
        for i,(a,b,c) in enumerate(zip(gi,go,rw,strict=True)):
            core.keys(a['arrays'],('grad:basis','grad:output','grad:raw'));core.keys(b['arrays'],('grad:output',))
            require(np.array_equal(signal.unblob(a['arrays']['grad:basis']),np.eye(192,dtype=np.float32)[i]) and
                np.array_equal(signal.unblob(a['arrays']['grad:output']),n) and
                np.array_equal(signal.unblob(a['arrays']['grad:raw']),np.zeros(10*h,np.float32)) and
                np.array_equal(signal.unblob(b['arrays']['grad:output']),j[i]) and np.array_equal(c,j[i]),'original scalar target/raw/J row identity')
    diagnostic=signal.spectra(t,result['responses']);require(diagnostic==result['diagnostic'],'fixed same-reference chi/gain recomputation')
    node,signals,receipt=signal.runtime_signal(t,result['responses'],diagnostic)
    queue=signal.decide_root(t,mode,node,signals,receipt);require(queue==result['queue'],'unchanged same-task runtime decision')
    d=queue['decision'];h=d['horizon'];require(d['program_type']=='progress','no unimplemented objective substitution')
    count=24//(2*h);dummy=core.Ledger(node_caps());dummy.scope=dict(saved_recompute=True)
    response=next(r for r in result['responses'] if r['horizon']==h)
    bank=common.generate(t,response,count,dummy);require(bank==result['bank'],'all actual candidates predetermined by same-node J')
    require(len(result['candidates'])==count,'full selected quota, no outcome filter')
    for k,row in enumerate(result['candidates']):
        require((row['slot'],row['mode'],row['index'],row['horizon'])==(s,mode,k,h) and row['actions']==bank['actions'][k]
            and row['whole']==row['segmented'] and row['status']=='model_replayed_not_admitted','all independent full-H candidates')
        scope=dict(slot=s,mode=mode,index=k,horizon=h);local=[e for e in events if e.get('scope')==scope]
        for phase in ('whole','segmented'):audit_trajectory(t,local,phase,h,row['actions'],row[phase],program=True)
        terminal=core.checked_path(t,t['root_context'],row['actions'],row['whole'])
        require(row['cost']==core.model_cost(t,terminal),'original Q terminal objective')
    winner=min(result['candidates'],key=lambda r:(r['cost'],r['index']))
    require(result['winner']==winner and result['commands']==winner['actions'][:5] and result['prefix_path']==winner['whole'][:1],
        'minimum modelQ/stable tie and exact first5, not H3 fullcommit')
    scope=dict(slot=s,mode=mode,index=winner['index'],horizon=1,operation='selected_prefix_verification')
    local=[e for e in events if e.get('scope')==scope]
    audit_trajectory(t,local,'whole',1,result['commands'],result['prefix_path'],program=True)
    expected=node_caps();expected.update(replay_calls=2*count+1,metric_calls=count+1)
    require(resources['attempted']==resources['completed']==expected and result['status']=='sealed','exact33 predict/API sample no padding')
    require([e['result'] for e in events if e['kind']=='local_plan_final']==[result],'durable complete local plan')
    return dict(horizon=h,candidates=count,resources=resources['completed'],signed_gain=diagnostic['signed_gain'],
        saved_array_SVD_recomputations=3,FLOPs=None,CUDA_ns=None)


def load_array(output,entry):
    path=Path(entry['path']);require(path.is_relative_to(output) and path.resolve()==path and
        len(path.relative_to(output).parts)<=2 and path.stat().st_size<=1024**2 and sha(path)==entry['file_sha256'],
        'own canonical bounded native array')
    b=path.read_bytes();stream=io.BytesIO(b);a=np.load(stream,allow_pickle=False)
    require(stream.tell()==len(b) and str(a.dtype)==entry['dtype'] and list(a.shape)==entry['shape'] and
        old.array_sha(a)==entry['array_sha256'],'full native array SHA/layout');return a


def consume(output,report_sha,prepared,prepare_sha):
    output=Path(output).absolute();p=verify(prepared,prepare_sha)
    require(sha(output/'report.json')==report_sha,'external report byte SHA')
    r=signal.read(output/'report.json',contract()['final_reserve_bytes']);rows=r['rows'];nodes=r['nodes']
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['fixed_denominator']==4 and
        r['node_denominator']==16 and r['known_tasks']==2 and
        [(x['episode_id'],x['source_slot'],x['mode']) for x in rows]==
        [(x['episode_id'],x['source_slot'],x['mode']) for x in roster()] and
        [(x['episode_id'],x['boundary']) for x in nodes]==[(x['episode_id'],x['boundary']) for x in node_roster()],
        'all4 methods/tasks and16 slots including failed/unreached')
    require(r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha and
        old.read(output/'execution_binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'execution prepare binding')
    for key in ('P1_admission','P3_admission','freshness','scientific_admission'):require(r[key] is False,'no scientific/fresh release')
    # Startup failure is evidence only, never a successful engineering run.
    if 'journal_sha256' not in r:
        require(r['status']=='partial_or_failed','no invented terminal journal')
        return dict(status='startup_or_incomplete_preserved',fixed_denominator=4,node_denominator=16,complete=False)
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
        path=output/(slot['episode_id']+f"_b{slot['boundary']:02}")/'events.jsonl'
        require(sha(path)==slot['native_journal_sha256'],'native journal bytes including partial work')
        es,tail=signal.journal(path);require(tail==slot['native_journal_tail'],'native journal prefix/terminal tail')
        audit_resources(slot['resource_receipt'],es)
    if r['status']!='known_same_feedback_recorded' or r['errors'] or r['pending']:
        return dict(schema=NAME+'.consume',status='saved_partial_preserved',complete=False,
            fixed_denominator=4,node_denominator=16,counts=counts,rows=rows,nodes=nodes,
            neural_paths_complete=False,P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False)
    references=source_data(p)['references']
    audited=[];latent_by_episode={}
    for row in rows:
        require(old.read(output/(row['episode_id']+'_episode.json'))==row,'durable episode')
        s=row['source_slot'];base=p['tasks'][s];eid=row['episode_id'];same=[e for e in events if e.get('episode_id')==eid]
        ref=references[s]
        warm=[e['receipt'] for e in same if e['kind']=='fresh_observation' and e['receipt']['phase']=='warmup']
        require([x['step'] for x in warm]==list(range(row['warmup_returned']+1)),'complete actual warmup trajectory')
        if row['initial_exact']:
            require(row['warmup_returned']==10,'same root requires completed warmup10')
            for k,b in enumerate((0,5,10)):
                ob=warm[b]
                require(ob['pixel_sha256']==old.array_sha(ref['history'][k]) and
                    np.array_equal(ob['state'],ref['states'][k]) and ob['snapshot']==ref['dynamics'][k],
                    'original known root/warmup actual RGB/state/dynamics identity')
        observed=[e['receipt'] for e in same if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan']
        require([x['step'] for x in observed]==list(range(row['control_returned']+1)) if row['initial_exact'] else not observed,
            'full actual control observation trajectory')
        dispatch=[e for e in same if e['kind']=='authorized_dispatch'];returns=[e for e in same if e['kind']=='control_return']
        require(len(dispatch)==len(returns)==len(row['executed_raw'])==row['control_returned'],'no unreturned action fabricated')
        require(row['full_horizon']==(row['status']=='completed20') and (not row['full_horizon'] or row['control_returned']==20),
            'completed20 is actual full horizon, not first5')
        if row['full_horizon']:require([e['step'] for e in row['encodings']]==[0,5,10,15,20],'all five real encodings')
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
            if b==0:cached['latent_history'][-1]=z.tolist();cached['action_history']=moving.normalized(base,actual)
            else:cached=moving.roll(cached,z,actual,base)
            slot=next(x for x in nodes if x['episode_id']==eid and x['boundary']==b)
            if slot['status']!='sealed':continue
            request=slot['request'];validate_request(request,base)
            require(request['task']['root_context']==cached and request['raw10']==actual.tolist(),
                'own full actual observed history, never root-J or predicted-child reuse')
            result=slot['receipt'];journal=output/ns/'events.jsonl'
            require(sha(journal)==slot['native_journal_sha256'],'all native journal bytes')
            native_events,tail=signal.journal(journal);require(tail==slot['native_journal_tail'],'all native chain tail')
            audited.append(dict(episode_id=eid,boundary=b,**audit_plan(request,result,native_events,slot['resource_receipt'])))
            seal=slot['seal'];require(old.read(output/(ns+'_seal.json'))==seal and
                seal['seal_sha256']==digest({k:v for k,v in seal.items() if k!='seal_sha256'}) and
                seal['request_sha256']==digest(request) and seal['plan_sha256']==digest(result) and
                seal['commands']==result['commands'] and seal['commands_sha256']==digest(result['commands']),
                'durable request/selected raw5 seal')
            selected=[x for x in dispatch if x['packet_boundary']==b]
            sealed=next(e['index'] for e in same if e['kind']=='boundary_sealed' and e['boundary']==b)
            encoded_event=next(e['index'] for e in same if e['kind']=='boundary_encoded' and e['actual_step']==b)
            require(ob['observed_event_index']<encoded_event<sealed and
                [x['step'] for x in selected]==list(range(b+1,b+1+len(selected))) and len(selected)<=5 and
                all(x['index']>sealed for x in selected) and [x['command'] for x in selected]==result['commands'][:len(selected)],
                'actual encode/compile/seal before first5 dispatch; no fullcommit H3')
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
            require(ret['reward']==-metric['native_distance'] and ret['terminated']==metric['native_success'],
                'original native agent+object position and angle outcome; no object-only substitution')
        if returns:
            metric=old.kernel.metric(ref['goal'],observed[-1]['state'])
            require(metric==row['final_metrics'] and metric['native_success']==row['native_success'] and
                row['terminated']==returns[-1]['terminated'] and row['truncated']==returns[-1]['truncated'],
                'actual terminal20 or native-stop outcome retained')
    clean=r['status']=='known_same_feedback_recorded' and not r['errors'] and not r['pending']
    if clean:
        require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'] and
            r['final_check']['model_unchanged'] is True and r['source_manifest_sha256']==digest(p['sources']),
            'unchanged actual frozen model/weights/source closure')
        require(r['precision_before']==r['precision_after']==dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,
            math=True,flash=False,efficient=False,cudnn_sdp=False),'actual MATH/TF32/autocast guards')
        require(type(r['wall_seconds']) in (int,float) and 0<=r['wall_seconds']<contract()['wall_seconds'] and
            len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=contract()[k] for v,k in
                zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),
            'measured whole-job wall/CUDA memory caps')
        require(r['per_episode_neural']==node_accounting(nodes),'each arm actual API/sample/AD/SVD/wall axes')
        require(r['completed20']==sum(x['full_horizon'] for x in rows) and
            r['equal_actual_execution_horizon']==all(rows[2*s]['control_returned']==rows[2*s+1]['control_returned'] for s in (0,1)),
            'actual all20/equal-horizon outcomes not assumed')
        require(counts['public_step']['completed']==sum(x['warmup_returned']+x['control_returned'] for x in rows)
            and counts['physics_step']['completed']==2*counts['reset']['completed']+10*counts['public_step']['completed'],
            'full physical census, not just compiler cost')
        if all(x['full_horizon'] for x in rows):
            require(counts['encoder']==counts['projector']==dict(attempted=20,completed=20) and
                counts['public_step']==dict(attempted=120,completed=120) and
                counts['explicit_render']==dict(attempted=128,completed=128) and
                counts['goal_bind']==counts['close']==dict(attempted=4,completed=4),'full all20 physical/visual census')
    return dict(schema=NAME+'.consume',status='saved_same_feedback_verified_NOT_admitted' if clean else 'saved_partial_preserved',
        complete=clean,fixed_denominator=4,node_denominator=16,audited=audited,counts=counts,
        completed20=r.get('completed20'),equal_actual_execution_horizon=r.get('equal_actual_execution_horizon'),
        native_metrics=[x['final_metrics'] for x in rows],equal_FLOPs=False,equal_wall=False,
        per_episode_neural=r['per_episode_neural'],
        P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False,independent_neural_replay=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+key)
    parser.add_argument('--authorize-known-cuda-and-physics',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(schema=verify(a.prepared,a.prepare_sha)['schema'],verified=True)
    elif a.command=='consume':result=consume(a.output,a.report_sha,a.prepared,a.prepare_sha)
    else:
        require(a.authorize_known_cuda_and_physics,'MAIN explicit CUDA+physics authorization')
        result=run(a.prepared,a.prepare_sha,a.output,authorization=AUTH)
    print(raw(result).decode())
    if a.command=='run' and result['status']!='known_same_feedback_recorded':raise SystemExit(2)


if __name__=='__main__':main()
