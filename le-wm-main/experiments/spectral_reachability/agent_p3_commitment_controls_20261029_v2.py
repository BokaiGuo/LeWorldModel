"""Six-arm complete-commit HOST CUDA/native driver and own strict saved consumer."""
from __future__ import annotations
import argparse
from copy import deepcopy
from collections import Counter, defaultdict
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

from experiments.spectral_reachability import agent_p3_commitment_controls_20261029_v1 as controls
from experiments.spectral_reachability import agent_p3_fullcommit_live_20261028_v1 as legacy

NAME='agent_p3_commitment_controls_20261029_v2'
ROOT=common.ROOT
AUTH='MAIN_KNOWN2_COMMITMENT_CONTROLS_V2_CUDA_AND_PHYSICS'
signal,native,core=common.signal,common.native,common.core
old=moving.old
require,sha,digest,raw=core.require,signal.sha,core.digest,core.canonical
BOUNDARIES=(5,10,15)
MODES=controls.MODES


from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v1 as template
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v2 as harden
from experiments.spectral_reachability import agent_p3_child_native_physical_20261028_v1 as frames


def contract():
    c=legacy.contract()
    c.update(episodes=6,node_slots=18,rule=controls.contract(),
        program_predict_caps=dict(fixedH1=27,fixedH3=51,adaptive=51),
        inclusive_predict_caps=dict(fixedH1=51,fixedH3=75,adaptive=75),
        physical_caps=dict(constructor=6,reset=18,set_state=36,goal_render=18,
            goal_setter=24,goal_bind=6,public_step=180,physics_step=1836,
            explicit_render=192,close=6),
        visual_caps=dict(encoder=24,projector=24),wall_seconds=2700,
        output_bytes=768*1024**2,final_reserve_bytes=96*1024**2,
        budget_match='notmatched',independent_confirmation=False,
        initialization='each arm independently constructor/reset/place/reset/warm10; no poststate',
        production_enabled=True)
    return c

def predict_axes(n):
    c=dict.fromkeys(core.AXES,0)
    c.update(predict_API=n,predict_samples=n,predict_context_tokens=3*n,
        action_encoder_calls=n,action_encoder_samples=n,action_encoder_context_tokens=3*n,
        predictor_calls=n,predictor_samples=n,predictor_context_tokens=3*n,
        pred_proj_calls=n,pred_proj_rows=3*n)
    return c


def costs(h=None,n=None,*,schedule=False):
    c=predict_axes(8+(h*(2*n+1) if h is not None else 0))
    c.update(jacobian_API=2,VJP_directions=384,grad_API=384,raw_backward_callbacks=384,
        array_SVD_calls=2+int(h is not None),solve_calls=int(h is not None),
        signal_compile_calls=2,program_compile_calls=int(h is not None),
        replay_calls=2*n+1 if h is not None else 0,metric_calls=n or 0,schedule_decisions=int(schedule))
    return c


def arm_caps(mode):
    return controls.caps(mode)

def node_caps():return arm_caps('adaptive')


def node_roster():
    return [dict(episode_id=f's{s}_{m}',source_slot=s,mode=m,boundary=b,status='not_reached',
        request=None,receipt=None,error=None) for s in (0,1) for m in MODES for b in BOUNDARIES]


def remaining_caps(mode,slots):
    used={k:sum(x['resource_receipt']['attempted'][k] for x in slots if x.get('resource_receipt')) for k in core.AXES}
    c=arm_caps(mode);require(all(used[k]<=c[k] for k in c),'nonrefundable arm resource cap')
    return {k:c[k]-used[k] for k in c}


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
                caps=arm_caps(mode),FLOPs=None,CUDA_ns=None,backward_engine_calls=None,
                inclusive_operation_wall_ns={resource:sum(r['inclusive_operation_wall_ns'].get(resource,0)
                    for r in receipts) for resource in sorted({k for r in receipts for k in r['inclusive_operation_wall_ns']})},
                wall_semantics='nested inclusive; never summed into runtime or FLOPs'))
    return out


def roster():
    example=template.roster()[0]
    rows=[]
    for s in (0,1):
        for mode in MODES:
            row=deepcopy(example)
            row.update(episode_id=f's{s}_{mode}',source_slot=s,mode=mode,
                commitments=[],prefix_exact=False,refusal=None,actual_Q20=None)
            rows.append(row)
    return rows

class Store(moving.Store):
    def __init__(self,path):
        self.path=Path(path).absolute();self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME) and
            self.path.resolve()==self.path and not self.path.exists(),'new exclusive runtime output')
        require(shutil.disk_usage(self.path.parent).free>=contract()['disk_floor_bytes']+contract()['output_bytes'],
            '768MiB output plus200MiB floor before start')
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


def validate_request(request,base):
    return controls.validate_request(request,base)


def plan_node(request,base,factory,store,ledger,*,permit_ram=False):
    return controls.plan_node(request,base,factory,store,ledger,permit_ram=permit_ram)

def prefix_binding(layer):
    a=old.raw5(layer['commands'][:5])
    return dict(commands=a.tolist(),raw_float32_sha256=old.array_sha(a),
        source_selection_sha256=layer['selection_sha256'])


def initialization_intent(ref,plan):
    """Pre-dispatch recipe identity, not a snapshot-restoration shortcut."""
    return dict(reset_seed=ref['slot']['reset_seed'],root_seed=ref['slot']['root_seed'],
        recipe_sha256=digest(plan),warmup_raw_sha256=old.array_sha(ref['raw_history']),
        reference_sha256=digest(dict(slot=ref['slot'],root_sha256=ref['root_sha256'],
            history_sha256=old.array_sha(ref['history']),states_sha256=old.array_sha(np.asarray(ref['states'])),
            dynamics=ref['dynamics'],goal=np.asarray(ref['goal']).tolist(),
            goal_rgb_sha256=old.array_sha(ref['goal_rgb']))))


def native_return_contract(goal,state,reward,terminated,truncated):
    """Pinned env.py eval_state: native7 reward is NOT wrapped angle cost.

    Preserve even the source's negative min-angle for differences beyond2pi;
    wrapped-angle diagnostics never redefine native success or reward.
    """
    g,s=old.source.vector(goal),old.source.vector(state)
    pos=np.linalg.norm(g[:4]-s[:4]);a=np.abs(g[4]-s[4])
    a=np.minimum(a,2*np.pi-a)
    success=bool(pos<20 and a<np.pi/9);distance=float(np.linalg.norm(g-s))
    require(type(reward) in (int,float) and type(terminated) is bool and type(truncated) is bool
        and reward==-distance and terminated is success,'native7 reward vs native XY+angle termination')
    return dict(native_success=success,native7_distance=distance,native7_reward=-distance)


def episode(env,ref,data,layer,base,row,slots,visual,runtime,meter,store,planner_factory,*,permit_ram=False):
    if not moving.setup(env,ref,data['plan'],row,meter,store):return
    binding=prefix_binding(layer);row['prefix']=binding;store.put(row['episode_id']+'_prefix_commit.json',binding)
    obs=moving.packet(env,binding['commands'],start=0,row=row,ref=ref,meter=meter,store=store)
    if obs is None:return
    match={k:obs[k] for k in ('pixel_sha256','state','snapshot','goal','goal_rgb_sha256','goal_pose')}
    paired=data.setdefault('paired_prefix',{})
    if row['mode']=='fixedH1':paired[row['source_slot']]=deepcopy(match)
    else:require(paired.get(row['source_slot'])==match,'exact fresh same-source prefix before feedback')
    row['completed_packets']+=1
    row['prefix_exact']=True;context=deepcopy(base['root_context']);active=None
    for boundary in BOUNDARIES:
        slot=next(x for x in slots if x['episode_id']==row['episode_id'] and x['boundary']==boundary)
        slot['status']='attempted';meter.region='encoding'
        z,observation=moving.encode_boundary(visual,runtime,data['visual_tasks'][row['source_slot']],layer,row,obs,
            env.unwrapped,meter,store,boundary)
        raw10=np.concatenate((ref['raw_history'],np.asarray(row['executed_raw'],np.float32).reshape(-1,2)))[-10:]
        context=moving.roll(context,z,raw10,base)
        pending=None if active is None else {k:active[k] for k in ('start','end','seal_sha256')}
        request=controls.make_request(base,context,slot=row['source_slot'],mode=row['mode'],
            boundary=boundary,raw10=raw10,observation=observation,commitment=pending)
        validate_request(request,base);slot['request']=request;meter.region='compiling'
        ns=row['episode_id']+f'_b{boundary:02}';store.put(ns+'_request.json',request)
        state=old.state_seal(env.unwrapped);journal=NativeJournal(store,ns,runtime['check_limits'])
        prior=[x for x in slots if x['episode_id']==row['episode_id'] and x is not slot]
        ledger=core.Ledger(remaining_caps(row['mode'],prior),sink=journal.event);planned=None
        try:
            planned=plan_node(request,base,planner_factory,journal,ledger,permit_ram=permit_ram)
            require(old.state_seal(env.unwrapped)==state,'planning cannot mutate native state/goal')
        finally:
            journal.close();slot['native_journal_sha256']=sha(journal.path);slot['native_journal_tail']=journal.tail
            slot['resource_receipt']=ledger.receipt() if planned is None else planned['resource_receipt']
        slot['receipt']=planned;slot['status']=planned['status']
        seal=dict(schema=NAME+'.dispatch',episode_id=row['episode_id'],boundary=boundary,
            request_sha256=digest(request),plan_sha256=digest(planned),commands=planned['commands'],
            commands_sha256=digest(planned['commands']),native_journal_sha256=slot['native_journal_sha256'],
            status=planned['status'],commitment=pending)
        seal['seal_sha256']=digest(seal);slot['seal']=seal;store.put(ns+'_seal.json',seal)
        store.emit('boundary_sealed',episode_id=row['episode_id'],boundary=boundary,seal_sha256=seal['seal_sha256'])
        if planned['status']=='refused':
            row.update(status='contract_refused',refusal=dict(boundary=boundary,reason=planned['refusal']));return
        if active is None:
            require(planned['status']=='sealed','idle requires complete full-H release')
            active=dict(start=boundary,end=boundary+len(planned['commands']),seal_sha256=seal['seal_sha256'],
                commands=deepcopy(planned['commands']))
            row['commitments'].append(deepcopy(active))
        else:require(planned['status']=='observed_commitment' and planned['commands'] is None,'paid signal cannot replace active commitment')
        offset=boundary-active['start'];commands=active['commands'][offset:offset+5]
        require(len(commands)==5,'whole committed packet, never padding')
        obs=moving.packet(env,commands,start=boundary,row=row,ref=ref,meter=meter,store=store)
        if row['control_returned']==boundary+5:row['completed_packets']+=1
        if obs is None:
            if row['control_returned']==20:obs=row['last_observation'];break
            return
        if boundary+5==active['end']:active=None
    require(row['control_returned']==20,'full requested public20 actually returned')
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
        fixed_denominator=6,node_denominator=18,known_tasks=2,P1_admission=False,P3_admission=False,
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
        ('completed20','contract_refused','native_stopped_warmup','native_stopped_control') for r in rows)
    report.update(status='known_fullcommit_recorded' if clean else 'partial_or_failed',counts=meter.counts,
        pending=meter.pending,completed20=sum(r['full_horizon'] for r in rows),
        equal_actual_execution_horizon=all(len({r['control_returned'] for r in rows if r['source_slot']==s})==1 for s in (0,1)),
        wall_seconds=time.monotonic()-data.get('job_started',meter.started),budget_match='notmatched',equal_FLOPs=False,equal_wall=False)
    store.finalizing=True
    store.emit('run_finished',status=report['status'],counts=meter.counts,nodes_sha256=digest(slots))
    report.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    return report


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build():
    accepted=controls.build()  # mandatory full original saved reconstruction
    p=signal.read(controls.PREPARED)
    p.update(schema=NAME+'.prepare',contract=contract(),
        engineering_authority=accepted['authority'],
        controls_contract=controls.contract(),historical_seen_known2=True)
    files=set(map(Path,p['sources']))|set(own_files())|set(map(Path,accepted['bindings']))
    p['sources']={str(f.resolve()):sha(f) for f in sorted(files)}
    return p

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
                errors=[],fixed_denominator=6,node_denominator=18,known_tasks=2,contract=contract())
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


audit_resources=legacy.audit_resources
audit_trajectory=legacy.audit_trajectory

def audit_plan(request,result,events,resources,base):
    t=validate_request(request,base);s=request['source_slot'];mode=request['mode']
    core.keys(result,('schema','request_sha256','status','responses','diagnostic','decision','bank','candidates','winner','selected_path','commands','resource_receipt','refusal','error'))
    require(result['schema']==controls.NAME+'.node_result' and result['request_sha256']==digest(request)
        and result['error'] is None,'frozen node schema/request/nonfailure identity')
    audit_resources(resources,events)
    require(resources==result['resource_receipt'],'same node resource receipt')
    allowed={'resource_attempt','resource_return','io_input','io_return','origin','packet','boundary',
        'nominal','AD_terminal','jacobian','direction','raw_backward','program_packet','program_boundary','local_signal',controls.NAME+'.node_final'}
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
    final=[e['result'] for e in events if e['kind']==controls.NAME+'.node_final']
    require(final==[result],'complete durable local result')
    if request['commitment'] is not None:
        require(result['status']=='observed_commitment' and result['decision'] is None and
            result['commands'] is None and result['bank'] is None and result['candidates']==[] and
            result['winner'] is None and result['selected_path'] is None and result['refusal'] is None,
            'intermediate paid observation never replaces commitment')
        expected=costs()
    else:
        node,signals,receipt=signal.runtime_signal(t,result['responses'],diagnostic)
        d=controls.decision(t,mode,node,signals,receipt)
        require(d==result['decision'],'frozen three-strategy schedule')
        h=d['horizon'];count=d['candidates']
        reason=('unsupported_original_decision' if d['program_type']!=d['extension_type'] or
            d['program_type']!='progress' or not d['continue_expanding'] else
            'fullH_does_not_fit' if 5*h>20-request['boundary'] else None)
        if reason is not None:
            require(result['status']=='refused' and result['refusal']==reason and
                result['commands'] is None and result['bank'] is None and result['candidates']==[]
                and result['winner'] is None and result['selected_path'] is None,'whole-node refusal; no fallback')
            expected=costs(schedule=True)
        else:
            dummy=core.Ledger(node_caps());dummy.scope=dict(saved_recompute=True)
            bank=legacy.generate(t,next(r for r in result['responses'] if r['horizon']==h),count,dummy)
            require(bank==result['bank'] and len(result['candidates'])==count,'natural full candidate denominator')
            for k,row in enumerate(result['candidates']):
                require(row['index']==k and
                    row['actions']==bank['actions'][k] and row['whole']==row['segmented'] and
                    row['status']=='returned','all fullH candidate identities')
                scope=dict(slot=s,mode=mode,index=k,horizon=h,phase='program')
                local=[e for e in events if e.get('scope')==scope]
                for phase in ('whole','segmented'):
                    audit_trajectory(t,local,phase,h,row['actions'],row[phase],program=True)
                terminal=core.checked_path(t,t['root_context'],row['actions'],row['whole'])
                require(row['cost']==core.model_cost(t,terminal),'same original Q fullH score')
            winner=min(result['candidates'],key=lambda r:(r['cost'],r['index']))
            require(result['winner']==winner and result['commands']==winner['actions'] and
                len(result['commands'])==5*h and result['selected_path']==winner['whole'] and
                result['status']=='sealed' and result['refusal'] is None,'stable winner and FULL selected path')
            local=[e for e in events if e.get('scope')==dict(slot=s,mode=mode,index=winner['index'],horizon=h,phase='selected')]
            audit_trajectory(t,local,'whole',h,result['commands'],result['selected_path'],program=True)
            expected=costs(h,count,schedule=True)
    require(resources['attempted']==resources['completed']==expected,'exact natural fullH costs; API/samples distinct')
    return dict(status=result['status'],horizon=result['decision']['horizon'] if result['decision'] else None,
        candidates=len(result['candidates']),resources=resources['completed'],signed_gain=diagnostic['signed_gain'],FLOPs=None)


def load_array(output,entry):
    path=Path(entry['path']);require(path.is_relative_to(output) and path.resolve()==path and
        len(path.relative_to(output).parts)<=2 and path.stat().st_size<=1024**2 and sha(path)==entry['file_sha256'],
        'own canonical bounded native array')
    b=path.read_bytes();stream=io.BytesIO(b);a=np.load(stream,allow_pickle=False)
    require(stream.tell()==len(b) and str(a.dtype)==entry['dtype'] and list(a.shape)==entry['shape'] and
        old.array_sha(a)==entry['array_sha256'],'full native array SHA/layout');return a


FIELDS=dict(harden.FIELDS,initialization_intent={'episode_id','reset_seed','root_seed',
    'recipe_sha256','warmup_raw_sha256','reference_sha256'})
COMMON,REGIONS=harden.COMMON,harden.REGIONS


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


def terminal_disposition(row):
    """A short run is never a clean stop merely because its label says so."""
    require(type(row['terminated']) is bool and type(row['truncated']) is bool and
        type(row['initial_exact']) is bool and type(row['prefix_exact']) is bool and
        type(row['full_horizon']) is bool,'actual native booleans, not truthy labels')
    status=row['status'];control=row['control_returned'];warm=row['warmup_returned']
    stopped=row['terminated'] or row['truncated']
    if status=='completed20':
        require(control==20 and warm==10 and row['initial_exact'] and row['full_horizon'],
            'completed20 requires actual full independent initialization/control')
    elif status=='native_stopped_control':
        require(1<=control<20 and stopped and warm==10 and row['initial_exact'] and
            not row['full_horizon'],'short control only on actual native stop')
    elif status=='native_stopped_warmup':
        require(1<=warm<=10 and control==0 and stopped and not row['initial_exact'] and
            not row['full_horizon'],'warmup native stop, never a task success certification')
    else:
        require(status=='contract_refused' and control in BOUNDARIES and warm==10 and
            row['initial_exact'] and not stopped and not row['full_horizon'],
            'short clean nonnative termination requires explicit full-node refusal')
    require(row['prefix_exact']==bool(row['encodings']),
        'clean reached feedback requires same-root prefix proof')


def census(events, report):
    """Account for EVERY event, every resource, every episode, including prefixes.

    Clean native-stop/refusal is complete engineering evidence, not completed20. Partial
    prefixes retain attempts/pending and are never granted complete census.
    """
    rows = report['rows']; nodes = report['nodes']; ids = [r['episode_id'] for r in roster()]
    require([r['episode_id'] for r in rows] == ids, 'fixed six episode denominator')
    require([(n['episode_id'], n['boundary']) for n in nodes] ==
            [(n['episode_id'], n['boundary']) for n in node_roster()], 'fixed eighteen node denominator')
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
            require(active < 6 and e['receipt'] == rows[active], 'exact ordered episode close, no duplicate')
            grouped[ids[active]].append(e); active += 1; continue
        require(active < 6, 'no event after six episode closes')
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
    require(active == 6 and not stack and not pending, 'all six episodes closed with no pending operations')
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
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['fixed_denominator']==6 and
        r['node_denominator']==18 and r['known_tasks']==2 and
        [(x['episode_id'],x['source_slot'],x['mode']) for x in rows]==
        [(x['episode_id'],x['source_slot'],x['mode']) for x in roster()] and
        [(x['episode_id'],x['boundary']) for x in nodes]==[(x['episode_id'],x['boundary']) for x in node_roster()],
        'all6 arms and18 slots including failed/unreached')
    require(r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha and
        old.read(output/'execution_binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'execution prepare binding')
    for key in ('P1_admission','P3_admission','freshness','scientific_admission'):require(r[key] is False,'no scientific/fresh release')
    # Startup failure is evidence only, never a successful engineering run.
    if 'journal_sha256' not in r:
        require(r['status']=='partial_or_failed','no invented terminal journal')
        return dict(status='startup_or_incomplete_preserved',fixed_denominator=6,node_denominator=18,complete=False)
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
            fixed_denominator=6,node_denominator=18,counts=counts,rows=rows,nodes=nodes,
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
            require(sha(journal)==slot['native_journal_sha256'],'all native journal bytes')
            native_events,tail=signal.journal(journal);require(tail==slot['native_journal_tail'],'all native chain tail')
            audited.append(dict(episode_id=eid,boundary=b,**audit_plan(request,result,native_events,slot['resource_receipt'],base)))
            seal=slot['seal'];require(old.read(output/(ns+'_seal.json'))==seal and
                seal['seal_sha256']==digest({k:v for k,v in seal.items() if k!='seal_sha256'}) and
                seal['request_sha256']==digest(request) and seal['plan_sha256']==digest(result) and
                seal['commands']==result['commands'] and seal['commands_sha256']==digest(result['commands']),
                'durable request/selected fullH seal')
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
                    seal_sha256=origin['seal']['seal_sha256'],commands=origin['receipt']['commands']),
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
        committed=[]
        for n in [x for x in nodes if x['episode_id']==row['episode_id'] and x['status']=='sealed']:
            c=n['receipt']['commands']
            committed.append(dict(start=n['boundary'],end=n['boundary']+len(c),seal_sha256=n['seal']['seal_sha256'],commands=c))
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
            require(counts['encoder']==counts['projector']==dict(attempted=24,completed=24) and
                counts['public_step']==dict(attempted=180,completed=180) and
                counts['explicit_render']==dict(attempted=192,completed=192) and
                counts['goal_bind']==counts['close']==dict(attempted=6,completed=6),'full all20 physical/visual census')
    return dict(schema=NAME+'.consume',status='saved_fullcommit_verified_NOT_admitted' if clean else 'saved_partial_preserved',
        complete=clean,fixed_denominator=6,node_denominator=18,audited=audited,counts=counts,
        completed20=r.get('completed20'),equal_actual_execution_horizon=r.get('equal_actual_execution_horizon'),
        native_metrics=[x['final_metrics'] for x in rows],budget_match='notmatched',equal_FLOPs=False,equal_wall=False,
        per_episode_neural=r['per_episode_neural'],physical_census=physical_census,RGB_preprocessing=rgb_audit,
        P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False,independent_neural_replay=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+key)
    parser.add_argument('--receipt')
    parser.add_argument('--authorize-known-cuda-and-physics',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(schema=verify(a.prepared,a.prepare_sha)['schema'],verified=True)
    elif a.command=='consume':result=consume(a.output,a.report_sha,a.prepared,a.prepare_sha)
    else:
        require(a.authorize_known_cuda_and_physics,'MAIN explicit CUDA+physics authorization')
        result=run(a.prepared,a.prepare_sha,a.output,authorization=AUTH)
    if a.receipt:
        require(a.command=='consume','receipt only for saved consumer')
        target=exclusive(a.receipt);payload=raw(result)
        with target.open('xb') as f:
            require(f.write(payload)==len(payload),'whole exclusive receipt');f.flush();os.fsync(f.fileno())
        signal.fsync_dir(target.parent)
        result=dict(path=str(target),sha256=sha(target),status=result['status'],complete=result['complete'])
    print(raw(result).decode())
    if a.command=='run' and result['status']!='known_fullcommit_recorded':raise SystemExit(2)


if __name__=='__main__':main()


