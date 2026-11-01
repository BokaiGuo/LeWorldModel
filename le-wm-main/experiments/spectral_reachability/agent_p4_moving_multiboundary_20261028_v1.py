"""MAIN-only moving 20-step I/R driver: feedback at 5,10,15, no CPU fallback.

Reuses frozen native/visual/durable machinery and bounded-ridge mathematics.
Only the rolling-history adapter is new; no goal/Q/model fitting or admission.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_live_feedback_physical_20261027_v1 as old
from experiments.spectral_reachability import p1_moving_live_suffix_cuda_20261027_v1 as suffix

NAME='agent_p4_moving_multiboundary_20261028_v1'
ROOT=old.ROOT
AUTH='MAIN_KNOWN_MOVING_MULTIBOUNDARY_CUDA_AND_PHYSICS_ONLY'
OLD_PREPARE=ROOT/'results/p1_moving_live_feedback_physical_20261027_v1_prepare_main01.json'
OLD_SHA='bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5'
OLD_SOURCE_SHA='6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9'
BOUNDARIES=(5,10,15)
MODES=('I','R')
require,sha,read,raw,source=old.require,old.sha,old.read,old.raw,old.source
F=suffix.frozen


def contract():
    return dict(layers=[list(x) for x in old.LAYERS],modes=list(MODES),feedback_boundaries=list(BOUNDARIES),
        episodes=8,independent_tasks=2,warmup_public_steps=10,control_public_steps=20,
        compile_slots=24,packet_steps=5,with_target=False,terminal_encode_step=20,
        context='own cached original RGB latents + each own fresh boundary RGB; two actual raw packets',
        control='R uses own rolling observed history; I only its own predicted rolling history',
        goal='original same-task goal192',Q=dict(kind='identity',dimension=192),
        damping=F.compiler.DAMPING,packet_logit_cap=float(.75*np.sqrt(5)),
        diagnostics='original suffix5 fixed at all boundaries; never selected or dispatched as fallback',
        physical_caps=dict(constructor=8,reset=24,set_state=48,goal_render=24,goal_setter=32,
            goal_bind=8,public_step=240,physics_step=2448,explicit_render=256),
        neural_caps=dict(encoder=32,projector=32,predict=120,predictor=120,action_encoder=120,
            pred_proj=120,ad_forward=24,jacobian=24,svd=24,jacobian_api=24,grad_api=24,vjp_directions=4608),
        per_completed_episode=dict(encoder=4,projector=4,predict=15,jacobian=3,svd=3,
            control_public_steps=20,warmup_public_steps=10),
        wall_seconds=600,output_cap_bytes=old.CAP,disk_floor_bytes=old.FLOOR,
        final_reserve_bytes=old.FINAL,cuda_allocated_cap=3*1024**3,cuda_reserved_cap=4*1024**3,
        flops=None,cuda_kernel_calls=None,device_time_ns=None,inherited_compute=None,
        equal_execution_horizon=True,equal_total_compute=False,unknown_internal_failed_work=None,
        P1_admission=False,P4_admission=False,independent_confirmation=False,training=False)


class Meter(old.Meter):
    def __init__(self,store,job_check=lambda:None):
        super().__init__(store,job_check)
        self.caps=contract()['physical_caps']|contract()['neural_caps']
        self.counts={k:dict(attempted=0,completed=0) for k in self.caps}


class Store(old.Store):
    def png_bytes(self,receipt):
        p=Path(receipt['path'])
        require(p.parent==self.path and p.stat().st_size<1024**2 and sha(p)==receipt['file_sha256'],
            'own bounded fresh PNG before legacy visual bridge')
        return p.read_bytes()


def normalized(task,raw10):
    a=np.asarray(raw10)
    require(a.shape==(10,2) and np.isfinite(a).all() and np.array_equal(a,a.astype(np.float32))
        and (abs(a)<=1).all(),'two complete own raw packets, no clipping')
    return ((a.astype(np.float32)-np.asarray(task['action_mean'],np.float32))/
        np.asarray(task['action_std'],np.float32)).reshape(2,10).tolist()


def roll(context,z,raw10,task):
    z=old.native(np.asarray(z,np.float32));F.core.context(context)
    out=dict(context,latent_history=deepcopy(context['latent_history'][1:])+[z.tolist()],
        action_history=normalized(task,raw10))
    F.core.context(out);return out


def validate_request(request):
    suffix.fields(request,'schema episode_id source_slot prefix_arm mode boundary context3 actualrawhistory10 diagnosticraw5 goal modelbinding observation','rolling compiler input')
    p=deepcopy(request)
    require(p['schema']==NAME+'.compile' and p['mode'] in MODES and type(p['boundary']) is int
        and p['boundary'] in BOUNDARIES,'fixed real feedback boundary')
    require(type(p['source_slot']) is int and p['source_slot'] in (0,1) and p['prefix_arm'] in old.ARMS,
        'fixed moving layer')
    require(type(p['episode_id']) is str and p['episode_id']==episode_id(p['source_slot'],p['prefix_arm'],p['mode']),
        'own episode identity')
    F.core.context(p['context3']);suffix.fields(p['context3'],
        'layout context_length latent_dim action_dim latent_history action_history','context')
    suffix.array(p['context3']['latent_history'],(3,192));suffix.array(p['goal'],(192,))
    raw10=suffix.array(p['actualrawhistory10'],(10,2));diag=old.raw5(p['diagnosticraw5'])
    require((abs(raw10)<=1).all() and (abs(diag)<1).all() and
        np.linalg.norm(np.arctanh(diag.astype(float)))<=float(.75*np.sqrt(5))+1e-6,'original raw/logit domain')
    mb=p['modelbinding'];require(mb==suffix.modelbinding(),'exact original model/source/normalizer')
    require(p['context3']['action_history']==normalized(mb,raw10),'full rolling actual raw normalization')
    suffix.fields(p['observation'],'boundary png_sha256 pixel_sha256 encoding_sha256 observed_index encoded_index observed_latent_sha256','observation hashes only')
    obs=p['observation'];require(obs['boundary']==p['boundary'] and type(obs['observed_index']) is int
        and type(obs['encoded_index']) is int and 0<=obs['observed_index']<obs['encoded_index'],
        'real observation before current encoding')
    for key in ('png_sha256','pixel_sha256','encoding_sha256','observed_latent_sha256'):F.core.hash_value(obs[key])
    if p['mode']=='R':require(obs['observed_latent_sha256']==old.packet_digest(p['context3']['latent_history'][-1]),
        'R own current encoding, not predicted substitute')
    # No simulator, native state, metric, success, future image or env object in
    # this exact-field whitelist. I carries actual-encoding hashes, not vectors.
    return p


def branch_input(p):
    return dict(observed_context=deepcopy(p['context3']),observed_raw_history=deepcopy(p['actualrawhistory10']),
        prefix_commands=deepcopy(p['actualrawhistory10'][5:]),old_suffix_commands=deepcopy(p['diagnosticraw5']),
        goal=deepcopy(p['goal']),q=np.eye(192).tolist(),action_mean=p['modelbinding']['action_mean'],
        action_std=p['modelbinding']['action_std'],packet_logit_cap=float(.75*np.sqrt(5)),branch_id=old.packet_digest(p))


def verify_compile(request,response):
    p=validate_request(request);r=deepcopy(response);h=r.pop('release_sha256')
    require(h==old.packet_digest(r) and r['schema']==NAME+'.compiled' and r['request_sha256']==old.packet_digest(p)
        and r['model_hash_before']==r['model_hash_after']==suffix.MODEL_SHA and r['source_verified'] is True,
        'complete own rolling compiler release')
    b=branch_input(p);F.check_result(b,r['row'])
    for key,want in (('resources',F.budget()),('AD_API_resources',dict(grad=1,jacobian=1)),
        ('predict_API_resources',dict(predict=5))):F.blind.exact_counts(r['row'][key],want)
    require(r['dispatch_raw5']==r['row']['new_suffix_commands'] and
        r['predicted_successor']==r['row']['evaluations'][1]['context'],'unchanged new solve/replayed successor')
    return old.raw5(r['dispatch_raw5'])


class RollingSuffix:
    """Real frozen CUDA mathematics with NEW strict rolling-input validator.

    The old suffix.validate_packet cannot be used at step10/15: it hardcodes
    warmup final5 and step5. No monkeypatch, fake boundary or old loader here.
    """
    def verify_suffix_seal(self,request,response):return verify_compile(request,response)

    def compile_suffix(self,request,artifact_sink,*,model,device,check_limits):
        require(getattr(device,'type',None)=='cuda','explicit HOST CUDA only')
        require(suffix._LOCK.acquire(blocking=False),'nonreentrant AD instrumentation')
        row=dict(index=0);sink=suffix.Sink(artifact_sink,check_limits)
        try:
            import torch
            from torch.nn.attention import SDPBackend,sdpa_kernel
            require(torch.cuda.is_available() and not torch.is_inference_mode_enabled(),'CUDA/autograd scope')
            p=validate_request(request);b=branch_input(p);row['branch_id']=b['branch_id']
            def model_hash():
                check_limits();F.observed.enc.frozen(model)
                require(all(x.device==device for x in list(model.parameters())+list(model.buffers())),'all model tensors CUDA')
                return F.observed.enc.tensor_hash(model)
            before=model_hash();require(before==suffix.MODEL_SHA,'original frozen model')
            sink._write('request.json',raw(p));sync=lambda:torch.cuda.synchronize(device)
            meter=F.Meter(sink.event,sync);meter.require_native=True
            modules=F.Modules(model,sink,'branch_00',meter,sync)
            ad=F.SuffixAD(torch,F.blind.ARMS[1],sink.event,sync);ad.stage=lambda:meter.stage
            api=suffix.PredictAPI(model,meter,sink,sync);backend=F.d2p.TorchBackend(model,device)
            mm=torch.backends.cuda.matmul.allow_tf32;cn=torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
                    try:
                        with ad,modules,api:F.evaluate(b,backend,meter,sink,row)
                    finally:
                        row.update(resources={**modules.counts(),**meter.counts()},AD_API_resources=ad.counts(),
                            predict_API_resources=api.counts(),module_ledger=deepcopy(modules.entries),
                            math_ledger=deepcopy(meter.entries),AD_API_ledger=deepcopy(ad.events),predict_API_ledger=deepcopy(api.entries))
            finally:torch.backends.cuda.matmul.allow_tf32=mm;torch.backends.cudnn.allow_tf32=cn
            F.check_result(b,row);F.saved_arrays(b,row,sink.load)
            require(p['modelbinding']==suffix.modelbinding(),'final original source closure')
            response=dict(schema=NAME+'.compiled',request_sha256=old.packet_digest(p),row=row,
                model_hash_before=before,model_hash_after=model_hash(),source_verified=True,
                dispatch_raw5=row['new_suffix_commands'],predicted_successor=row['evaluations'][1]['context'],
                artifact_sha256={n:F.observed.enc.digest(v) for n,v in sink.files.items()})
            response['release_sha256']=old.packet_digest(response);verify_compile(p,response)
            sink._write('receipt.json.gz',gzip.compress(raw(response),mtime=0),final=True);return response
        except BaseException as exc:
            partial=dict(schema=NAME+'.compile_failure',row=row,error=repr(exc),commands_released=False)
            if not sink.broken:
                try:sink._write('failure.json.gz',gzip.compress(raw(partial),mtime=0),final=True)
                except BaseException as secondary:exc.add_note('failure persist: '+repr(secondary))
            raise
        finally:suffix._LOCK.release()


def episode_id(slot,arm,mode):return f'e{old.LAYERS.index((slot,arm)):02}_{mode}'


def encode_boundary(handler,runtime,task,layer,row,obs,body,meter,store,step):
    """Unmodified Handler visual code via explicit unique native-step5 namespace.

    Real step/PNG identity is separately persisted, never reported as step5.
    This copies the SAME current PNG bytes, not an earlier image or a new render.
    """
    virtual=row['episode_id']+f'_b{step:02}'
    alias=virtual+'_plan_05.png';h=store.write(alias,store.png_bytes(obs))
    require(h==obs['file_sha256'],'byte-identical current PNG adapter')
    mapping=dict(schema=NAME+'.visual_bridge',episode_id=row['episode_id'],actual_step=step,
        original_png={k:obs[k] for k in ('path','file_sha256','pixel_sha256')},
        legacy_episode_id=virtual,legacy_step=5,alias_png=str(store.path/alias))
    store.put(virtual+'_visual_bridge.json',mapping)
    request=dict(schema=old.NAME+'.encode_request',episode_id=virtual,observed_step=5,
        png=dict(path=str(store.path/alias),file_sha256=h,pixel_sha256=obs['pixel_sha256']),
        task_export=layer['task_export'],model_binding_sha256=source.digest(task['model_bindings']))
    request['request_sha256']=source.digest(request)
    sink=old.Sink(store,meter,virtual,'encode');meter.region='encoding'
    before=deepcopy(meter.counts);state=old.state_seal(body)
    response=handler.encode_observation(deepcopy(request),sink,**runtime);sink.complete()
    require(old.state_seal(body)==state,'encoder cannot change state/goal')
    old.check_delta(old.counts_delta(meter,before),dict(encoder=1,projector=1))
    require(response['request_sha256']==request['request_sha256'] and response['episode_id']==virtual
        and response['source_verified'] is response['model_unchanged'] is True and response['device_type']=='cuda'
        and response['latent'] in sink.artifacts,'own current actual encoding')
    z=old.native(store.load_array(response['latent']))
    receipt=dict(mapping=mapping,request=request,response=response,actual_step=step)
    store.put(virtual+'_encoding.json',receipt)
    index=store.index;store.emit('boundary_encoded',episode_id=row['episode_id'],actual_step=step,
        mapping_sha256=source.digest(mapping),encoding_sha256=source.digest(receipt))
    row['encodings'].append(dict(step=step,receipt_sha256=source.digest(receipt),latent=response['latent']))
    observation=dict(boundary=step,png_sha256=obs['file_sha256'],pixel_sha256=obs['pixel_sha256'],
        encoding_sha256=source.digest(receipt),observed_index=obs['observed_event_index'],encoded_index=index,
        observed_latent_sha256=old.packet_digest(z.tolist()))
    return z,observation


def setup(env,ref,source_plan,row,meter,store):
    """Same exact native reset/warmup as original driver, NEVER prefix restore."""
    body=env.unwrapped;slot=ref['slot'];meter.region='setup'
    obs,info=env.reset(seed=slot['reset_seed']);original_goal=source.vector(info['goal_state'])
    requested=source.sample_state(original_goal,list(body.block.shapes),dict(root_seed=slot['root_seed']),0,0,source_plan)
    obs,info=env.reset(seed=0,options=dict(state=requested.copy(),goal_state=original_goal.copy()))
    image,receipt=old.capture(env,obs,row,store,'warmup',0)
    history,states,snapshots,actual=[image],[source.vector(obs['state'])],[receipt['snapshot']],[]
    meter.region='warmup'
    for step,command in enumerate(old.kernel.raw_history(ref['raw_history']),1):
        obs,reward,terminated,truncated,info=env.step(command.copy());row['warmup_returned']+=1
        image,receipt=old.capture(env,obs,row,store,'warmup',step)
        value=np.asarray(body.latest_action).copy();actual.append(value)
        require(old.kernel.moving.exact(value,command),'literal warmup raw bits')
        store.emit('warmup_return',episode_id=row['episode_id'],step=step,command=value.tolist(),
            terminated=bool(terminated),truncated=bool(truncated))
        if terminated or truncated:
            row.update(status='native_stopped_warmup',terminated=bool(terminated),truncated=bool(truncated));return False
        if step in (5,10):history.append(image);states.append(source.vector(obs['state']));snapshots.append(receipt['snapshot'])
    require(old.kernel.moving.exact(np.asarray(actual),ref['raw_history']) and
        old.kernel.moving.exact(np.asarray(history),ref['history']) and
        old.kernel.moving.exact(np.asarray(states),ref['states']) and snapshots==ref['dynamics'],
        'exact same original moving root/history/dynamics')
    meter.region='goal_bind';state=old.kernel.moving.snapshot(body);pose=np.asarray(body.goal_pose).copy()
    meter.call('goal_bind',lambda:old.kernel.moving.attach_goal(body,ref['goal'],ref['goal_rgb']))
    require(old.kernel.moving.snapshot(body)==state and np.array_equal(pose,body.goal_pose),'same task goal, no pose/state intervention')
    _,obs0=old.capture(env,obs,row,store,'plan',0)
    require(obs0['pixel_sha256']==old.array_sha(ref['history'][-1]) and
        source.root_hash(source.vector(obs['state']),ref['goal'])==ref['root_sha256'],'same root and goal')
    row['initial_exact']=True;return True


def packet(env,commands,*,start,row,ref,meter,store,gate=None):
    meter.region='prefix' if start==0 else 'suffix';body=env.unwrapped
    for j,planned in enumerate(old.raw5(commands)):
        command=planned if gate is None else gate.take(gate.episode,6+j)
        require(np.array_equal(command,planned),'packet-local gate exact raw')
        step=start+j+1
        store.emit('authorized_dispatch',episode_id=row['episode_id'],step=step,
            packet_boundary=start,legacy_gate_ordinal=None if gate is None else 6+j,command=command.tolist())
        obs,reward,terminated,truncated,info=env.step(command.copy());row['control_returned']+=1
        row['executed_raw'].append(np.asarray(body.latest_action).tolist())
        _,receipt=old.capture(env,obs,row,store,'plan',step)
        if step in (15,20):row['checkpoints'].append(deepcopy(receipt))
        require(old.kernel.moving.exact(np.asarray(body.latest_action),command),'actual sealed raw bits')
        require(np.array_equal(info['goal_state'],ref['goal']) and np.array_equal(info['goal'],ref['goal_rgb']),
            'task goal unchanged throughout 20 steps')
        old.native_transition(body,ref['goal'],obs,reward,terminated,truncated,row)
        store.emit('control_return',episode_id=row['episode_id'],step=step,command=command.tolist(),
            terminated=bool(terminated),truncated=bool(truncated),reward=float(reward))
        if terminated or truncated:
            row['status']='native_stopped_control';return None
    return receipt


def compile_boundary(handler,runtime,task,layer,row,context,raw10,observation,body,meter,store,step):
    request=dict(schema=NAME+'.compile',episode_id=row['episode_id'],source_slot=layer['source_slot'],
        prefix_arm=layer['arm'],mode=row['mode'],boundary=step,context3=deepcopy(context),
        actualrawhistory10=np.asarray(raw10,np.float32).tolist(),diagnosticraw5=old.raw5(layer['commands'][5:]).tolist(),
        goal=deepcopy(task['gn_task']['goal']),modelbinding=deepcopy(task['modelbinding']),observation=observation)
    validate_request(request)
    ns=row['episode_id']+f'_b{step:02}';store.put(ns+'_compile_request.json',request)
    meter.region='compiling';sink=old.Sink(store,meter,ns,'compile')
    before=deepcopy(meter.counts);state=old.state_seal(body)
    response=handler.compile_suffix(deepcopy(request),sink,**runtime);sink.complete()
    require(old.state_seal(body)==state,'compiler cannot mutate live env')
    wanted=dict(predict=5,predictor=5,action_encoder=5,pred_proj=5,ad_forward=1,
        jacobian=1,svd=1,jacobian_api=1,grad_api=1,vjp_directions=192)
    old.check_delta(old.counts_delta(meter,before),wanted)
    commands=old.raw5(handler.verify_suffix_seal(deepcopy(request),deepcopy(response)))
    require(old.state_seal(body)==state,'pure verifier cannot mutate live env')
    old.check_delta(old.counts_delta(meter,before),wanted)
    for name,h in response['artifact_sha256'].items():
        require(name in sink.files and sink.files[name]['sha256']==h,'own durable compiler arrays')
    store.put(ns+'_compile_response.json',response)
    gate_id=ns+'_dispatch'
    seal=dict(schema=NAME+'.dispatch',episode_id=gate_id,actual_episode_id=row['episode_id'],boundary=step,
        request_sha256=old.packet_digest(request),response_sha256=old.packet_digest(response),
        remaining_public_steps=5,commands_sha256=old.array_sha(commands),commands=commands.tolist())
    seal['seal_sha256']=source.digest(seal);store.put(ns+'_dispatch_seal.json',seal)
    store.emit('boundary_sealed',episode_id=row['episode_id'],boundary=step,seal_sha256=seal['seal_sha256'])
    return commands,deepcopy(response['predicted_successor']),old.DispatchGate(gate_id,seal,commands),seal


def episode(env,ref,source_plan,layer,task,row,handler,runtime,meter,store,slots,paired):
    if not setup(env,ref,source_plan,row,meter,store):return
    prefix=old.raw5(layer['commands'][:5]);store.put(row['episode_id']+'_prefix_commit.json',dict(commands=prefix.tolist(),
        source_selection_sha256=layer['selection_sha256']))
    obs=packet(env,prefix,start=0,row=row,ref=ref,meter=meter,store=store)
    if obs is None:return
    key=(layer['source_slot'],layer['arm']);match={k:obs[k] for k in ('pixel_sha256','state','snapshot')}
    if row['mode']=='I':paired[key]=match
    else:require(paired.get(key)==match,'same-layer I/R exact fresh prefix')
    observed=deepcopy(task['gn_task']['root_context']);imagined=None
    for boundary in BOUNDARIES:
        # Only completed actual packets enter the chronological raw history.
        raw10=np.concatenate((ref['raw_history'],np.asarray(row['executed_raw'],np.float32)))[-10:]
        actual,observation=encode_boundary(handler,runtime,task,layer,row,obs,env.unwrapped,meter,store,boundary)
        observed=roll(observed,actual,raw10,task['gn_task'])
        if imagined is None:imagined=roll(task['gn_task']['root_context'],layer['imagined_prefix'],raw10,task['gn_task'])
        else:require(imagined['action_history']==normalized(task['gn_task'],raw10),'imagined history uses own actually executed packets')
        context=observed if row['mode']=='R' else imagined
        slot=next(s for s in slots if s['episode_id']==row['episode_id'] and s['boundary']==boundary)
        slot['status']='attempted';store.emit('compile_slot_attempted',slot=deepcopy(slot))
        commands,imagined,gate,seal=compile_boundary(handler,runtime,task,layer,row,context,raw10,observation,
            env.unwrapped,meter,store,boundary)
        slot.update(status='sealed',seal_sha256=seal['seal_sha256']);store.emit('compile_slot_sealed',slot=deepcopy(slot))
        obs=packet(env,commands,start=boundary,row=row,ref=ref,meter=meter,store=store,gate=gate)
        if obs is None:return
        row['completed_packets']+=1
    # Terminal20 is evaluation only; NEVER a fourth planning boundary.
    terminal,_=encode_boundary(handler,runtime,task,layer,row,obs,env.unwrapped,meter,store,20)
    goal=np.asarray(task['gn_task']['goal'],np.float32)
    row.update(full_horizon=True,status='completed20',actual_Q20=float(np.mean((terminal.astype(float)-goal.astype(float))**2)))


def roster():
    rows=[dict(episode_id=episode_id(s,a,m),source_slot=s,prefix_arm=a,mode=m,status='not_started',
        first_error=None,secondary_errors=[],warmup_returned=0,control_returned=0,completed_packets=0,
        executed_raw=[],encodings=[],initial_exact=False,full_horizon=False,native_success=None,
        terminated=None,truncated=None,PNG=0,checkpoints=[]) for s,a in old.LAYERS for m in MODES]
    slots=[dict(episode_id=r['episode_id'],boundary=b,status='not_reached',seal_sha256=None) for r in rows for b in BOUNDARIES]
    return rows,slots


def execute(p,data,store,handler,factory,*,authorization,runtime,final_model_check):
    require(authorization==AUTH and p['contract']==contract(),'MAIN authorization/new complete contract')
    require([(x['source_slot'],x['arm']) for x in p['layers']]==list(old.LAYERS),'all four fixed layers')
    require(set(runtime)=={'model','device','check_limits'} and getattr(runtime['device'],'type',None)=='cuda',
        'explicit shared CUDA runtime')
    meter=Meter(store,runtime['check_limits']);rows,slots=roster();errors=[];paired={}
    result=dict(schema=NAME+'.report',status='running',rows=rows,compile_slots=slots,errors=errors,
        contract=contract(),fixed_denominator=8,independent_tasks=2,full_resource_equivalence=False,
        P1_admission=False,P4_admission=False,independent_confirmation=False)
    original_check=runtime['check_limits']
    def checked():original_check();meter.check()
    runtime=dict(runtime,check_limits=checked)
    try:
        store.put('roster.json',dict(rows=rows,compile_slots=slots));store.emit('run_started',contract=contract())
        for row in rows:
            meter.episode=row['episode_id'];meter.region='setup';env=None
            layer=next(x for x in p['layers'] if x['source_slot']==row['source_slot'] and x['arm']==row['prefix_arm'])
            try:
                env=meter.call('constructor',lambda:factory(meter));body=env.unwrapped
                require(body._live_meter is meter and body._live_constructor_complete is True,'meter before native constructor')
                require(body.with_target is False and body.render_action is False and body.render_size==224
                    and body.relative is True and body.action_scale==100 and body.control_hz==10 and body.dt==.01,
                    'unchanged native renderer/control')
                row['status']='running'
                episode(env,data['references'][row['source_slot']],data['plan'],layer,data['tasks'][row['source_slot']],
                    row,handler,runtime,meter,store,slots,paired)
            except BaseException as exc:
                old.kernel.retain(row,exc,'episode')
                for slot in slots:
                    if slot['episode_id']==row['episode_id'] and slot['status']=='attempted':slot['status']='failed_pending'
            finally:
                meter.region='closing'
                if env is not None:
                    try:env.close()
                    except BaseException as exc:old.kernel.retain(row,exc,'close')
                store.put(row['episode_id']+'_episode.json',row);store.emit('episode_closed',receipt=row)
    except BaseException as exc:errors.append(dict(stage='execution_or_durability',message=repr(exc)))
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final source/input SHA: '+f)
        result['model_final_check']=final_model_check()
        require(result['model_final_check']['model_unchanged'] is True,'whole-job frozen model')
    except BaseException as exc:errors.append(dict(stage='final_seals',message=repr(exc)))
    errors.extend(meter.secondary)
    try:
        require(not meter.pending and all(v['attempted']==v['completed'] for v in meter.counts.values()),'pending work retained')
        counts=meter.counts
        require(counts['public_step']['completed']==sum(r['warmup_returned']+r['control_returned'] for r in rows)
            and counts['physics_step']['completed']==2*counts['reset']['completed']+10*counts['public_step']['completed']
            and counts['set_state']['completed']==2*counts['reset']['completed']
            and counts['goal_setter']['completed']==counts['reset']['completed']+counts['goal_bind']['completed']
            and counts['explicit_render']['completed']==sum(r['PNG'] for r in rows),'full native operation census')
        if all(r['full_horizon'] for r in rows):
            require(all(counts[k]==dict(attempted=v,completed=v) for k,v in contract()['neural_caps'].items()),'complete8 neural census')
    except BaseException as exc:errors.append(dict(stage='final_census',message=repr(exc)))
    clean=not errors and all(r['first_error'] is None and r['status'] in
        ('completed20','native_stopped_warmup','native_stopped_control') for r in rows)
    result.update(status='known_moving_multiboundary_recorded' if clean else 'partial_or_failed',
        counts=meter.counts,pending=deepcopy(meter.pending),completed20=sum(r['full_horizon'] for r in rows),
        execution_wall_seconds=time.monotonic()-meter.started,
        wall_seconds=time.monotonic()-data.get('job_started',meter.started))
    if result['wall_seconds']>=600:result['status']='partial_or_failed';errors.append(dict(stage='wall',message='600s whole job'))
    try:
        store.emit('run_finished',status=result['status'],counts=meter.counts,compile_slots=slots)
        result.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
        store.put('report.json',result,final=True)
    except BaseException as exc:result['status']='partial_or_failed';errors.append(dict(stage='final_write',message=repr(exc)))
    return result


def build():
    require(sha(old.__file__)==OLD_SOURCE_SHA and sha(suffix.__file__)==old.BACKEND_SHA,'original physical/math sources')
    original=old.verify(OLD_PREPARE,OLD_SHA)
    bindings=dict(original['bindings']);bindings[str(OLD_PREPARE)]=OLD_SHA
    for path in (Path(__file__).resolve(),ROOT/'tests'/(NAME+'_tests.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(path)]=sha(path)
    return dict(schema=NAME+'.prepare',original_prepared=str(OLD_PREPARE),original_prepare_sha256=OLD_SHA,
        layers=deepcopy(original['layers']),modelbinding=deepcopy(original['modelbinding']),
        original_manifest=original['manifest'],contract=contract(),bindings=bindings,
        slots=roster()[1],P1_admission=False,P4_admission=False,scientific_pass=False)


def prepare(output):
    p=build();path=Path(output).absolute()
    require(path.name.startswith(NAME) and not path.exists(),'new exclusive prepared file')
    require(len(raw(p))<512*1024,'bounded prepare')
    # Original Store fsync/exclusive write and disk-floor checks, no environment.
    require(old.shutil.disk_usage(path.parent).free>=old.FLOOR+len(raw(p)),'prepare200MiB floor')
    with path.open('xb') as f:require(f.write(raw(p))==len(raw(p)),'short prepare');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),contract=p['contract'])


def verify(path,expected):
    require(sha(path)==expected,'external prepare byte SHA');p=read(path)
    require(p==build(),'complete new prepare reconstruction');return p


def run(prepared,expected,output,*,authorization):
    started=time.monotonic();require(authorization==AUTH,'MAIN explicit known-task CUDA+physics authorization')
    p=verify(prepared,expected);output=Path(output).absolute()
    require(output.parent==ROOT/'results' and output.name.startswith(NAME),'exclusive new runtime directory')
    manifest=read(p['original_manifest'])
    data=old.kernel.load_source(manifest['sourcepack'],old.SOURCE_SHA,manifest['source_prepared'],manifest['source_prepare_sha256'])
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    data['tasks']={}
    for layer in p['layers']:
        interface=encoder.load_task(layer['task_export'],layer['task_sha256'])
        data['tasks'][layer['source_slot']]=dict(gn_task=interface['gn_task'],model_bindings=encoder.MODELS,modelbinding=p['modelbinding'])
    store=Store(output);store.put('execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    rows,slots=roster();store.put('startup_roster.json',dict(rows=rows,compile_slots=slots))
    try:
        import torch
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def check():
            require(time.monotonic()-started<600 and torch.cuda.memory_allocated(device)<=3*1024**3
                and torch.cuda.memory_reserved(device)<=4*1024**3,'600s/3GiB/4GiB whole-job CUDA caps');store.check()
        check();model=load_compatible_pretrained(encoder.POLICY,device=device).eval().requires_grad_(False);check()
        encoder.low.frozen(model);before=encoder.low.tensor_hash(model)
        require(before==p['modelbinding']['tensor_sha256'],'actual original frozen model hash')
        def final():
            check();encoder.low.frozen(model);after=encoder.low.tensor_hash(model)
            return dict(before=before,after=after,model_unchanged=before==after)
        handler=old.Handler(RollingSuffix())
        return execute(p,dict(data,job_started=started),store,handler,old.native_factory,authorization=AUTH,
            runtime=dict(model=model,device=device,check_limits=check),final_model_check=final)
    except BaseException as exc:
        failure=dict(schema=NAME+'.startup_or_uncaught_failure',error=repr(exc),rows=rows,compile_slots=slots,
            fixed_denominator=8,independent_tasks=2,P1_admission=False,P4_admission=False)
        try:store.put('startup_failure.json',failure,final=True)
        except BaseException as secondary:exc.add_note('failure persistence: '+repr(secondary))
        raise


def reconcile_events(events,report):
    """Independent durable operation/return census, not a hash-only assertion."""
    counts={k:dict(attempted=0,completed=0) for k in contract()['physical_caps']|contract()['neural_caps']}
    pending={};previous=None;last_id=-1
    for index,event in enumerate(events):
        require(event['index']==index and event['previous_sha256']==previous and
            event['event_sha256']==source.digest({k:v for k,v in event.items() if k!='event_sha256'}),'complete journal chain')
        previous=event['event_sha256'];kind=event['kind']
        if kind not in ('operation_attempted','operation_completed'):continue
        entry={k:event[k] for k in ('operation_id','episode_id','resource','units','region')}
        ident=entry['operation_id'];resource=entry['resource'];units=entry['units']
        require(type(ident) is int and type(units) is int and units>0 and resource in counts,'literal operation census')
        if kind=='operation_attempted':
            require(ident>last_id and ident not in pending,'single no-retry operation attempt')
            last_id=ident;pending[ident]=entry;counts[resource]['attempted']+=units
        else:
            require(pending.pop(ident,None)==entry,'actual return matches durable attempt')
            counts[resource]['completed']+=units
    require(events and events[-1]['kind']=='run_finished' and previous==report['journal_chain_tail'],
        'terminal journal, never infer completion from partial files')
    require(counts==report['counts'] and {str(k):v for k,v in pending.items()}==
        {str(k):v for k,v in report['pending'].items()},'independent ledger/actual report reconciliation')
    require(events[-1]['compile_slots']==report['compile_slots'],'terminal24-slot binding')
    return counts


def consume(output,report_sha,prepared,prepare_sha):
    """Read-only saved arithmetic and original full-array validators. No replay."""
    output=Path(output).absolute();p=verify(prepared,prepare_sha)
    require(sha(output/'report.json')==report_sha,'external terminal report byte SHA')
    report=read(output/'report.json');expected_rows,expected_slots=roster()
    require(report['schema']==NAME+'.report' and report['contract']==contract()
        and report['fixed_denominator']==8 and report['independent_tasks']==2
        and report['P1_admission'] is report['P4_admission'] is False,'fixed known-task report only')
    require(len(report['rows'])==8 and [r['episode_id'] for r in report['rows']]==[r['episode_id'] for r in expected_rows]
        and len(report['compile_slots'])==24 and
        [(s['episode_id'],s['boundary']) for s in report['compile_slots']]==
        [(s['episode_id'],s['boundary']) for s in expected_slots],'full8/24 denominator, no failed-slot dropping')
    require(read(output/'execution_binding.json')==dict(prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha),
        'actual prepare execution binding')
    jp=output/'journal.jsonl';require(jp.stat().st_size<=old.CAP and sha(jp)==report['journal_sha256'],'durable full journal bytes')
    data=jp.read_bytes();require(data.endswith(b'\n'),'no partial terminal event')
    events=[json.loads(line) for line in data.splitlines()];counts=reconcile_events(events,report)
    for row in report['rows']:
        require(read(output/(row['episode_id']+'_episode.json'))==row,'durable full episode')
    # Failed/pending work is preserved, NEVER promoted by a good subset.
    if report['status']!='known_moving_multiboundary_recorded' or report['errors'] or report['pending']:
        return dict(schema=NAME+'.consume',status='saved_partial_preserved_NOT_complete',fixed_denominator=8,
            compile_denominator=24,rows=report['rows'],compile_slots=report['compile_slots'],counts=counts,
            P1_admission=False,P4_admission=False,physical_execution_authorized=False)
    audited=[]
    for slot in report['compile_slots']:
        if slot['status']!='sealed':continue
        eid=slot['episode_id'];step=slot['boundary'];ns=eid+f'_b{step:02}'
        request=read(output/(ns+'_compile_request.json'));response=read(output/(ns+'_compile_response.json'))
        commands=verify_compile(request,response);seal=read(output/(ns+'_dispatch_seal.json'))
        require(seal['actual_episode_id']==eid and seal['boundary']==step and seal['seal_sha256']==slot['seal_sha256']
            and seal['seal_sha256']==source.digest({k:v for k,v in seal.items() if k!='seal_sha256'})
            and seal['request_sha256']==old.packet_digest(request) and seal['response_sha256']==old.packet_digest(response)
            and seal['commands_sha256']==old.array_sha(commands) and seal['commands']==commands.tolist(),'own durable selection seal')
        folder=output/(ns+'_compile')
        def local(name):
            require(Path(name).name==name and not (folder/name).is_symlink(),'flat original arrays')
            return folder/name
        for name,h in response['artifact_sha256'].items():require(sha(local(name))==h,'original compiler artifact SHA')
        def load(entry):
            path=local(entry['file']);require(path.stat().st_size<=1024**2,'bounded saved array')
            payload=path.read_bytes();require(hashlib.sha256(payload).hexdigest()==entry['sha256'],'full array bytes')
            stream=io.BytesIO(payload);a=np.load(stream,allow_pickle=False)
            require(stream.tell()==len(payload) and str(a.dtype)==entry['dtype'] and list(a.shape)==entry['shape'],'exact original array')
            return a
        F.saved_arrays(branch_input(request),response['row'],load)
        same=[e for e in events if e.get('episode_id')==eid]
        encoded=next(e['index'] for e in same if e['kind']=='boundary_encoded' and e['actual_step']==step)
        sealed=next(e['index'] for e in same if e['kind']=='boundary_sealed' and e['boundary']==step)
        dispatch=[e for e in same if e['kind']=='authorized_dispatch' and e['packet_boundary']==step]
        require(encoded<sealed and all(e['index']>sealed for e in dispatch)
            and [e['step'] for e in dispatch]==list(range(step+1,step+1+len(dispatch)))
            and [e['command'] for e in dispatch]==commands[:len(dispatch)].tolist(),'observe/encode/seal before real dispatch')
        audited.append(dict(episode_id=eid,boundary=step,arrays_verified=True,dispatched=len(dispatch)))
    return dict(schema=NAME+'.consume',status='saved_multiboundary_verified_NOT_scientific_pass',
        fixed_denominator=8,compile_denominator=24,independent_tasks=2,counts=counts,compiled=audited,
        completed20=report['completed20'],native_metrics=[r.get('final_metrics') for r in report['rows']],
        source_sha256=sha(Path(__file__)),report_sha256=report_sha,
        independent_neural_replay=False,equal_total_compute=False,P1_admission=False,P4_admission=False,
        physical_execution_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha'):p.add_argument('--'+key)
    p.add_argument('--authorize-known-cuda-and-physics',action='store_true');a=p.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':result=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command=='consume':result=consume(a.output,a.report_sha,a.prepared,a.prepare_sha)
    else:
        require(a.authorize_known_cuda_and_physics,'MAIN explicit authorization')
        result=run(a.prepared,a.prepare_sha,a.output,authorization=AUTH)
    print(source.canonical({k:v for k,v in result.items() if k not in ('rows','compile_slots')}))
    if a.command=='run' and result['status']!='known_moving_multiboundary_recorded':raise SystemExit(2)


if __name__=='__main__':main()
