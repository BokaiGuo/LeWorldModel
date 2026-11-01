"""Real RGB-feedback gradient MPC20, MAIN HOST CUDA/native physics only.

Frozen original model/Q/normalizer; scalar action gradients, not model training.
Known2 roots/four original protocol slots, no scientific admission.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v1 as p3
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v2 as transport
from experiments.spectral_reachability import p1_constructive_gradient_mpc_cuda_20261027_v1 as gradient

NAME='agent_p3_feedback_gradient_mpc20_20261028_v1'
AUTH='MAIN_KNOWN2_FEEDBACK_GRADIENT_MPC20_CUDA_AND_PHYSICS'
ROOT=p3.ROOT
moving,old,core,signal=p3.moving,p3.old,p3.core,p3.signal
require,sha,digest,raw=p3.require,p3.sha,p3.digest,p3.raw
BOUNDARIES=p3.BOUNDARIES


def contract():
    return dict(episodes=4,known_roots=2,slots=16,boundaries=list(BOUNDARIES),terminal=20,packet_steps=5,
        episode_slots='original s0/s1 fixed/adaptive labels are reference slots ONLY; identical gradient algorithm in all4',
        execute='optimize full10 at EACH own real observed boundary, execute first5 ONLY; discard remainder',
        prediction_horizon=10,final_node_horizon='still10 at boundary15; speculative endpoint25 is not actual_Q20',
        feedback='cached own observed latent3 and actual raw10; never imagined state promoted to observation',
        optimizer=dict(updates=32,step_size=1.,initial_logits=np.zeros((2,10)).tolist(),
            recurrence='original projected_logits(x-gradient,cap), all32 proposals evaluated',
            selection='stable minimum(original float64 terminal meanQ, iteration), then independent triple fullH2 replay',
            packet_logit_cap=float(.75*np.sqrt(5)),chart='NumPy tanh(float32 logits) -> literal float32 raw10',
            derivative='scalar meanQ w.r.t. fresh rawfloat32 leaf[10,2]; analytic 1-raw^2 chain ONCE',
            warm_start=False,line_search=False,model_training=False),
        node_optimizer_budget=gradient.budget(),node_native_caps=node_caps(),
        max_predict_API=16*136,max_scalar_reverse_sweeps=16*32,
        physical_caps=deepcopy(p3.contract()['physical_caps']),visual_caps=deepcopy(p3.contract()['visual_caps']),
        wall_seconds=1800,output_bytes=512*1024**2,final_reserve_bytes=64*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,with_target=False,render_action=False,
        TF32=False,autocast=False,SDP='MATH',CUDA_only=True,no_retry=True,no_padding=True,no_outcome_input=True,
        equal_requested_execution_horizon=True,equal_actual_execution_horizon=None,equal_FLOPs=False,equal_wall=False,
        total_FLOPs=None,backward_engine_calls=None,scalar_grad_and_VJP_are_same_reverse_sweeps=True,
        P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False)


def node_caps():
    c=dict.fromkeys(core.AXES,0)
    c.update(predict_API=136,predict_samples=136,predict_context_tokens=408,
        action_encoder_calls=136,action_encoder_samples=136,action_encoder_context_tokens=408,
        predictor_calls=136,predictor_samples=136,predictor_context_tokens=408,pred_proj_calls=136,pred_proj_rows=408,
        grad_API=32,VJP_directions=32,raw_backward_callbacks=32)
    return c


def roster():return p3.roster()
def node_roster():return p3.node_roster()
def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def local_task(base,context,identity):
    t=core.validate_task(base);core.context(context);t['root_context']=deepcopy(context)
    t['binding'].update(context_sha256=digest(context),task_id=digest([base['binding']['task_id'],NAME,identity]),
        task_sha256=digest([base,context,identity]),task_contract_sha256=digest([NAME,identity]))
    return core.validate_task(t)


def validate_request(r,base=None):
    core.keys(r,('schema','episode_id','source_slot','mode','boundary','base_task_sha256','task','raw10','observation'))
    require(r['schema']==NAME+'.node' and type(r['source_slot']) is int and r['source_slot'] in (0,1)
        and r['mode'] in p3.MODES and r['episode_id']==f"s{r['source_slot']}_{r['mode']}"
        and type(r['boundary']) is int and r['boundary'] in BOUNDARIES,'own fixed actual node identity')
    t=core.validate_task(r['task']);a=core.finite(r['raw10'],(10,2),True)
    require(len(t['goal'])==192 and np.array_equal(t['projection'],np.eye(192)) and (abs(a)<=1).all()
        and t['root_context']['action_history']==moving.normalized(t,a),'original identityQ and actual raw10 once-normalized')
    core.hash_value(r['base_task_sha256']);ob=r['observation']
    core.keys(ob,('boundary','png_sha256','pixel_sha256','encoding_sha256','observed_index','encoded_index','observed_latent_sha256'))
    require(ob['boundary']==r['boundary'] and type(ob['observed_index']) is int and type(ob['encoded_index']) is int
        and 0<=ob['observed_index']<ob['encoded_index'] and
        ob['observed_latent_sha256']==old.packet_digest(t['root_context']['latent_history'][-1]),'current own RGB observed then encoded')
    for k in ('png_sha256','pixel_sha256','encoding_sha256','observed_latent_sha256'):core.hash_value(ob[k])
    if base is not None:
        identity={k:deepcopy(r[k]) for k in ('episode_id','source_slot','mode','boundary','raw10','observation')}
        require(r['base_task_sha256']==digest(base) and t==local_task(base,t['root_context'],identity),'same original goal/model/Q/normalizer')
    return t


def gradient_task(request,template):
    """Original H2 optimizer ABI inside one newly observed node, not old receipt."""
    t=validate_request(request);gradient.core.validate_task(template)
    require(all(template[k]==t[k] for k in ('goal','projection','action_mean','action_std')),'original H2 template goal/Q/normalizer')
    out=deepcopy(template);out.update(root_context=deepcopy(t['root_context']),root_raw_action_history=deepcopy(request['raw10']),
        task_contract_sha256=digest([NAME,request]))
    gradient.core.validate_task(out);return out


class NativeGradient(gradient.NativeGradient):
    """Instance-local full native I/O observer, explicit scalar grad; NO torch/global patches."""
    kind='HOST_CUDA'
    def __init__(self,model,device,task,public_task,journal,ledger):
        super().__init__(model,device,task)
        self.observer=p3.native.NativeProgram(model,device,public_task,journal)
        self.observer.ledger=ledger;self.serial=0;self.mode=None
    @contextmanager
    def guard(self,meter):
        with super().guard(meter),self.observer.observe():yield
    def _start(self,task):
        z,h=super()._start(task);o=self.observer
        o.phase=f'forward_{self.serial:03}';o.step=0;self.serial+=1
        o.ledger.scope=dict(forward=o.phase,group=self._active_meter.group,mode=self.mode)
        o.store.event(dict(kind='forward_started',phase=o.phase,group=self._active_meter.group,mode=self.mode))
        return z,h
    def _step(self,task,z,h,raw_action):
        torch=self.torch;o=self.observer
        mean,std=(self._tensor(task[k]) for k in ('action_mean','action_std'))
        token=((raw_action-mean)/std).reshape(10);outgoing=torch.cat((h,token[None,None]),1)
        o.arrays('program_packet',latent=z,history=h,raw=raw_action,mean=mean,std=std,token=token,outgoing=outgoing)
        encoded=self.model.action_encoder(outgoing)
        predicted=o.call('predict',lambda:self.model.predict(z,encoded),dict(latent=z,act_emb=encoded),shape=list(z.shape))
        require(tuple(predicted.shape)==(1,3,192) and bool(torch.isfinite(predicted).all()),'original full native prediction3 ABI')
        z,h=torch.cat((z[:,1:],predicted[:,-1:]),1),outgoing[:,1:]
        o.arrays('program_boundary',latent=z,history=h);o.step+=1
        return z,h
    def rollout(self,task,actions,meter,mode):
        self.mode=mode;path=super().rollout(task,actions,meter,mode)
        self.observer.store.event(dict(kind='forward_finished',phase=self.observer.phase,commands=actions.tolist(),contexts=path))
        return path
    def gradient(self,task,incumbent,meter):
        torch=self.torch;o=self.observer;self.mode='gradient'
        with torch.enable_grad():
            a=self._tensor(incumbent['commands']).requires_grad_(True);z,h=self._start(task);paths=[]
            for k in range(2):
                z,h=self._step(task,z,h,a[5*k:5*k+5]);paths.append(self._saved(z,h))
                meter.emit('gradient_forward_packet',packet=k,context=paths[-1])
            o.store.event(dict(kind='forward_finished',phase=o.phase,commands=incumbent['commands'],contexts=paths))
            loss=((z[0,-1]-self._tensor(task['goal']))**2).mean()
            before=dict(task_id=gradient.core.task_id(task),at_logits=deepcopy(incumbent['logits']),contexts=paths,
                at_terminal=z[0,-1].detach().cpu().tolist(),native_float32_loss=float(loss.detach().cpu()),
                saved_float64_cost=gradient.core.model_cost(task,paths[-1]['latent_history'][-1]))
            meter.emit('gradient_forward_complete',forward=before)
            require(paths==incumbent['contexts'],'fresh fullH2 scalar-gradient forward equals own current iterate')
            def callback(value):return o.ledger.call('raw_backward',lambda:o._raw_return(value))
            hook=a.register_hook(callback)
            def backward():
                require(loss.ndim==0 and a.is_leaf and a.requires_grad and a.grad is None,'fresh raw leaf / scalar loss only')
                e=dict(index=int(meter.group.split('_')[-1])-1,resource='grad',evaluation_id=meter.group,
                    completed=False,scalar_outputs=1,input_shape=[10,2],retain_graph=False,create_graph=False)
                meter.emit('AD_API_attempt',entry=e)
                value=o.ledger.call('vjp',lambda:o.call('grad',
                    lambda:torch.autograd.grad(loss,a,retain_graph=False,create_graph=False)[0],dict(loss=loss,raw=a)),directions=1)
                meter.sync();meter.emit('AD_API_completed',entry=dict(e,completed=True));return value
            try:g=meter.call('backward',backward).detach().cpu().numpy().astype(float)
            finally:hook.remove()
        chain=1-np.asarray(incumbent['commands'],float)**2
        d=dict(**before,raw_control_gradient=g.tolist(),objective_gradient=(g*chain).reshape(2,10).tolist(),
            tanh_chain=chain.tolist(),scalar_reverse_sweeps=1)
        meter.emit('gradient_complete',gradient=d);return d


def plan_node(request,template,model,device,journal,check,*,runner=None,permit_ram=False):
    public=validate_request(request);task=gradient_task(request,template)
    ledger=core.Ledger(node_caps(),sink=journal.event)
    meter=gradient.Meter(lambda e:journal.event(dict(kind='optimizer',event=e)),
        (lambda:None) if permit_ram else lambda:__import__('torch').cuda.synchronize(device))
    result=dict(schema=NAME+'.plan',status='attempted',request_sha256=digest(request),gradient_task_sha256=digest(task),
        result=None,commands=None,optimizer_counts=None,resource_receipt=None)
    try:
        check();journal.event(dict(kind='node_started',request_sha256=digest(request),gradient_task_sha256=digest(task)))
        if runner is None:
            require(not permit_ram and getattr(device,'type',None)=='cuda','actual HOST CUDA compiler')
            backend=NativeGradient(model,device,task,public,journal,ledger)
            with backend.guard(meter):optimized=gradient.optimize(task,backend,meter)
        else:
            require(permit_ram is True,'RAM runner is never accepted by production CLI')
            optimized=runner(task,meter,ledger,journal)
        selected=gradient.choice(task,optimized)
        require(ledger.attempted==ledger.completed==node_caps(),'exact actual scalar/forward/module census')
        result.update(status='sealed',result=optimized,commands=deepcopy(selected['commands'][:5]),
            optimizer_counts=meter.counts(),resource_receipt=ledger.receipt())
        journal.event(dict(kind='node_frozen',plan_sha256=digest(result),commands=result['commands']))
        return result
    finally:
        if result['status']!='sealed':
            result.update(optimizer_counts=meter.counts(),resource_receipt=ledger.receipt())
            journal.event(dict(kind='node_failed',partial=result))


class Store(p3.Store):
    def __init__(self,path):
        self.path=Path(path).absolute();self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve()==self.path
            and not self.path.exists() and shutil.disk_usage(self.path.parent).free>=contract()['disk_floor_bytes']+contract()['output_bytes'],
            'new own512MiB output plus200MiB reserve')
        self.path.mkdir();signal.fsync_dir(self.path.parent);self.index=0;self.previous=None


Meter=p3.Meter  # Physical/visual caps EXACT unchanged; not a scheduler alias.


def episode(env,ref,data,layer,base,template,row,nodes,visual,runtime,meter,store,*,runner=None,permit_ram=False):
    if not moving.setup(env,ref,data['plan'],row,meter,store):return
    context=deepcopy(base['root_context']);obs=row['last_observation']
    for boundary in BOUNDARIES:
        node=next(n for n in nodes if n['episode_id']==row['episode_id'] and n['boundary']==boundary);node['status']='attempted'
        meter.region='encoding'
        z,ob=moving.encode_boundary(visual,runtime,data['visual_tasks'][row['source_slot']],layer,row,obs,env.unwrapped,meter,store,boundary)
        raw10=np.concatenate((ref['raw_history'],np.asarray(row['executed_raw'],np.float32).reshape(-1,2)))[-10:]
        if boundary==0:context['latent_history'][-1]=z.tolist();context['action_history']=moving.normalized(base,raw10)
        else:context=moving.roll(context,z,raw10,base)
        identity=dict(episode_id=row['episode_id'],source_slot=row['source_slot'],mode=row['mode'],boundary=boundary,raw10=raw10.tolist(),observation=ob)
        request=dict(schema=NAME+'.node',base_task_sha256=digest(base),task=local_task(base,context,identity),**identity)
        validate_request(request,base);node['request']=request;meter.region='compiling';ns=row['episode_id']+f'_b{boundary:02}'
        store.put(ns+'_request.json',request);state=old.state_seal(env.unwrapped)
        journal=p3.NativeJournal(store,ns,runtime['check_limits']);planned=None
        try:
            planned=plan_node(request,template,runtime['model'],runtime['device'],journal,runtime['check_limits'],runner=runner,permit_ram=permit_ram)
            require(old.state_seal(env.unwrapped)==state,'model action optimization cannot mutate native state/goal')
        finally:
            journal.close();node['native_journal_sha256']=sha(journal.path);node['native_journal_tail']=journal.tail
        node.update(receipt=planned,status='sealed',resource_receipt=planned['resource_receipt'])
        seal=dict(schema=NAME+'.dispatch',episode_id=row['episode_id'],boundary=boundary,request_sha256=digest(request),
            plan_sha256=digest(planned),commands=planned['commands'],commands_sha256=digest(planned['commands']),native_journal_sha256=node['native_journal_sha256'])
        seal['seal_sha256']=digest(seal);node['seal']=seal;store.put(ns+'_seal.json',seal)
        store.emit('boundary_sealed',episode_id=row['episode_id'],boundary=boundary,seal_sha256=seal['seal_sha256'])
        obs=moving.packet(env,planned['commands'],start=boundary,row=row,ref=ref,meter=meter,store=store)
        if obs is None:
            if row['control_returned']==boundary+5:row['completed_packets']+=1
            if row['control_returned']==20:obs=row['last_observation'];break
            return
        row['completed_packets']+=1
    terminal,_=moving.encode_boundary(visual,runtime,data['visual_tasks'][row['source_slot']],layer,row,obs,env.unwrapped,meter,store,20)
    row.update(full_horizon=True,status='completed20',actual_Q20=core.model_cost(base,dict(context,latent_history=context['latent_history'][1:]+[terminal.tolist()])))


def execute(p,data,store,visual,runtime,factory,*,authorization,final_check,runner=None,permit_ram=False):
    require(authorization==AUTH and p['contract']==contract() and (runner is None or permit_ram is True),'own MAIN explicit feedback gradient authorization')
    require(set(runtime)=={'model','device','check_limits'} and getattr(runtime['device'],'type',None)=='cuda','explicit single HOST CUDA runtime')
    rows,nodes=roster(),node_roster();meter=Meter(store,runtime['check_limits']);errors=[]
    report=dict(schema=NAME+'.report',status='running',contract=contract(),rows=rows,nodes=nodes,errors=errors,
        counts=meter.counts,pending=meter.pending,fixed_denominator=4,node_denominator=16,known_roots=2,
        P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False)
    data['live_report']=report;store.put('roster.json',dict(rows=rows,nodes=nodes));store.emit('run_started',contract=contract())
    for row in rows:
        env=None;meter.episode=row['episode_id'];meter.region='setup'
        try:
            env=meter.call('constructor',lambda:factory(meter));body=env.unwrapped
            require(body._live_meter is meter and body._live_constructor_complete is True and body.with_target is False and body.render_action is False
                and body.render_size==224 and body.relative is True and body.action_scale==100 and body.control_hz==10 and body.dt==.01,'original metered native environment')
            row['status']='running';s=row['source_slot']
            episode(env,data['references'][s],data,p['layers'][s],p['tasks'][s],p['templates'][s],row,nodes,visual,runtime,meter,store,runner=runner,permit_ram=permit_ram)
        except BaseException as exc:
            old.kernel.retain(row,exc,'episode')
            for node in nodes:
                if node['episode_id']==row['episode_id'] and node['status']=='attempted':node.update(status='failed',error=repr(exc))
        finally:
            meter.region='closing';prior=store.finalizing;store.finalizing=True;entered=False
            def close():
                nonlocal entered
                entered=True;return env.close()
            try:
                if env is not None:meter.call('close',close)
            except BaseException as exc:
                old.kernel.retain(row,exc,'close')
                if env is not None and not entered:
                    try:env.close()
                    except BaseException as secondary:old.kernel.retain(row,secondary,'emergency_close')
            finally:store.finalizing=prior
            store.put(row['episode_id']+'_episode.json',row);store.emit('episode_closed',receipt=row)
    try:
        report['final_check']=final_check();require(report['final_check']['model_unchanged'] is True,'whole model unchanged')
        require(not meter.pending and not meter.secondary,'all native operation returns')
        c=meter.counts
        require(c['public_step']['completed']==sum(r['warmup_returned']+r['control_returned'] for r in rows)
            and c['physics_step']['completed']==2*c['reset']['completed']+10*c['public_step']['completed']
            and c['set_state']['completed']==2*c['reset']['completed'] and c['goal_setter']['completed']==c['reset']['completed']+c['goal_bind']['completed']
            and c['explicit_render']['completed']==sum(r['PNG'] for r in rows),'actual whole native axis identities')
    except BaseException as exc:errors.append(dict(stage='final_check',error=repr(exc)))
    errors.extend(meter.secondary)
    clean=not errors and all(r['first_error'] is None and r['status'] in ('completed20','native_stopped_warmup','native_stopped_control') for r in rows)
    report.update(status='known_feedback_gradient_recorded' if clean else 'partial_or_failed',counts=meter.counts,pending=meter.pending,
        completed20=sum(r['full_horizon'] for r in rows),wall_seconds=time.monotonic()-data.get('job_started',meter.started))
    store.finalizing=True;store.emit('run_finished',status=report['status'],counts=meter.counts,nodes_sha256=digest(nodes))
    report.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    return report


def build():
    original=p3.build();templates=[]
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    for s,layer in enumerate(original['layers']):
        t=enc.load_task(layer['task_export'],layer['task_sha256'])['gn_task'];base=original['tasks'][s]
        require(all(t[k]==base[k] for k in ('root_context','goal','projection','action_mean','action_std'))
            and t['packet_logit_cap']==float(.75*np.sqrt(5)) and t['model_sha256']==enc.MODELS[str(enc.POLICY/'weights.pt')],
            'same real known P3 task/context/checkpoint/identityQ/action domain')
        templates.append(t)
    require(gradient.UPDATES==32 and gradient.STEP_SIZE==1. and gradient.budget()['predict']==136,'frozen original scalar optimizer configuration')
    sources=deepcopy(original['sources'])
    for module in (p3,transport,gradient,gradient.screen,gradient.gn,gradient.core):sources[str(Path(module.__file__).resolve())]=sha(module.__file__)
    sources.update({str(f):sha(f) for f in own_files()})
    for f,h in sources.items():require(sha(f)==h,'complete source/model/task closure')
    return dict(schema=NAME+'.prepare',contract=contract(),layers=original['layers'],tasks=original['tasks'],templates=templates,
        policy=original['policy'],expected_model_hash=original['expected_model_hash'],modelbinding=original['modelbinding'],
        source_manifest=original['source_manifest'],sources=sources,original_native_prepare=original['original_native_prepare'],
        known_only=True,P1_admission=False,P3_admission=False,freshness=False)


def exclusive(path):
    p=Path(path).absolute();require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and p.resolve()==p and not p.exists(),
        'new exclusive own-stem output');return p


def write(path,value):
    path=exclusive(path);payload=raw(value)
    require(len(payload)<=contract()['final_reserve_bytes'] and shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(payload),'bounded new metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared_no_execution'))


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'actual own prepare SHA');p=signal.read(path)
    require(p==build() and sha(path)==expected,'entire task/model/history/optimizer/caps prepare reconstruction');return p


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA AND physics before I/O')
    started=time.monotonic();p=verify(prepared,expected);output=exclusive(output);data=p3.source_data(p)
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    data['visual_tasks']={s:dict(gn_task=t,model_bindings=enc.MODELS) for s,t in enumerate(p['templates'])};data['job_started']=started
    store=Store(output);store.put('execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    store.put('startup_roster.json',dict(rows=roster(),nodes=node_roster()))
    model=before=host=report=None;errors=[];flags=None
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def check():
            require(time.monotonic()-started<contract()['wall_seconds'] and torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes']
                and torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'whole-stage wall/CUDA cap');store.check()
            require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32 and not torch.is_autocast_enabled('cuda'),'TF32/autocast off')
        check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);check()
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original frozen checkpoint tensor SHA')
        for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['sources'],'presealed model class')
        def final():
            enc.low.frozen(model);after=signal.model_hash(model);require(before==after,'all persistent/nonpersistent model tensors unchanged')
            for f,h in p['sources'].items():require(sha(f)==h,'source/weights/config after SHA: '+f)
            return dict(model_unchanged=True,before=before,after=after,source_manifest_sha256=digest(p['sources']))
        def precision():
            return dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
                autocast=torch.is_autocast_enabled('cuda'),math=torch.backends.cuda.math_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),
                efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            numeric=precision()
            report=execute(p,data,store,old.Handler(None),dict(model=model,device=device,check_limits=check),old.native_factory,authorization=AUTH,final_check=final)
            report['precision_before']=numeric;report['precision_after']=precision()
            require(numeric==report['precision_after']==dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False),'actual original MATH precision')
        torch.cuda.synchronize(device);check()
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        if report is None:report=data.get('live_report') or dict(schema=NAME+'.report',status='partial_or_failed',contract=contract(),rows=roster(),nodes=node_roster(),errors=[],fixed_denominator=4,node_denominator=16,known_roots=2)
        report['errors'].extend(errors)
        if errors:report['status']='partial_or_failed'
        report.update(prepared=str(Path(prepared).absolute()),prepare_sha256=expected,host=host,wall_seconds=time.monotonic()-started,
            P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False,source_manifest_sha256=digest(p['sources']),CUDA_peak_bytes=None)
        if model is not None:
            try:
                report['model_before']=before;report['model_after']=signal.model_hash(model);require(before==report['model_after'],'cleanup frozen model')
                report['CUDA_peak_bytes']=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
            except BaseException as exc:report['errors'].append(dict(stage='cleanup_model',error=repr(exc)));report['status']='partial_or_failed'
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        if report['wall_seconds']>=contract()['wall_seconds']:report['errors'].append(dict(stage='wall',error='whole-stage cap'));report['status']='partial_or_failed'
        store.put('report.json',report,final=True)
    return dict(output=str(output),report_sha256=sha(output/'report.json'),status=report['status'])


OPT_FIELDS={
    'operation_attempted':{'event'},'operation_completed':{'event'},'AD_API_attempt':{'entry'},'AD_API_completed':{'entry'},
    'evaluation_request':{'mode','logits','commands'},'evaluation_complete':{'mode','prediction'},'native_packet':{'mode','packet','context'},
    'gradient_request':{'iteration','logits','commands'},'gradient_forward_packet':{'packet','context'},'gradient_forward_complete':{'forward'},
    'gradient_complete':{'gradient'},'proposal_frozen':{'iteration','step_size','logits','parent_logits'},
    'iterate_saved':{'candidate'},'winner_frozen_before_replay':{'iteration','candidate'}}


def audit_plan(request,template,result,events):
    t=validate_request(request);gt=gradient_task(request,template)
    require(result['schema']==NAME+'.plan' and result['status']=='sealed' and result['request_sha256']==digest(request)
        and result['gradient_task_sha256']==digest(gt),'exact new observed-node model plan')
    known={'node_started','node_frozen','optimizer','forward_started','forward_finished','program_packet','program_boundary',
        'io_input','io_return','raw_backward','resource_attempt','resource_return'}
    require(all(e['kind'] in known for e in events),'unknown/failed native event rejected, no filtering')
    fields={'node_started':{'request_sha256','gradient_task_sha256'},'node_frozen':{'plan_sha256','commands'},
        'optimizer':{'event'},'forward_started':{'phase','group','mode'},'forward_finished':{'phase','commands','contexts'},
        **{k:{'scope','phase','step','operation_index','arrays'} for k in ('program_packet','program_boundary','io_input','io_return','raw_backward')},
        'resource_attempt':{'entry'},'resource_return':{'entry'}}
    require(all(set(e)=={'kind'}|fields[e['kind']] for e in events),'all native event fields accounted, no ignored labels')
    opts=[]
    for e in events:
        if e['kind']=='optimizer':
            core.keys(e,('kind','event'));o=e['event'];require(o['kind'] in OPT_FIELDS and
                set(o)=={'kind','evaluation_id'}|OPT_FIELDS[o['kind']],'complete exact optimizer native ABI');opts.append(o)
    gradient.choice(gt,result['result']);gradient.journal_check(gt,result['result'],opts)
    require(result['commands']==result['result']['final_prediction']['commands'][:5]
        and result['optimizer_counts']==result['result']['resources'],'selected first5 only / actual optimizer ledger')
    p3.audit_resources(result['resource_receipt'],events)
    require(result['resource_receipt']['attempted']==result['resource_receipt']['completed']==node_caps(),'full original API/module/grad units')
    require(events[0]==dict(kind='node_started',request_sha256=digest(request),gradient_task_sha256=digest(gt))
        and events[-1]==dict(kind='node_frozen',plan_sha256=digest(result),commands=result['commands']),'exact durable new node bookends')
    starts=[e for e in events if e['kind']=='forward_started'];ends=[e for e in events if e['kind']=='forward_finished']
    require(len(starts)==len(ends)==68 and [e['phase'] for e in starts]==[f'forward_{i:03}' for i in range(68)]
        and [e['phase'] for e in ends]==[e['phase'] for e in starts],'all136 native forward packets, no opaque RAMreceipt')
    # Gradient raw is exactly the last evaluated current iterate, not a new proposal.
    prior=None;expected=[]
    for e in opts:
        if e['kind']=='evaluation_complete':
            prior=e['prediction'];expected.append((e['evaluation_id'],e['mode'],prior['commands'],prior['contexts']))
        elif e['kind']=='gradient_forward_complete':expected.append((e['evaluation_id'],'gradient',prior['commands'],e['forward']['contexts']))
    for start,end,(group,mode,commands,path) in zip(starts,ends,expected,strict=True):
        require((start['group'],start['mode'])==(group,mode) and end['commands']==commands and end['contexts']==path,'literal forward/evaluation/gradient order')
        p3.audit_trajectory(t,events,start['phase'],2,commands,path,program=True)
    ops=result['resource_receipt']['events'];linked={}
    for e in events:
        if e['kind'] not in ('io_input','io_return','raw_backward'):continue
        ident=e['operation_index'];require(type(ident) is int and 0<=ident<len(ops),'every IO bound to actual charged op')
        op=ops[ident];require(e['scope']==op['scope'],'native scope bound to actual operation')
        linked.setdefault(ident,[]).append(e)
        for b in e['arrays'].values():require(signal.unblob(b).dtype==np.float32,'native float32 tensor bytes')
    positions={id(e):i for i,e in enumerate(events)}
    attempts={e['entry']['index']:i for i,e in enumerate(events) if e['kind']=='resource_attempt'}
    returns={e['entry']['index']:i for i,e in enumerate(events) if e['kind']=='resource_return'}
    for op in ops:
        es=linked.get(op['index'],[])
        if op['resource']=='vjp':require(es==[],'VJP direction is inclusive same reverse, not another tensor call');continue
        want=['raw_backward'] if op['resource']=='raw_backward' else ['io_input','io_return']
        require([e['kind'] for e in es]==want,'all actual native entry/input/return buffers')
        attempt,returned=attempts[op['index']],returns[op['index']]
        require(all(attempt<positions[id(e)]<returned and e['phase']==op['scope']['forward'] for e in es),'input/return buffers inside exact native operation')
        if op['resource'] in ('action_encoder','predict','predictor','pred_proj'):
            names={'action_encoder':('0',),'predict':('latent','act_emb'),'predictor':('0','1'),'pred_proj':('0',)}[op['resource']]
            core.keys(es[0]['arrays'],[op['resource']+':'+k for k in names]);core.keys(es[1]['arrays'],[op['resource']+':output'])
            require(list(signal.unblob(es[0]['arrays'][op['resource']+':'+names[0]]).shape)==op['shape'],'charged shape equals actual native input')
        if op['resource']=='grad':
            a,b=({k:signal.unblob(v) for k,v in e['arrays'].items()} for e in es)
            core.keys(a,('grad:loss','grad:raw'));core.keys(b,('grad:output',))
            # NativeSignal.blob uses ascontiguousarray: a runtime scalar is
            # serialized as shape[1], never an arbitrary vector objective.
            require(a['grad:loss'].shape==(1,) and a['grad:raw'].shape==b['grad:output'].shape==(10,2),'original scalar-blob ABI / rawleaf shape')
            group=op['scope']['group'];it=int(group.split('_')[-1])-1;g=result['result']['gradients'][it]
            require(float(a['grad:loss'][0])==g['native_float32_loss'] and np.array_equal(a['grad:raw'],result['result']['candidates'][it]['commands'])
                and np.array_equal(b['grad:output'],g['raw_control_gradient']),'loss/raw point/actual derivative bound to own original gradient')
        elif op['resource']=='raw_backward':
            a=signal.unblob(es[0]['arrays']['value']);it=int(op['scope']['group'].split('_')[-1])-1
            require(np.array_equal(a,result['result']['gradients'][it]['raw_control_gradient']),'actual raw leaf callback matches scalar grad')
    return dict(predict_API=136,scalar_grad=32,VJP_directions=32,backward_engine_calls=None,FLOPs=None,winner=result['result']['winner_iteration'])


def physical_census(events,r):
    """Unchanged physical ABI checker, NOT an original P3 algorithm receipt.

    Real new schema/chain/contract are checked first. Only transport checker
    constants/status are projected into local non-mutating views; no file,
    external receipt, model schedule or original hash is issued/rewritten.
    """
    require(r['schema']==NAME+'.report' and r['contract']==contract() and events[0]['contract']==contract()
        and events[-1]['status']==r['status'],'actual new physical contract and transport bookends')
    require(contract()['physical_caps']==p3.contract()['physical_caps'] and contract()['visual_caps']==p3.contract()['visual_caps'],
        'only exact same native resource ABI may share physical census')
    view=deepcopy(events);vr=deepcopy(r)
    view[0]['contract']=p3.contract()
    if r['status']=='known_feedback_gradient_recorded':vr['status']=view[-1]['status']='known_same_feedback_recorded'
    return transport.census(view,vr)


def consume(output,report_sha,prepared,prepare_sha):
    output=Path(output).absolute();p=verify(prepared,prepare_sha)
    require(sha(output/'report.json')==report_sha,'external new feedback-gradient report SHA')
    r=signal.read(output/'report.json',contract()['final_reserve_bytes'])
    require(r['schema']==NAME+'.report' and r['contract']==contract() and len(r['rows'])==4 and len(r['nodes'])==16
        and r['fixed_denominator']==4 and r['node_denominator']==16 and r['known_roots']==2
        and all(r[k] is False for k in ('P1_admission','P3_admission','freshness','scientific_admission')),'own full denominators/nonadmission')
    require(r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha
        and old.read(output/'execution_binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'actual own execution binding')
    if 'journal_sha256' not in r:
        require(r['status']=='partial_or_failed','startup failure not success')
        return dict(schema=NAME+'.consume',status='startup_failure_preserved',complete=False,rows=r['rows'],nodes=r['nodes'],P1_admission=False,P3_admission=False)
    events=transport.load_events(output,r);census=physical_census(events,r)
    for slot in r['nodes']:
        if 'native_journal_sha256' not in slot:continue
        journal=output/(slot['episode_id']+f"_b{slot['boundary']:02}")/'events.jsonl'
        require(sha(journal)==slot['native_journal_sha256'],'all completed/partial node native journal bytes')
        es,tail=signal.journal(journal);require(tail==slot['native_journal_tail'],'actual complete native tail')
        if slot['status']=='sealed':audit_plan(slot['request'],p['templates'][slot['source_slot']],slot['receipt'],es)
        else:
            failures=[e for e in es if e['kind']=='node_failed'];require(len(failures)<=1,'unique optional failure disposition; interrupted prefix remains unknown')
            if failures:p3.audit_resources(failures[0]['partial']['resource_receipt'],es)
    clean=r['status']=='known_feedback_gradient_recorded' and not r['errors'] and not r['pending'] and census['complete']
    if not clean:return dict(schema=NAME+'.consume',status='partial_failclosed',complete=False,rows=r['rows'],nodes=r['nodes'],physical_census=census,P1_admission=False,P3_admission=False)
    rgb=transport.audit_rgb(output,r,p);references=p3.source_data(p)['references'];audited=[]
    for row in r['rows']:
        eid=row['episode_id'];s=row['source_slot'];base=p['tasks'][s];ref=references[s]
        require(old.read(output/(eid+'_episode.json'))==row,'durable actual episode')
        es=[e for e in events if e.get('episode_id')==eid];warm=[e['receipt'] for e in es if e['kind']=='fresh_observation' and e['receipt']['phase']=='warmup']
        if row['initial_exact']:
            for k,b in enumerate((0,5,10)):
                require(warm[b]['pixel_sha256']==old.array_sha(ref['history'][k]) and np.array_equal(warm[b]['state'],ref['states'][k])
                    and warm[b]['snapshot']==ref['dynamics'][k],'original root/warm10 state/history/dynamics exact')
        observed=[e['receipt'] for e in es if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan']
        dispatch=[e for e in es if e['kind']=='authorized_dispatch'];returns=[e for e in es if e['kind']=='control_return']
        require(len(dispatch)==len(returns)==len(row['executed_raw'])==row['control_returned'],'actual returned raw action count')
        cached=deepcopy(base['root_context'])
        for encoded in row['encodings']:
            b=encoded['step'];ns=eid+f'_b{b:02}';er=old.read(output/(ns+'_encoding.json'));response=er['response']
            require(old.source.digest(er)==encoded['receipt_sha256'] and response['source_verified'] is response['model_unchanged'] is True
                and response['device_type']=='cuda' and response['model_hash_before']==response['model_hash_after']==p['modelbinding']['tensor_sha256'],
                'actual current frozen HOST CUDA visual authority')
            z=p3.load_array(output,response['latent']);old.native(z)
            if b==20:
                require(row['full_horizon'] and row['actual_Q20']==float(np.mean((z.astype(float)-np.asarray(base['goal']))**2)),'actualQ20, NOT speculative horizon25')
                continue
            a=np.concatenate((ref['raw_history'],np.asarray(row['executed_raw'][:b],np.float32).reshape(-1,2)))[-10:]
            if b==0:cached['latent_history'][-1]=z.tolist();cached['action_history']=moving.normalized(base,a)
            else:cached=moving.roll(cached,z,a,base)
            slot=next(n for n in r['nodes'] if n['episode_id']==eid and n['boundary']==b)
            if slot['status']!='sealed':continue
            request=slot['request'];validate_request(request,base)
            require(request['task']['root_context']==cached and request['raw10']==a.tolist()
                and old.read(output/(ns+'_request.json'))==request,'own actual observed history/raw, no root-only reuse')
            seal=slot['seal'];require(old.read(output/(ns+'_seal.json'))==seal and seal['schema']==NAME+'.dispatch'
                and seal['seal_sha256']==digest({k:v for k,v in seal.items() if k!='seal_sha256'}) and seal['request_sha256']==digest(request)
                and seal['plan_sha256']==digest(slot['receipt']) and seal['commands']==slot['receipt']['commands']
                and seal['native_journal_sha256']==slot['native_journal_sha256'],'new model-only native plan frozen before dispatch')
            ob=next(x for x in observed if x['step']==b)
            sealed=next(e['index'] for e in es if e['kind']=='boundary_sealed' and e['boundary']==b)
            encoded_event=next(e['index'] for e in es if e['kind']=='boundary_encoded' and e['actual_step']==b)
            selected=[e for e in dispatch if e['packet_boundary']==b]
            require(ob['observed_event_index']<encoded_event<sealed and request['observation']['encoded_index']==encoded_event
                and [e['command'] for e in selected]==seal['commands'][:len(selected)] and all(e['index']>sealed for e in selected),
                'observe -> encode -> optimize -> seal -> first5 dispatch causal chain')
            audited.append(dict(episode_id=eid,boundary=b,request_sha256=digest(request),plan_sha256=digest(slot['receipt'])))
        for d,ret,ob,command in zip(dispatch,returns,observed[1:],row['executed_raw'],strict=True):
            metric=old.kernel.metric(ref['goal'],ob['state'])
            require(d['step']==ret['step']==ob['step'] and d['command']==ret['command']==command and ret['index']>d['index']
                and np.array_equal(ob['goal'],ref['goal']) and ob['goal_rgb_sha256']==old.array_sha(ref['goal_rgb'])
                and np.asarray(ob['goal_pose'],np.float64).tobytes()==np.asarray(warm[-1]['goal_pose'],np.float64).tobytes()
                and ret['reward']==-metric['native_distance'] and ret['terminated']==metric['native_success'],
                'actual native whole agent+object+angle reward/success/unchanged goalpose, no object-only substitution')
        if returns:require(old.kernel.metric(ref['goal'],observed[-1]['state'])==row['final_metrics'],'actual final native task metric')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash']
        and r['final_check']['model_unchanged'] is True and r['source_manifest_sha256']==digest(p['sources']),'full frozen model/source identity')
    require(r['precision_before']==r['precision_after']==dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
        and 0<=r['wall_seconds']<contract()['wall_seconds'] and len(r['CUDA_peak_bytes'])==2
        and all(type(v) is int and 0<=v<=contract()[k] for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'whole numeric/wall/CUDA caps')
    for f,h in p['sources'].items():require(sha(f)==h,'final original sources/models unchanged')
    require(sha(output/'report.json')==report_sha and sha(output/'journal.jsonl')==r['journal_sha256'],'final output closure')
    return dict(schema=NAME+'.consume',status='real_feedback_gradient_verified_NOT_admitted',complete=True,fixed_denominator=4,node_denominator=16,
        known_roots=2,rows=r['rows'],nodes=r['nodes'],physical_census=census,RGB_preprocessing=rgb,audited=audited,
        native_metrics=[r['final_metrics'] for r in r['rows']],equal_FLOPs=False,total_FLOPs=None,backward_engine_calls=None,
        P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('prepared','prepare-sha','output','report-sha','receipt'):q.add_argument('--'+key)
    q.add_argument('--authorize-known-cuda-and-physics',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build())
    elif a.command=='verify':v=dict(schema=verify(a.prepared,a.prepare_sha)['schema'],verified=True)
    elif a.command=='run':
        require(a.authorize_known_cuda_and_physics,'MAIN explicit CUDA AND native physics flag')
        v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'new own durable receipt path required');v=write(a.receipt,consume(a.output,a.report_sha,a.prepared,a.prepare_sha))
    print(raw(v).decode(),flush=True)


if __name__=='__main__':main()
