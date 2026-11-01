"""Prefix-specific native CUDA AD response. prepare/verify/inspect are saved-only.

MAIN exclusively owns run. No encoder, fitting, physics, initial-J reuse, or
closed-loop efficacy claim. Serial reverse AD has 192 scalar VJP directions.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sysconfig
import time
import numpy as np
from experiments.spectral_reachability import p4_feedback_root_sourcepack_20261027_v1 as source
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler

NAME='p4_feedback_root_cuda_response_20261027_v1'
REPO=source.REPO
io=source.saved
require=io.require
RESERVE=200*1024**2
OUTPUT_CAP=12*1024**2
REPORT_RESERVE=1024**2
ARMS=('imagined','actual')
PHASES=('nominal','jacobian',*[f'candidate_{i}' for i in range(10)],'replay')
SOURCEPACK=REPO/'results/p4_feedback_root_sourcepack_20261027_v1_root0_prepare01'
SOURCEPACK_SHA='4de418b03fb7e4a664d9e5d04d8c78b7f7ac467e89f04677329b463fb6489fe2'


def caps(*,wall_seconds,allocated_bytes,reserved_bytes,grad_api_per_arm,engine_api_per_arm):
    for n in (allocated_bytes,reserved_bytes,grad_api_per_arm,engine_api_per_arm):
        require(type(n) is int and n>0,'explicit positive integer cap')
    require(type(wall_seconds) in (int,float) and math.isfinite(wall_seconds) and wall_seconds>0,'explicit finite walltime cap')
    require(reserved_bytes>=allocated_bytes and grad_api_per_arm>=192 and engine_api_per_arm>=192,'caps must permit planned serial AD, reserved>=allocated')
    return dict(wall_seconds=float(wall_seconds),allocated_bytes=allocated_bytes,reserved_bytes=reserved_bytes,
        grad_api_per_arm=grad_api_per_arm,engine_api_per_arm=engine_api_per_arm,
        jacobian_api_per_arm=1,predict_per_arm=13,action_encoder_per_arm=13,
        mathematical_VJP_per_arm=192,context_tokens_per_arm=39,arms=2,
        encoder_cap=0,projector_cap=0,physics_steps_cap=0,backward_api_cap=0,
        output_bytes_cap=OUTPUT_CAP,remaining_disk_reserve_bytes=RESERVE,retries=False,
        deadline_semantics='POSIX alarm plus synchronized boundary checks; not GPU-kernel preemption')


def validate_pack(p):
    require(p['schema']==source.NAME+'.protocol' and p['status']=='prepared_missing_actual_response' and
        p['runtime_admissible'] is p['response_completed'] is p['independent_confirmation'] is p['P1_admission'] is p['P4_admission'] is False,'saved-only sourcepack scope')
    require([r['arm'] for r in p['response_requests']]==list(ARMS),'exact paired arm order')
    require(p['root_sha256'] not in p['prior_root_sha256s'],'prior root exclusion')
    for r in p['response_requests']:
        io.context(r['context'])
        require(r['boundary_step']==5 and r['must_recompile'] is True and r['can_reuse_initial_jacobian'] is False and
            r['context_sha256']==io.digest(r['context']),'root-specific causal compile request')
    a,b=(r['context'] for r in p['response_requests'])
    require(a['action_history']==b['action_history'] and a['latent_history'][:2]==b['latent_history'][:2], 'same causal older history, only latest observed boundary differs')
    io.array(p['goal'],(192,));q=io.array(p['q'],(192,192))
    require(np.allclose(q,q.T,rtol=0,atol=1e-8) and np.allclose(q@q,q,rtol=0,atol=1e-8) and np.trace(q)>.5,'original nonempty Q')
    for k in ('mean','std'):io.array(p['normalization'][k],(2,))
    require(np.all(np.asarray(p['normalization']['std'])>0),'original positive normalization')
    require({Path(x).name for x in p['model_bindings']}=={'config.json','weights.pt'} and
        len({str(Path(x).parent) for x in p['model_bindings']})==1,'one original local frozen model')
    return p


def sources():
    site=Path(sysconfig.get_paths()['purelib'])
    local=[Path(__file__).resolve(),REPO/'tests'/('test_'+NAME+'.py'),REPO/'innerdocs'/(NAME+'.md'),
        Path(source.__file__).resolve(),REPO/'tests'/('test_'+source.NAME+'.py'),REPO/'innerdocs'/(source.NAME+'.md'),
        Path(compiler.__file__).resolve(),REPO/'experiments/spectral_reachability/pusht_full_packet_chart_20261027.py',
        REPO/'experiments/spectral_reachability/pusht_d2p_packet_contract.py',REPO/'experiments/pusht_protocol_contracts.py']
    extra=['torch/autograd/__init__.py','torch/autograd/functional.py','torch/autograd/graph.py']
    return sorted(set(io.source_files()+local+[site/x for x in extra]))


def material(pack_dir,pack_sha,limits):
    p=validate_pack(source.verify(pack_dir,pack_sha))
    require(limits==caps(**dict(wall_seconds=limits['wall_seconds'],allocated_bytes=limits['allocated_bytes'],reserved_bytes=limits['reserved_bytes'],
        grad_api_per_arm=limits['grad_api_per_arm'],engine_api_per_arm=limits['engine_api_per_arm'])),'exact cap contract')
    config=io.read(next(x for x in p['model_bindings'] if Path(x).name=='config.json'))
    require(config['encoder'].get('pretrained') is False and config['_target_']=='stable_worldmodel.wm.lewm.LeWM','local model; no pretrained download')
    return dict(schema=NAME+'.protocol',sourcepack=str(Path(pack_dir).resolve()),sourcepack_prepare_sha256=pack_sha,
        root_sha256=p['root_sha256'],source_arm=p['source_arm'],active_metric_id=p['active_metric_id'],
        requests=p['response_requests'],goal=p['goal'],q=p['q'],normalization=p['normalization'],model_bindings=p['model_bindings'],
        prefix_evidence=p['prefix_evidence'],limits=limits,arm_order=list(ARMS),
        compiler=dict(horizon_tokens=1,jacobian_shape=[192,10],coordinate='tanh(logits10).reshape(5,2)',
            AD='torch.autograd.functional.jacobian reverse-mode vectorize=False strict=True create_graph=False',
            bank='sealed spectral_bank: zero,ridge,2 radii x 2 modes x 2 signs',
            damping=compiler.DAMPING,radius=compiler.CAP,radii=list(compiler.RADII),
            selection='minimum actual nonlinear original-Q goal MSE; stable candidate index ties',
            independent_replay='full successor context exact equality; no physical action dispatch'),
        work=dict(predict_samples=26,action_encoder_calls=26,jacobian_API_calls=2,grad_API_calls_expected=384,
            engine_API_entries_expected=384,mathematical_VJP_directions=384,processed_context_tokens=78,
            encoder=0,projector=0,physics=0,FD=0,training=False),
        known_roots_only=True,independent_confirmation=False,runtime_admissible=False,P1_admission=False,P4_admission=False)


class Store:
    def __init__(self,path):
        self.path=Path(path).absolute();self.path.mkdir(exist_ok=False)
    def check(self,n=0,final=False):
        require(io.disk_bytes(self.path)+n<=OUTPUT_CAP-(0 if final else REPORT_RESERVE),'bounded output cap')
        require(shutil.disk_usage(self.path).free>=RESERVE+n,'200MiB reserve')
    def write(self,name,value,final=False):
        raw=(io.io.canonical(value)+'\n').encode();self.check(len(raw),final)
        with (self.path/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        io.fsync_directory(self.path);return io.sha(self.path/name)


def prepare(pack_dir,pack_sha,output,limits):
    output=Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free>=RESERVE+OUTPUT_CAP,'NEW prepare / storage headroom')
    p=material(pack_dir,pack_sha,limits);files=sources()
    require(sum(f.stat().st_size for f in files)<6*1024**2,'bounded explicit source closure')
    p['sources_sha256']={str(f):io.sha(f) for f in files}
    store=Store(output);store.write('protocol.json',p)
    seal=io.create_seal(files=files,args_json=io.io.canonical(dict(protocol_sha256=io.sha(output/'protocol.json'),limits=limits)),output_dir=output/'source_seal')
    store.write('prepare.json',dict(schema=NAME+'.prepare',status='prepared_not_run',protocol_sha256=io.sha(output/'protocol.json'),source_seal=seal))
    return io.sha(output/'prepare.json')


def verify(folder,expected):
    folder=Path(folder).resolve();require(io.sha(folder/'prepare.json')==expected,'external prepare SHA')
    r=io.read(folder/'prepare.json');p=io.read(folder/'protocol.json')
    require(r['schema']==NAME+'.prepare' and r['status']=='prepared_not_run' and io.sha(folder/'protocol.json')==r['protocol_sha256'],'prepared identity')
    io.verify_seal(output_dir=r['source_seal']['output_dir'],expected_manifest_sha256=r['source_seal']['manifest_sha256'])
    for path,h in p['sources_sha256'].items():require(io.sha(path)==h,'source drift')
    rebuilt=material(p['sourcepack'],p['sourcepack_prepare_sha256'],p['limits']);rebuilt['sources_sha256']=p['sources_sha256']
    require(io.same(rebuilt,p),'exact response protocol reconstruction');return p


class Journal:
    def __init__(self,store):
        self.store=store;self.stream=(store.path/'events.jsonl').open('x');io.fsync_directory(store.path)
        self.failed=False
    def __call__(self,event):
        require(not self.failed,'journal previously failed')
        try:
            raw=(io.io.canonical(event)+'\n').encode();self.store.check(len(raw))
            self.stream.write(raw.decode());self.stream.flush();os.fsync(self.stream.fileno())
        except BaseException:self.failed=True;raise
    def close(self):self.stream.close()


class Ledger:
    """Fail-sticky nonrefundable attempts; mathematical/API/module units separate."""
    def __init__(self,limits,emit,clock=time.monotonic,sample=lambda:(0,0),sync=lambda:None):
        self.limits,self.emit,self.clock,self.sample,self.sync=limits,emit,clock,sample,sync
        self.started=clock();self.arm=None;self.phase=None;self.phase_index=-1;self.arm_index=-1
        self.entries=[];self.blocked=[];self.failed=False;self.active=[];self.peaks=[0,0]
    def check(self):
        require(not self.failed,'failed ledger; no retries/refunds')
        require(self.clock()-self.started<=self.limits['wall_seconds'],'walltime cap')
        peak=self.sample();self.peaks=[max(a,b) for a,b in zip(self.peaks,peak)]
        require(peak[0]<=self.limits['allocated_bytes'] and peak[1]<=self.limits['reserved_bytes'],'CUDA allocated/reserved cap')
    def reject(self,kind,message):
        self.failed=True;event=dict(kind=kind,arm=self.arm,phase=self.phase,error=message)
        self.blocked.append(event)
        self.emit(dict(event='blocked',**event));raise ValueError(message)
    def begin_arm(self,arm):
        self.check();require(not self.active and self.arm_index+1<len(ARMS) and arm==ARMS[self.arm_index+1],'arm sequence')
        if self.arm is not None:self.complete_arm()
        self.arm_index+=1;self.arm=arm;self.phase_index=-1;self.phase=None
        self.emit(dict(event='arm_intent',arm=arm))
    def set_phase(self,phase):
        self.check();require(not self.active and self.phase_index+1<len(PHASES) and phase==PHASES[self.phase_index+1],'phase sequence')
        self.phase_index+=1;self.phase=phase;self.emit(dict(event='phase_intent',arm=self.arm,phase=phase))
    def call(self,kind,callback,device='cuda',shape=None):
        try:
            self.check();require(self.arm in ARMS and self.phase in PHASES,'active work slot')
            require(device=='cuda','CUDA only, no neural CPU fallback')
            own=[e for e in self.entries if e['arm']==self.arm and e['kind']==kind]
            if kind in ('action_encoder','predict'):
                require(shape==((1,3,10) if kind=='action_encoder' else (1,3,192)),'native full3 shape')
                require(len(own)<13 and not any(e['arm']==self.arm and e['phase']==self.phase and e['kind']==kind for e in self.entries),'one module entry per work phase')
                if kind=='predict':require(any(e['arm']==self.arm and e['phase']==self.phase and e['kind']=='action_encoder' and e['completed'] for e in self.entries),'encoder before predict')
                require(self.active==(['jacobian_api'] if self.phase=='jacobian' else []),'module not during reverse/reentrant work')
            elif kind in ('jacobian_api','grad_api','engine_api'):
                require(self.phase=='jacobian','AD only in compile phase')
                limit={'jacobian_api':1,'grad_api':self.limits['grad_api_per_arm'],'engine_api':self.limits['engine_api_per_arm']}[kind]
                require(len(own)<min(limit,1 if kind=='jacobian_api' else 192),'AD/API cap or exact useful-work ceiling')
                require(self.active=={'jacobian_api':[],'grad_api':['jacobian_api'],'engine_api':['jacobian_api','grad_api']}[kind],'AD nesting/entry mismatch')
                if kind=='grad_api':require(any(e['arm']==self.arm and e['kind']=='predict' and e['phase']=='jacobian' and e['completed'] for e in self.entries),'Jacobian forward before reverse')
            else:raise ValueError('forbidden API/module '+kind)
            e=dict(index=len(self.entries),arm=self.arm,phase=self.phase,kind=kind,completed=False)
            self.entries.append(e);self.active.append(kind)
            self.emit(dict(event='attempt',**e));value=callback();self.sync();self.check()
            e['completed']=True;self.emit(dict(event='return',**e));return value
        except BaseException as exc:
            self.failed=True
            if 'e' in locals():e['error']=io.error('dispatch',exc);self.emit(dict(event='error',**e))
            else:self.blocked.append(dict(kind=kind,error=str(exc)));self.emit(dict(event='blocked',kind=kind,arm=self.arm,phase=self.phase,error=str(exc)))
            raise
        finally:
            if 'e' in locals():self.active.pop()
    def complete_arm(self):
        self.check();require(self.phase_index==len(PHASES)-1 and not self.active,'all phases completed')
        own=[e for e in self.entries if e['arm']==self.arm]
        require(all(e['completed'] for e in own),'no failed/partial calls')
        require({k:sum(e['kind']==k for e in own) for k in ('action_encoder','predict','jacobian_api','grad_api','engine_api')}==
            dict(action_encoder=13,predict=13,jacobian_api=1,grad_api=192,engine_api=192),'exact useful work, not just under cap')
    def counts(self):
        return {arm:{k:dict(attempted=sum(e['arm']==arm and e['kind']==k for e in self.entries),
            completed=sum(e['arm']==arm and e['kind']==k and e['completed'] for e in self.entries))
            for k in ('action_encoder','predict','jacobian_api','grad_api','engine_api')} for arm in ARMS}


def compile_response(p,backend,ledger,store,rows):
    try:return _compile_response(p,backend,ledger,store,rows)
    except BaseException as exc:
        ledger.failed=True
        ledger.emit(dict(event='compile_failed',arm=ledger.arm,phase=ledger.phase,error=io.error('compile',exc)))
        raise


def _compile_response(p,backend,ledger,store,rows):
    goal=io.array(p['goal'],(192,));q=io.array(p['q'],(192,192))
    for request in p['requests']:
        arm=request['arm'];ledger.begin_arm(arm);backend.bind(request)
        row=dict(arm=arm,status='started',root_sha256=p['root_sha256'],boundary_step=5,context_sha256=request['context_sha256'],candidates=[])
        rows.append(row);store.write(arm+'.intent.json',dict(request=request,root_sha256=p['root_sha256'],active_metric_id=p['active_metric_id']))
        ledger.set_phase('nominal');nominal_context=backend.nominal();io.context(nominal_context)
        io.history(request['context'],nominal_context,np.zeros((5,2)),p['normalization'])
        nominal=np.asarray(nominal_context['latent_history'])[-1]
        store.write(arm+'.nominal.json',nominal_context)
        ledger.set_phase('jacobian');jac,forward_context=backend.jacobian();j=io.array(jac,(192,10))
        store.write(arm+'.jacobian_forward.json',forward_context)
        store.write(arm+'.jacobian.json',j.tolist())
        require(io.same(forward_context,nominal_context),'AD forward matches fresh native nominal context')
        require(np.linalg.norm(j)>0,'nonzero real action derivative required')
        bank=compiler.spectral_bank(j,nominal,goal,q)
        require(len(bank['params'])==len(bank['kinds'])==10,'exact ten useful scored candidates')
        store.write(arm+'.bank.json',bank)
        for i,(params,kind) in enumerate(zip(bank['params'],bank['kinds'],strict=True)):
            logits=io.array(params,(10,));require(np.linalg.norm(logits)<=compiler.CAP+1e-10,'shared logit ball')
            commands=np.tanh(logits.astype(np.float32)).reshape(5,2).tolist()
            ledger.set_phase('candidate_'+str(i));store.write(arm+f'.candidate_{i}.intent.json',dict(index=i,commands=commands,kind=kind))
            context=backend.candidate(commands);io.context(context)
            io.history(request['context'],context,commands,p['normalization'])
            cost=float(np.mean((q@(np.asarray(context['latent_history'])[-1]-goal))**2));require(math.isfinite(cost),'finite nonlinear Q cost')
            candidate=dict(index=i,kind=kind,logits=logits.tolist(),commands=commands,context=context,cost=cost)
            store.write(arm+f'.candidate_{i}.json',candidate);row['candidates'].append(candidate)
        winner=min(row['candidates'],key=lambda x:(x['cost'],x['index']))
        ledger.set_phase('replay');replay=backend.replay(winner['commands']);store.write(arm+'.replay.json',replay)
        require(io.same(replay,winner['context']),'independent exact full successor replay')
        ledger.complete_arm()
        row.update(status='response_completed',winner=deepcopy(winner),independent_replay_exact=True,
            initial_J_reused=False,actual_physics=False,context_specific_AD=True)
        store.write(arm+'.response.json',row)


class Native:
    """No old initial-J or saved predicted output is accepted by this backend."""
    def __init__(self,torch,model,device,p):
        from experiments.spectral_reachability.pusht_d2p_packet_contract import normalize_live_packet
        self.t,self.model,self.device,self.p=torch,model,device,p;self.normalize=normalize_live_packet
    def bind(self,r):
        self.request=r;self.context=deepcopy(r['context'])
        self.emb=self.t.tensor(self.context['latent_history'],dtype=self.t.float32,device=self.device)[None]
        self.history=self.t.tensor(self.context['action_history'],dtype=self.t.float32,device=self.device)[None]
    def transition(self,raw):
        token=self.normalize(raw,mean=self.p['normalization']['mean'],std=self.p['normalization']['std'])
        outgoing=self.t.cat((self.history,token[None,None]),dim=1)
        successor=self.model.predict(self.emb,self.model.action_encoder(outgoing))[:,-1]
        require(bool(self.t.isfinite(successor).all()),'finite native successor')
        return self.t.cat((self.emb[:,1:],successor[:,None]),dim=1),outgoing[:,1:]
    def export(self,state,history):
        return dict(self.context,latent_history=state.detach()[0].cpu().tolist(),action_history=history.detach()[0].cpu().tolist())
    def nominal(self):
        with self.t.no_grad():return self.export(*self.transition(self.emb.new_zeros(5,2)))
    def jacobian(self):
        forward=[]
        def terminal(u):
            states,hist=self.transition(self.t.tanh(u).reshape(5,2));forward.append(self.export(states,hist))
            return states[0,-1]
        zero=self.emb.new_zeros(10).requires_grad_(True)
        j=self.t.autograd.functional.jacobian(terminal,zero,create_graph=False,strict=True,vectorize=False,strategy='reverse-mode')
        require(len(forward)==1,'one fresh AD forward')
        return j.detach().cpu().numpy(),forward[0]
    def candidate(self,commands):
        raw=self.t.tensor(commands,dtype=self.t.float32,device=self.device)
        with self.t.no_grad():return self.export(*self.transition(raw))
    def replay(self,commands):
        # Reconstruct from the original request, never reuse candidate tensors.
        t=self.t
        state=t.tensor(self.request['context']['latent_history'],dtype=t.float32,device=self.device)[None]
        hist=t.tensor(self.request['context']['action_history'],dtype=t.float32,device=self.device)[None]
        raw=t.tensor(commands,dtype=t.float32,device=self.device)
        token=self.normalize(raw,mean=self.p['normalization']['mean'],std=self.p['normalization']['std'])
        with t.no_grad():
            outgoing=t.cat((hist,token[None,None]),dim=1)
            next_state=self.model.predict(state,self.model.action_encoder(outgoing))[:,-1]
            result=self.export(t.cat((state[:,1:],next_state[:,None]),dim=1),outgoing[:,1:])
        io.context(result);return result


def install_hooks(t,model,ledger,hooks):
    for obj,key,kind in ((model,'predict','predict'),(model.action_encoder,'forward','action_encoder'),
        (t.autograd.functional,'jacobian','jacobian_api'),(t.autograd,'grad','grad_api'),
        (t.autograd,'_engine_run_backward','engine_api'),(t.autograd.graph,'_engine_run_backward','engine_api')):
        original=getattr(obj,key)
        def guarded(*args,_kind=kind,_original=original,**kwargs):
            tensor=args[0] if _kind in ('predict','action_encoder') else None
            return ledger.call(_kind,lambda:_original(*args,**kwargs),device='cuda' if tensor is None else getattr(getattr(tensor,'device',None),'type',None),
                shape=None if tensor is None else tuple(tensor.shape))
        hooks.install(obj,key,guarded)
    for obj,key,kind in ((model,'encode','encode'),(model.encoder,'forward','encoder'),(model.projector,'forward','projector'),(t.autograd,'backward','backward_api')):
        def denied(*args,_kind=kind,**kwargs):return ledger.reject(_kind,'forbidden '+_kind)
        hooks.install(obj,key,denied)


def run(folder,expected,output):
    p=verify(folder,expected);output=Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free>=RESERVE+OUTPUT_CAP,'NEW run / 200MiB reserve')
    store=Store(output);store.write('execution_binding.json',dict(prepared=str(Path(folder).resolve()),prepare_sha256=expected,sourcepack_prepare_sha256=p['sourcepack_prepare_sha256']))
    journal=Journal(store);ledger=Ledger(p['limits'],journal);rows=[];errors=[];hooks=io.Hooks()
    model=None;before=after=None;host=None;source_ok=False;old_alarm=None;old_timer=None
    try:
        require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'no existing process alarm; run only MAIN dedicated process')
        def deadline(signum,frame):raise TimeoutError('sealed walltime cap exceeded')
        old_alarm=signal.signal(signal.SIGALRM,deadline);old_timer=signal.setitimer(signal.ITIMER_REAL,p['limits']['wall_seconds'])
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats(device)
        ledger.sample=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        ledger.sync=lambda:torch.cuda.synchronize(device)
        # Allocator fraction is an additional physical allocation guard, not a substitute for measured caps.
        total=torch.cuda.get_device_properties(device).total_memory
        require(p['limits']['reserved_bytes']<=total,'reserved cap exceeds device memory')
        torch.cuda.set_per_process_memory_fraction(p['limits']['reserved_bytes']/total,device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        policy=Path(next(x for x in p['model_bindings'] if Path(x).name=='weights.pt')).parent
        model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
        ledger.sync();ledger.check();io.frozen(model)
        require(all(b.device==device for b in model.buffers()),'CUDA buffers only')
        before=io.tensor_hash(model)
        install_hooks(torch,model,ledger,hooks)
        with sdpa_kernel(SDPBackend.MATH):compile_response(p,Native(torch,model,device,p),ledger,store,rows)
    except BaseException as exc:errors.append(io.error('execution',exc))
    finally:
        # Disable deadline for preservation/verification; it cannot admit a timed-out run.
        execution_wall=ledger.clock()-ledger.started
        if old_timer is not None:signal.setitimer(signal.ITIMER_REAL,0)
        if old_alarm is not None:signal.signal(signal.SIGALRM,old_alarm)
        hooks.restore(errors)
        if model is not None:
            try:io.frozen(model);after=io.tensor_hash(model);ledger.check()
            except BaseException as exc:errors.append(io.error('final_model_or_resources',exc))
        try:verify(folder,expected);source_ok=True
        except BaseException as exc:errors.append(io.error('final_source',exc))
        complete=(execution_wall<=p['limits']['wall_seconds'] and not errors and not ledger.failed and not ledger.blocked and not journal.failed and source_ok and
            before is not None and before==after and len(rows)==2 and all(r['status']=='response_completed' for r in rows))
        try:journal(dict(event='run_finished',complete=complete,errors=errors))
        except BaseException as exc:errors.append(io.error('final_journal',exc));complete=False
        journal.close()
        report=dict(schema=NAME+'.receipt',status='two_prefix_specific_responses_saved' if complete else 'partial_or_failed',
            root_sha256=p['root_sha256'],source_arm=p['source_arm'],active_metric_id=p['active_metric_id'],
            prepare_sha256=expected,sourcepack_prepare_sha256=p['sourcepack_prepare_sha256'],model_bindings=p['model_bindings'],
            rows=rows,errors=errors,blocked=ledger.blocked,counts=ledger.counts(),entries=ledger.entries,
            limits=p['limits'],peaks=dict(allocated_bytes=ledger.peaks[0],reserved_bytes=ledger.peaks[1]),
            wall_seconds=execution_wall,total_wall_including_preservation=ledger.clock()-ledger.started,model_hash_before=before,model_hash_after=after,
            model_unchanged=before is not None and before==after,source_verified=source_ok,host=host,
            journal_sha256=io.sha(store.path/'events.jsonl'),new_encoder_calls=0,new_projector_calls=0,new_physical_steps=0,
            training=False,known_roots_only=True,independent_confirmation=False,runtime_admissible=False,P1_admission=False,P4_admission=False,
            scope='saved actual-prefix conditional response compilation; not live feedback control or physical improvement')
        report['output_files_sha256']={f.name:io.sha(f) for f in sorted(store.path.iterdir()) if f.is_file()}
        store.write('receipt.json',report,final=True)
    require(complete,'incomplete response; preserve receipt/journal, no automatic retry');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('inspect','prepare','verify','run'))
    parser.add_argument('--sourcepack',type=Path,default=SOURCEPACK);parser.add_argument('--sourcepack-sha',default=SOURCEPACK_SHA)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    parser.add_argument('--wall-seconds',type=float);parser.add_argument('--allocated-bytes',type=int);parser.add_argument('--reserved-bytes',type=int)
    parser.add_argument('--grad-api-per-arm',type=int);parser.add_argument('--engine-api-per-arm',type=int)
    a=parser.parse_args()
    if a.command=='inspect':
        p=validate_pack(source.verify(a.sourcepack,a.sourcepack_sha));print(json.dumps(dict(root=p['root_sha256'],requests=2,status='saved_source_verified_only',actual_new_calls=0)));return
    if a.command=='prepare':
        limits=caps(wall_seconds=a.wall_seconds,allocated_bytes=a.allocated_bytes,reserved_bytes=a.reserved_bytes,grad_api_per_arm=a.grad_api_per_arm,engine_api_per_arm=a.engine_api_per_arm)
        print(prepare(a.sourcepack,a.sourcepack_sha,a.output,limits));return
    if a.command=='verify':p=verify(a.prepared,a.prepare_sha);print(json.dumps(p['work']));return
    r=run(a.prepared,a.prepare_sha,a.output);print(r['status'])


if __name__=='__main__':main()
