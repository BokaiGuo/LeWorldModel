"""Fixed all42 prefixes, eight nonlinear suffix action-gradient steps; MAIN CUDA only.

New model-only algorithm; no outcomes, training, new visual acquisition or physics.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import inspect
from importlib.machinery import PathFinder
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_allmode_prefix_20261028_v2 as source
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v1 as transport
from experiments.spectral_reachability import p1_constructive_gradient_mpc_cuda_20261027_v1 as gd

NAME='agent_p1_nonlinear_suffix_library_20261028_v2'
AUTH='MAIN_FIXED42_NONLINEAR_SUFFIX_HOST_CUDA_ONLY'
ROOT=source.ROOT
core,signal,native=transport.core,transport.signal,transport.native
require,sha,digest,raw=core.require,signal.sha,core.digest,core.canonical
PARENT=ROOT/'results/agent_allmode_prefix_20261028_v2_cuda_MAIN01'
PARENT_SHA='c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d'
CAP=source.compiler.CAP


def contract():
    return dict(known_roots=2,root_conditions=4,prefixes_per_slot=42,candidate_denominator=168,
        order=[list(k) for k in source.ORDER],updates=8,eta=1.,suffix_iterates=9,
        prefix='original sealed literal rawfloat32[5,2], unchanged all42 including duplicates/null modes',
        incumbent='original float64 bounded_ridge witness then original NumPy float32 cast+tanh; NO postcast norm gate or repair',
        new_iterate='original gd.screen.projected_logits on float64 update, float32 inward rounding; suffix packet only',
        chart='np.tanh(np.asarray(logits,np.float32)).reshape(5,2); actual literal float32 buffers',
        derivative='scalar native float32 meanQ w.r.t. fresh suffix raw[5,2]; analytic (1-raw^2) exactly once',
        objective='original identity192 mean squared terminal Q only; no stage/reach/beta/penalty',
        selection='per-prefix stable(Q,iterate), then entire42 stable(Q,prefix_index); incumbent wins exact ties',
        prefix_context='own root fixed prefix prediction; never teacher/actual-mid context',
        new_predict=3376,new_scalar_reverse=1344,per_candidate_predict=20,per_candidate_reverse=8,
        per_slot_final_replay_predict=4,new_J=0,new_SVD=0,new_encoder=0,new_visual_projector=0,new_physics=0,
        original_allmode_compute=deepcopy(source.contract()['total']),total_FLOPs=None,equal_FLOPs=False,
        reverse_semantics='grad_API,VJP_directions,raw_backward_callbacks count the SAME scalar reverse sweeps, not additive',
        wall_seconds=1800,output_bytes=512*1024**2,final_reserve_bytes=64*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',training=False,outcome_input=False,retry=False,
        P1_admission=False,freshness=False,physical_release=False,scientific_admission=False)


def caps(n,reverse):
    c=dict.fromkeys(core.AXES,0)
    c.update(predict_API=n,predict_samples=n,predict_context_tokens=3*n,
        action_encoder_calls=n,action_encoder_samples=n,action_encoder_context_tokens=3*n,
        predictor_calls=n,predictor_samples=n,predictor_context_tokens=3*n,
        pred_proj_calls=n,pred_proj_rows=3*n,grad_API=reverse,VJP_directions=reverse,raw_backward_callbacks=reverse)
    return c


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
def chart(x):return np.tanh(np.asarray(x,np.float32)).reshape(5,2)
def bits(x):return np.asarray(x,np.float32).tobytes().hex()
def cost(goal,c):return float(np.mean((np.asarray(c['latent_history'][-1],np.float64)-np.asarray(goal,np.float64))**2))


def proposal(x,g):
    # Reuse the reviewed projection without changing its two-packet ABI.
    return gd.screen.projected_logits(np.stack((np.asarray(x,float)-np.asarray(g,float),np.zeros(10))),CAP)[0]


def incumbent_domain(c):
    x=source.compiler.finite(c['suffix_params'],(10,));s=c['suffix_solve']
    require(s['params']==c['suffix_params'] and s['radius']==CAP and s['norm']==float(np.linalg.norm(x))
        and s['norm']<=CAP and bits(chart(x))==bits(c['suffix_commands'])
        and np.array_equal(np.asarray(c['suffix_commands']),chart(x)),
        'original float64 ridge witness and exact original deployed cast/chart, no inverse/repair')
    return dict(solver_float64_norm=float(np.linalg.norm(x)),postcast_float64_norm=float(np.linalg.norm(x.astype(np.float32).astype(float))),
        cap=CAP,postcast_norm_is_diagnostic_only=True,original_raw_bits=bits(c['suffix_commands']))


def task_from(slot,expected_model_hash):
    # New task ABI binding, not a forged original core task/strategy receipt.
    t=dict(root_context=deepcopy(slot['packet']['root_context']),goal=deepcopy(slot['packet']['goal']),
        projection=deepcopy(slot['q']),action_mean=source.kernel.ACTION_MEAN.tolist(),action_std=source.kernel.ACTION_STD.tolist())
    require(np.array_equal(t['projection'],np.eye(192)),'original identity Q')
    b={k:digest([NAME,k,slot['task_id'],slot['task_sha256']]) for k in core.BINDINGS}
    b.update(task_id=slot['task_id'],task_sha256=slot['task_sha256'],blind_sha256=slot['blind_sha256'],
        model_sha256=expected_model_hash,goal_sha256=digest(t['goal']),projection_sha256=digest(t['projection']),
        normalizer_sha256=digest([t['action_mean'],t['action_std']]),context_sha256=digest(t['root_context']),
        action_domain_sha256=digest(core.action_domain()))
    t['binding']=b;return core.validate_task(t)


def material(p,r):
    require(len(p['slots'])==len(r['rows'])==4,'all four original known root-condition slots')
    slots=[]
    for i,(s,row) in enumerate(zip(p['slots'],r['rows'],strict=True)):
        require(s['index']==row['index']==i and (s['root_slot'],s['condition'])==source.ORDER[i],'fixed slot-major identity')
        source.validate_compilation(s['packet'],s['q'],row['result'])
        cs=[]
        for j,c in enumerate(row['result']['candidates']):
            domain=incumbent_domain(c)
            cs.append(dict(index=j,prefix_params=deepcopy(c['prefix_params']),prefix=deepcopy(c['prefix_commands']),
                prefix_bits=bits(c['prefix_commands']),proposal_kind=c['proposal_kind'],
                logits=deepcopy(c['suffix_params']),incumbent_raw=deepcopy(c['suffix_commands']),incumbent_domain=domain,
                original_candidate_sha256=digest(c),original_predicted_prefix=deepcopy(c['predicted_prefix']),
                original_terminal_context=deepcopy(c['context']),original_terminal_Q=c['nonlinear_terminal_cost']))
        require(len(cs)==42 and [c['index'] for c in cs]==list(range(42)),'all42 immutable prefixes/incumbents')
        slots.append(dict(index=i,root_slot=s['root_slot'],condition=s['condition'],task=task_from(s,p['expected_model_hash']),
            candidates=cs,original_generation_winner=dict(index=row['result']['winner']['index'],
                commands=deepcopy(row['result']['winner']['commands']),selection='original stage+ridge reach, beta1')))
    return slots


# Exact installed class-source omissions established by MAIN's load-only trace.
# PathFinder walks package metadata WITHOUT importing/executing torch/transformers.
INSTALLED_CLASS_MODULES=(
    'torch.nn.modules.activation','torch.nn.modules.batchnorm',
    'torch.nn.modules.container','torch.nn.modules.conv',
    'torch.nn.modules.dropout','torch.nn.modules.linear',
    'torch.nn.modules.normalization','transformers.activations')
V1_SOURCE=ROOT/'experiments/spectral_reachability/agent_p1_nonlinear_suffix_library_20261028_v1.py'
V1_SOURCE_SHA='9472538fa3c63ca66626177030b3e9b656737b4444578f7338af76948995e864'


def installed_class_sources():
    sources={}
    for name in INSTALLED_CLASS_MODULES:
        search=None;parts=name.split('.')
        for i in range(len(parts)):
            spec=PathFinder.find_spec('.'.join(parts[:i+1]),search)
            require(spec is not None,'required installed model class module missing: '+name)
            if i+1<len(parts):
                require(spec.submodule_search_locations is not None,'required installed model package: '+name)
                search=list(spec.submodule_search_locations)
        require(type(spec.origin) is str and Path(spec.origin).is_absolute(),
            'real absolute installed source origin: '+name)
        path=Path(spec.origin).resolve()
        require(path.is_file() and path.suffix=='.py','existing installed Python class source: '+name)
        sources[name]=dict(path=str(path),sha256=sha(path))
    require(len(sources)==8 and len({v['path'] for v in sources.values()})==8,
        'all eight distinct installed class sources, no skipping')
    return sources


def merge_class_sources(bindings):
    require(sha(V1_SOURCE)==V1_SOURCE_SHA,'sealed v1 source unchanged')
    resolved=installed_class_sources()
    additions={v['path']:v['sha256'] for v in resolved.values()}
    additions[str(V1_SOURCE)]=V1_SOURCE_SHA
    for path,h in additions.items():
        require(path not in bindings or bindings[path]==h,'installed/upstream binding conflict: '+path)
        bindings[path]=h
    return resolved


def build(parent_output=PARENT,parent_sha=PARENT_SHA):
    parent_output=Path(parent_output).resolve()
    require(parent_output==PARENT and parent_sha==PARENT_SHA,'fixed sealed allmode42 authority only')
    authority=source.consume(parent_output,parent_sha)
    r=source.overlay.read(parent_output/'report.json',source.CAP);p=source.verify(r['prepared'],r['prepare_sha256'])
    slots=material(p,r);bindings=dict(p['bindings'])
    bindings[str(parent_output/'report.json')]=parent_sha;bindings[str(Path(r['prepared']).resolve())]=r['prepare_sha256']
    for name,h in r['artifact_sha256'].items():bindings[str(parent_output/name)]=h
    for module in (source,transport,signal,native,core,gd,gd.screen):bindings[str(Path(module.__file__).resolve())]=sha(module.__file__)
    for f in own_files():bindings[str(f)]=sha(f)
    installed=merge_class_sources(bindings)
    check_bindings(bindings)
    return dict(schema=NAME+'.prepare',arguments=dict(parent_output=str(parent_output),parent_sha=parent_sha),
        contract=contract(),slots=slots,policy=p['policy'],expected_model_hash=p['expected_model_hash'],bindings=bindings,
        upstream_strict_status=authority['status'],full_upstream_rebuilt_here=True,original_compute=source.contract()['total'],
        planned_new_compute=caps(3376,1344),actual_new_compute=None,installed_class_sources=installed)


def check_bindings(bindings):
    for f,h in bindings.items():require(sha(f)==h,'complete unchanged input/source/model SHA: '+f)


def verify(prepared,expected):
    core.hash_value(expected);require(sha(prepared)==expected,'own prepare SHA')
    p=signal.read(prepared,16*1024**2)
    require(raw(p)==raw(build(**p['arguments'])) and sha(prepared)==expected,'whole strict original authority and own prepare reconstruction')
    return p


def write(path,value):
    path=Path(path).absolute();b=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve()==path and not path.exists()
        and len(b)<=64*1024**2 and shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(b),'exclusive bounded metadata')
    with path.open('xb') as f:require(f.write(b)==len(b),'full metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


class Store(transport.Store):
    def __init__(self,path,started):
        self.path=Path(path).absolute();self.finalizing=False;self.started=started
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve()==self.path
            and not self.path.exists() and shutil.disk_usage(self.path.parent).free>=contract()['output_bytes']+contract()['disk_floor_bytes'],
            'exclusive512MiB output plus200MiB floor')
        self.path.mkdir();signal.fsync_dir(self.path.parent)
    def check(self,size=4096):
        require(self.finalizing or time.monotonic()-self.started<contract()['wall_seconds'],'whole-stage1800s cap including verify; boundary watchdog')
        used=sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size<=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
            and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,'output/reserve/disk cap')
    def put(self,name,value,final=False):return self.write(name,raw(value),final=final)


class Backend:
    """Narrow H1 suffix scalar-gradient backend, no global AD patches."""
    def __init__(self,model,device,task,journal,ledger,check):
        require(getattr(device,'type',None)=='cuda','HOST CUDA only')
        import torch
        self.torch=torch;self.device=device;self.task=task;self.check=check;self.serial=0
        self.o=native.NativeProgram(model,device,task,journal);self.o.ledger=ledger
    @contextmanager
    def observe(self):
        with self.o.observe():yield self
    def tensor(self,v):return self.torch.tensor(v,device=self.device,dtype=self.torch.float32)
    def forward(self,c,a,label):
        t=self.torch;o=self.o;self.check()
        require(a.dtype==t.float32 and tuple(a.shape)==(5,2) and bool(t.isfinite(a).all()) and bool((a.abs()<=1).all()),'literal legal raw5')
        o.phase=f'f{self.serial:03}';self.serial+=1;o.step=0;o.ledger.scope=dict(forward=o.phase,label=label)
        o.store.event(dict(kind='forward_started',phase=o.phase,label=label,input=deepcopy(c),commands=a.detach().cpu().tolist()))
        z,h=(self.tensor(c[k])[None] for k in ('latent_history','action_history'))
        mean,std=(self.tensor(self.task[k]) for k in ('action_mean','action_std'))
        token=((a-mean)/std).reshape(10);outgoing=t.cat((h,token[None,None]),1)
        o.arrays('program_packet',latent=z,history=h,raw=a,mean=mean,std=std,token=token,outgoing=outgoing)
        emb=o.model.action_encoder(outgoing)
        predicted=o.call('predict',lambda:o.model.predict(z,emb),dict(latent=z,act_emb=emb),shape=list(z.shape))
        require(tuple(predicted.shape)==(1,3,192) and bool(t.isfinite(predicted).all()),'native3 full prediction ABI')
        z,h=t.cat((z[:,1:],predicted[:,-1:]),1),outgoing[:,1:]
        o.arrays('program_boundary',latent=z,history=h)
        after=dict(c,latent_history=z[0].detach().cpu().tolist(),action_history=h[0].detach().cpu().tolist())
        o.store.event(dict(kind='forward_finished',phase=o.phase,context=after));self.check();return z[0,-1],after
    def step(self,c,a,label):
        with self.torch.no_grad():return self.forward(c,self.tensor(a),label)[1]
    def gradient(self,c,a,label):
        t=self.torch;o=self.o
        with t.enable_grad():
            leaf=self.tensor(a).requires_grad_(True);y,after=self.forward(c,leaf,label)
            loss=((y-self.tensor(self.task['goal']))**2).mean()
            require(loss.ndim==0 and leaf.is_leaf and leaf.grad is None,'actual scalar objective/fresh suffix raw leaf')
            hook=leaf.register_hook(lambda v:o.ledger.call('raw_backward',lambda:o._raw_return(v)))
            e=dict(kind='AD_API_attempt',phase=o.phase,scalar_outputs=1,input_shape=[5,2],retain_graph=False,create_graph=False)
            o.store.event(e)
            try:
                g=o.ledger.call('vjp',lambda:o.call('grad',lambda:t.autograd.grad(loss,leaf,retain_graph=False,create_graph=False)[0],
                    dict(loss=loss,raw=leaf)),directions=1)
            finally:hook.remove()
            self.check();o.store.event(dict(e,kind='AD_API_completed'))
        return dict(context=after,native_float32_loss=float(loss.detach().cpu()),raw_gradient=g.detach().cpu().numpy().tolist())


def evaluate(slot,c,backend,journal):
    t=slot['task'];root=t['root_context'];prefix=np.asarray(c['prefix'],np.float32)
    require(bits(prefix)==c['prefix_bits'],'frozen prefix before dispatch')
    journal.event(dict(kind='candidate_intent',candidate_sha256=digest(c),slot_index=slot['index'],prefix_index=c['index']))
    mid=backend.step(root,prefix,'prefix')
    require(mid['latent_history'][-1]==c['original_predicted_prefix'],'literal original prefix context exact')
    x=np.asarray(c['logits'],float);a=chart(x);inc=backend.step(mid,a,'iterate_0')
    require(inc==c['original_terminal_context'] and cost(t['goal'],inc)==c['original_terminal_Q'],'original incumbent exact new model replay')
    its=[dict(iteration=0,logits=x.tolist(),commands=a.tolist(),raw_bits=bits(a),context=inc,Q=cost(t['goal'],inc))];grads=[]
    for it in range(1,9):
        d=backend.gradient(mid,a,f'gradient_{it}')
        require(d['context']==its[-1]['context'],'fresh raw-gradient forward same actual point')
        g=np.asarray(d['raw_gradient'],float);require(g.shape==(5,2) and np.isfinite(g).all(),'finite suffix scalar derivative')
        chain=1-a.astype(float)**2;dx=(g*chain).reshape(10)
        d.update(chain=chain.tolist(),logit_gradient=dx.tolist());grads.append(d)
        x=proposal(x,dx);a=chart(x);after=backend.step(mid,a,f'iterate_{it}')
        its.append(dict(iteration=it,logits=x.tolist(),commands=a.tolist(),raw_bits=bits(a),context=after,Q=cost(t['goal'],after)))
    winner=min(range(9),key=lambda k:(its[k]['Q'],k));commands=np.concatenate((prefix,np.asarray(its[winner]['commands'],np.float32)))
    journal.event(dict(kind='candidate_selection',iteration=winner,commands=commands.tolist()))
    first=backend.step(root,commands[:5],'selected_prefix');last=backend.step(first,commands[5:],'selected_suffix')
    require(first==mid and last==its[winner]['context'],'independent wholeH2 selected full context')
    result=dict(index=c['index'],candidate_sha256=digest(c),mid=mid,iterates=its,gradients=grads,winner_iteration=winner,
        commands=commands.tolist(),selected_Q=its[winner]['Q'],selected_context=last)
    journal.event(dict(kind='candidate_return',result_sha256=digest(result)));return result


def final_replay(slot,results,backend,journal):
    require(len(results)==42 and [r['index'] for r in results]==list(range(42)),'complete library before terminal selection')
    winner=min(results,key=lambda r:(r['selected_Q'],r['index']))
    journal.event(dict(kind='library_selection',prefix_index=winner['index'],commands=winner['commands'],Q=winner['selected_Q']))
    contexts=[]
    for mode in ('whole','segmented'):
        c=slot['task']['root_context']
        for k,a in enumerate(np.asarray(winner['commands'],np.float32).reshape(2,5,2)):c=backend.step(c,a,f'{mode}_{k}')
        contexts.append(c)
    require(contexts[0]==contexts[1]==winner['selected_context'],'final whole/segmented exact native replay')
    return dict(prefix_index=winner['index'],iterate_index=winner['winner_iteration'],commands=winner['commands'],
        Q=winner['selected_Q'],context=contexts[-1],original_generation_winner=slot['original_generation_winner'],
        selection_digest=digest([NAME,slot['task']['binding'],winner]),physical_release=False)


def audit_native(task,receipt,events,schedule):
    transport.audit_resources(receipt,events)
    n=len(schedule);reverse=sum(label.startswith('gradient_') for label,_,_ in schedule)
    require(receipt['caps']==caps(n,reverse) and receipt['attempted']==receipt['completed']==caps(n,reverse),'exact actual native forward/reverse units')
    fields={'resource_attempt':{'entry'},'resource_return':{'entry'},
        'forward_started':{'phase','label','input','commands'},'forward_finished':{'phase','context'},
        **{k:{'scope','phase','step','operation_index','arrays'} for k in ('program_packet','program_boundary','io_input','io_return','raw_backward')},
        **{k:{'phase','scalar_outputs','input_shape','retain_graph','create_graph'} for k in ('AD_API_attempt','AD_API_completed')},
        'candidate_intent':{'candidate_sha256','slot_index','prefix_index'},'candidate_selection':{'iteration','commands'},
        'candidate_return':{'result_sha256'},'library_selection':{'prefix_index','commands','Q'}}
    require(all(e['kind'] in fields and set(e)=={'kind'}|fields[e['kind']] for e in events),'unknown native event/fields rejected in full')
    starts=[e for e in events if e['kind']=='forward_started'];ends=[e for e in events if e['kind']=='forward_finished']
    require(len(starts)==len(ends)==n,'all native packet forwards')
    ad=[e for e in events if e['kind'].startswith('AD_API_')]
    require(len(ad)==2*reverse and all(e['phase'] in {f'f{i:03}' for i,(label,_,_) in enumerate(schedule)
        if label.startswith('gradient_')} for e in ad),'all reverse lifecycle events, no uncharged AD phase')
    phases={};positions={id(e):i for i,e in enumerate(events)};spans={}
    for i,(st,en,(label,c,a)) in enumerate(zip(starts,ends,schedule,strict=True)):
        phase=f'f{i:03}';phases[phase]=label
        require(st==dict(kind='forward_started',phase=phase,label=label,input=c,commands=a)
            and en['phase']==phase,'literal input/command/forward ordering')
        start,end=positions[id(st)],positions[id(en)]
        require(start<end and (i==0 or positions[id(ends[i-1])]<start),
            'nonoverlapping literal forward entry/return order')
        spans[phase]=(start,end,positions[id(starts[i+1])] if i+1<n else len(events))
        transport.audit_trajectory(dict(task,root_context=c),events,phase,1,a,[en['context']],program=True)
    # Full actual lifecycle: reverse shares its immediately preceding forward's
    # phase, but occurs AFTER forward_finished and BEFORE any next forward.
    # Every tensor/resource/lifecycle record is assigned to exactly one span.
    for e in events:
        kind=e['kind'];phase=e.get('phase');resource=None
        if kind.startswith('resource_'):
            scope=e['entry']['scope'];phase=scope['forward'];resource=e['entry']['resource']
            require(phase in phases and scope==dict(forward=phase,label=phases[phase]),'resource exact forward scope')
        elif kind in ('io_input','io_return','raw_backward'):
            idx=e['operation_index'];require(type(idx) is int and 0<=idx<len(receipt['events']),'literal linked operation')
            resource=receipt['events'][idx]['resource']
        if phase is None or kind in ('forward_started','forward_finished'):continue
        require(phase in spans,'all actual phase events assigned')
        if kind in ('program_packet','program_boundary'):
            require(e['scope']==dict(forward=phase,label=phases[phase]) and e['operation_index'] is None and e['step']==0,
                'packet/boundary exact actual scope and step')
        start,end,next_start=spans[phase];at=positions[id(e)]
        reverse_event=kind.startswith('AD_API_') or resource in ('grad','vjp','raw_backward')
        if reverse_event:
            require(phases[phase].startswith('gradient_') and end<at<next_start,'reverse after own forward end before next start')
        else:require(start<at<end,'every forward tensor/resource inside own start/end; no cross-forward splice')
    ops=receipt['events'];linked={}
    attempts={e['entry']['index']:i for i,e in enumerate(events) if e['kind']=='resource_attempt'}
    returns={e['entry']['index']:i for i,e in enumerate(events) if e['kind']=='resource_return'}
    for phase,label in phases.items():
        local=[op for op in ops if op['scope']['forward']==phase]
        require([op['resource'] for op in local]==['action_encoder','predict','predictor','pred_proj']+
            (['vjp','grad','raw_backward'] if label.startswith('gradient_') else []),'exact per-phase native operation schedule')
        packet=next(e for e in events if e['kind']=='program_packet' and e['phase']==phase)
        boundary=next(e for e in events if e['kind']=='program_boundary' and e['phase']==phase)
        require(positions[id(packet)]<attempts[local[0]['index']] and returns[local[1]['index']]<positions[id(boundary)],
            'packet before native entry; boundary after complete predict return')
        if label.startswith('gradient_'):
            lifecycle=[e for e in ad if e['phase']==phase]
            require(len(lifecycle)==2 and lifecycle[0]['kind']=='AD_API_attempt' and lifecycle[1]['kind']=='AD_API_completed'
                and positions[id(lifecycle[0])]<attempts[local[4]['index']]<returns[local[4]['index']]<positions[id(lifecycle[1])],
                'actual AD intent brackets entire VJP/grad/callback entry-return')
    for e in events:
        if e['kind'] not in ('io_input','io_return','raw_backward'):continue
        idx=e['operation_index'];require(type(idx) is int and 0<=idx<len(ops),'actual operation-linked IO')
        op=ops[idx];require(e['scope']==op['scope']==dict(forward=e['phase'],label=phases[e['phase']])
            and attempts[idx]<positions[id(e)]<returns[idx],'nested entry/IO/return exact scope')
        for b in e['arrays'].values():require(signal.unblob(b).dtype==np.float32,'actual float32 native IO')
        linked.setdefault(idx,[]).append(e)
    for op in ops:
        es=linked.get(op['index'],[])
        if op['resource']=='vjp':require(es==[],'inclusive VJP direction only');continue
        require([e['kind'] for e in es]==(['raw_backward'] if op['resource']=='raw_backward' else ['io_input','io_return']),'all native buffers present')
        if op['resource'] in ('action_encoder','predict','predictor','pred_proj'):
            key={'action_encoder':'0','predict':'latent','predictor':'0','pred_proj':'0'}[op['resource']]
            require(list(signal.unblob(es[0]['arrays'][op['resource']+':'+key]).shape)==op['shape'],'actual charged input shape')
        if op['resource']=='raw_backward':core.keys(es[0]['arrays'],('value',))
    return starts,ends,linked


def audit_candidate(slot,c,r,receipt,events):
    require(r['index']==c['index'] and r['candidate_sha256']==digest(c) and len(r['iterates'])==9 and len(r['gradients'])==8,'full9/8 denominator and immutable input')
    task=slot['task'];incumbent_domain(dict(suffix_params=c['logits'],suffix_solve=dict(params=c['logits'],radius=CAP,
        norm=c['incumbent_domain']['solver_float64_norm']),suffix_commands=c['incumbent_raw']))
    x=np.asarray(c['logits'],float);schedule=[('prefix',task['root_context'],c['prefix'])];mid=r['mid']
    require(mid['latent_history'][-1]==c['original_predicted_prefix'],'bound original prefix prediction')
    for it,v in enumerate(r['iterates']):
        a=chart(x)
        require(v['iteration']==it and v['logits']==x.tolist() and v['commands']==a.tolist() and v['raw_bits']==bits(a)
            and v['Q']==cost(task['goal'],v['context']),'every literal iterate/raw/nonlinearQ')
        core.checked_path(task,mid,a.tolist(),[v['context']])
        if it==0:require(v['context']==c['original_terminal_context'] and v['Q']==c['original_terminal_Q'],'original incumbent remains unmodified')
        schedule.append((f'iterate_{it}',mid,a.tolist()))
        if it<8:
            d=r['gradients'][it];g=core.finite(d['raw_gradient'],(5,2),True);chain=1-a.astype(float)**2;dx=(g*chain).reshape(10)
            require(d['context']==v['context'] and d['chain']==chain.tolist() and d['logit_gradient']==dx.tolist(),'original analytic chain once')
            schedule.append((f'gradient_{it+1}',mid,a.tolist()));x=proposal(x,dx)
            require(np.linalg.norm(x.astype(float))<=CAP,'new projected iterate strict inward feasible float32, not incumbent gate')
    win=min(range(9),key=lambda k:(r['iterates'][k]['Q'],k));v=r['iterates'][win]
    commands=c['prefix']+v['commands']
    require(r['winner_iteration']==win and r['commands']==commands and bits(r['commands'][:5])==c['prefix_bits']
        and r['selected_context']==v['context'] and r['selected_Q']==v['Q']<=r['iterates'][0]['Q'],'stable incumbent-retaining nonlinear selection')
    schedule.extend([('selected_prefix',task['root_context'],c['prefix']),('selected_suffix',mid,v['commands'])])
    starts,ends,linked=audit_native(task,receipt,events,schedule)
    require(ends[0]['context']==mid and all(en['context']==v['context'] for en,v in zip(ends[1:18:2],r['iterates'],strict=True))
        and ends[-2]['context']==mid and ends[-1]['context']==r['selected_context'],'saved forward endpoints match result')
    for i,d in enumerate(r['gradients']):
        phase=f'f{2+2*i:03}'
        require(ends[2+2*i]['context']==d['context'],'fresh actual gradient endpoint')
        ad=[e for e in events if e['kind'].startswith('AD_API_') and e['phase']==phase]
        expected=dict(phase=phase,scalar_outputs=1,input_shape=[5,2],retain_graph=False,create_graph=False)
        require(ad==[dict(kind='AD_API_attempt',**expected),dict(kind='AD_API_completed',**expected)],'exact scalar reverse API lifecycle')
        gradops=[o for o in receipt['events'] if o['resource']=='grad' and o['scope']['forward']==phase]
        rawops=[o for o in receipt['events'] if o['resource']=='raw_backward' and o['scope']['forward']==phase]
        require(len(gradops)==len(rawops)==1,'one actual grad and callback')
        ins,outs=linked[gradops[0]['index']];core.keys(ins['arrays'],('grad:loss','grad:raw'));core.keys(outs['arrays'],('grad:output',))
        loss,a,g=(signal.unblob(b) for b in (ins['arrays']['grad:loss'],ins['arrays']['grad:raw'],outs['arrays']['grad:output']))
        cb=signal.unblob(linked[rawops[0]['index']][0]['arrays']['value'])
        require(loss.shape==(1,) and a.shape==g.shape==cb.shape==(5,2) and float(loss[0])==d['native_float32_loss']
            and np.array_equal(a,r['iterates'][i]['commands']) and np.array_equal(g,d['raw_gradient']) and np.array_equal(cb,g),
            'source-bound runtime scalar serialized[1], exact suffix rawleaf and returned gradient/callback')
    intent=dict(kind='candidate_intent',candidate_sha256=digest(c),slot_index=slot['index'],prefix_index=c['index'])
    returned=dict(kind='candidate_return',result_sha256=digest(r))
    require(events[0]==intent and events[-1]==returned and [e for e in events if e['kind'] in
        ('candidate_intent','candidate_selection','candidate_return','library_selection')]==
        [intent,dict(kind='candidate_selection',iteration=win,commands=commands),returned], 'exact candidate durable bookends and selector events')
    sels=[e for e in events if e['kind']=='candidate_selection']
    require(sels==[dict(kind='candidate_selection',iteration=win,commands=commands)] and
        events.index(ends[-3])<events.index(sels[0])<events.index(starts[-2]),'selection after ALL9 iterates before independent H2 replay')
    return r


def roster():return [dict(slot=i,prefix_index=j,status='not_started',result=None,resource_receipt=None) for i in range(4) for j in range(42)]


def dispatch(p,store,rows,finals,check,factory):
    """One actual all168 dispatcher; explicit RAM factory is test-only, not CLI."""
    require(len(p['slots'])==4 and len(rows)==168 and not finals,'fixed4/168 initial dispatcher denominator')
    for s in p['slots']:
        require(s['index'] in range(4) and len(s['candidates'])==42,'all42 per original slot')
        results=[]
        for c in s['candidates']:
            row=rows[s['index']*42+c['index']];row['status']='attempted'
            ns=f's{s["index"]}_p{c["index"]:02}'
            journal=transport.NativeJournal(store,ns,check);ledger=core.Ledger(caps(20,8),sink=journal.event)
            try:
                backend=factory(s['task'],journal,ledger)
                with backend.observe():row['result']=evaluate(s,c,backend,journal)
                require(ledger.attempted==ledger.completed==caps(20,8),'exact per-candidate20forward/8reverse')
                row['status']='sealed';results.append(row['result'])
            finally:
                row['resource_receipt']=ledger.receipt();journal.close()
                row.update(journal_file=ns+'/events.jsonl',journal_sha256=sha(journal.path),journal_tail=journal.tail)
                if row['status']!='sealed':row['status']='failed_partial'
                store.put(ns+'_result.json',row)
        ns=f's{s["index"]}_final';journal=transport.NativeJournal(store,ns,check);ledger=core.Ledger(caps(4,0),sink=journal.event)
        final=dict(slot=s['index'],status='attempted',plan=None);finals.append(final)
        try:
            backend=factory(s['task'],journal,ledger)
            with backend.observe():final['plan']=final_replay(s,results,backend,journal)
            require(ledger.attempted==ledger.completed==caps(4,0),'final whole/segmented four forwards');final['status']='sealed'
        finally:
            if final['status']!='sealed':final['status']='failed_partial'
            final.update(resource_receipt=ledger.receipt());journal.close()
            final.update(journal_file=ns+'/events.jsonl',journal_sha256=sha(journal.path),journal_tail=journal.tail)


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA authorization before IO')
    started=time.monotonic();p=verify(prepared,expected);store=Store(output,started);rows=roster();finals=[];errors=[]
    model=None;before=after=host=None;flags=None;sources_ok=False
    def check():
        store.check()
        require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes']
            and torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'CUDA memory caps')
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32 and not torch.is_autocast_enabled('cuda'),'TF32/autocast off')
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);check()
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original frozen tensor identity')
        for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'presealed loaded model classes')
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            dispatch(p,store,rows,finals,check,lambda task,journal,ledger:Backend(model,device,task,journal,ledger,check))
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        try:
            if model is not None:
                source.overlay.moving.low.frozen(model);after=signal.model_hash(model);require(before==after,'model all buffers unchanged')
            check_bindings(p['bindings']);sources_ok=True
        except BaseException as exc:errors.append(dict(stage='final_closure',error=repr(exc)))
        wall=time.monotonic()-started
        if wall>=contract()['wall_seconds']:errors.append(dict(stage='wall',error='whole-stage cap'))
        peaks=None
        if model is not None:peaks=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        good=not errors and len(finals)==4 and all(r['status']=='sealed' for r in rows+finals) and sources_ok and before==after and before is not None
        artifacts={str(f.relative_to(store.path)):sha(f) for f in store.path.rglob('*') if f.is_file()}
        report=dict(schema=NAME+'.report',status='all168_nonlinear_suffix_sealed_not_admitted' if good else 'partial_failclosed',
            contract=contract(),prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,finals=finals,errors=errors,
            fixed_denominator=168,slot_denominator=4,known_roots=2,host=host,model_before=before,model_after=after,
            source_verified=sources_ok,source_manifest_sha256=digest(p['bindings']),wall_seconds=wall,CUDA_peak_bytes=peaks,
            artifact_sha256=artifacts,P1_admission=False,freshness=False,physical_release=False,scientific_admission=False)
        store.put('report.json',report,final=True)
    return dict(status=report['status'],output=str(store.path),report_sha256=sha(store.path/'report.json'),errors=errors)


def load_journal(output,row):
    path=output/row['journal_file']
    require(sha(path)==row['journal_sha256'],'actual native journal SHA')
    events,tail=signal.journal(path);require(tail==row['journal_tail'],'complete native hashchain');return events


def consume(output,expected):
    output=Path(output).absolute();require(sha(output/'report.json')==expected,'external actual report SHA')
    r=signal.read(output/'report.json',contract()['final_reserve_bytes']);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['contract']==contract() and len(r['rows'])==168 and r['fixed_denominator']==168
        and r['slot_denominator']==4 and r['known_roots']==2 and all(r[k] is False for k in ('P1_admission','freshness','physical_release','scientific_admission')),
        'new honest schema, full168/4 denominators, no admission')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution binding')
    actual={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
    require(actual==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in output.rglob('*') if f.is_file())<=contract()['output_bytes'],'complete artifact census/output cap')
    for name,h in r['artifact_sha256'].items():
        path=Path(name);require(not path.is_absolute() and all(v not in ('.','..') for v in path.parts)
            and (output/path).resolve()==output/path and sha(output/path)==h,'all native artifacts/path/SHA')
    receipts=[]
    for i,row in enumerate(r['rows']):
        require(row['slot']==i//42 and row['prefix_index']==i%42,'all fixed library positions, no dropped/repeated row')
        if row['resource_receipt'] is not None:
            events=load_journal(output,row);transport.audit_resources(row['resource_receipt'],events);receipts.append(row['resource_receipt'])
            require(signal.read(output/f's{i//42}_p{i%42:02}_result.json',4*1024**2)==row,'durable candidate row')
            if row['status']=='sealed':audit_candidate(p['slots'][i//42],p['slots'][i//42]['candidates'][i%42],row['result'],row['resource_receipt'],events)
    plans=[]
    for i,f in enumerate(r['finals']):
        require(i<4 and f['slot']==i,'fixed final selection slot order');es=load_journal(output,f);receipts.append(f['resource_receipt'])
        transport.audit_resources(f['resource_receipt'],es)
        if f['status']!='sealed':continue
        s=p['slots'][i];group=r['rows'][42*i:42*i+42];require(all(x['status']=='sealed' for x in group),'no partial library final selection')
        w=min((x['result'] for x in group),key=lambda x:(x['selected_Q'],x['index']));plan=f['plan'];commands=w['commands']
        require(plan==dict(prefix_index=w['index'],iterate_index=w['winner_iteration'],commands=commands,Q=w['selected_Q'],context=w['selected_context'],
            original_generation_winner=s['original_generation_winner'],selection_digest=digest([NAME,s['task']['binding'],w]),physical_release=False),'stable entire42 terminal selector')
        mid=group[w['index']]['result']['mid'];schedule=[(f'{mode}_{k}',s['task']['root_context'] if k==0 else mid,commands[5*k:5*k+5]) for mode in ('whole','segmented') for k in range(2)]
        _,ends,_=audit_native(s['task'],f['resource_receipt'],es,schedule)
        require(es[0]==dict(kind='library_selection',prefix_index=w['index'],commands=commands,Q=w['selected_Q'])
            and [e for e in es if e['kind'] in ('candidate_intent','candidate_selection','candidate_return','library_selection')]==[es[0]]
            and [e['context'] for e in ends]==[mid,w['selected_context'],mid,w['selected_context']],'actual final winner replay context')
        plans.append(dict(plan,root_slot=s['root_slot'],condition=s['condition'],task_id=s['task']['binding']['task_id'],planner_report_sha256=expected))
    counts={kind:{k:sum(q[kind][k] for q in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    complete=r['status']=='all168_nonlinear_suffix_sealed_not_admitted' and not r['errors'] and len(plans)==4
    if complete:
        require(counts['attempted']==counts['completed']==caps(3376,1344),'exact whole3376/1344 actual execution')
        require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash']
            and r['source_verified'] is True and r['source_manifest_sha256']==digest(p['bindings']),'complete frozen model/source closure')
        require(0<=r['wall_seconds']<contract()['wall_seconds'] and len(r['CUDA_peak_bytes'])==2
            and all(0<=x<=contract()[k] for x,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'whole wall/CUDA caps')
    check_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report seal')
    return dict(schema=NAME+'.consume',status='all168_verified_science_blocked' if complete else 'partial_failclosed',complete=complete,
        report_sha256=expected,fixed_denominator=168,slot_denominator=4,known_roots=2,plans=plans if complete else [],
        rows=r['rows'],actual_new_compute=counts,inherited_compute=p['original_compute'],FLOPs=None,
        P1_admission=False,freshness=False,physical_release=False,scientific_admission=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha','receipt','authorization'):q.add_argument('--'+k)
    a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build())
    elif a.command=='verify':v=dict(verified=True,schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:
        require(a.receipt,'own durable saved receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),flush=True)


if __name__=='__main__':main()
