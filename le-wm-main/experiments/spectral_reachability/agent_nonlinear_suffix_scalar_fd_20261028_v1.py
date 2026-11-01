"""Fixed iterate0 scalar-raw-gradient FD; saved preparation, MAIN HOST CUDA only."""
from __future__ import annotations
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p1_nonlinear_suffix_library_20261028_v2 as src

NAME='agent_nonlinear_suffix_scalar_fd_20261028_v1'
AUTH='MAIN_FIXED240_SCALAR_FD_HOST_CUDA_ONLY'
ROOT=src.ROOT
signal,core=src.signal,src.core
require,sha,digest,raw=src.require,src.sha,src.digest,src.raw
UP=ROOT/'results/agent_p1_nonlinear_suffix_library_20261028_v2_cuda_MAIN01'
UP_SHA='f91c44779de94e4abb87f4250b0db9b12f9e8cbe4f980628c031cb78301882de'
STRICT=ROOT/'results/agent_p1_nonlinear_suffix_library_20261028_v2_consume_MAIN01.json'
STRICT_SHA='337330eda5e8f191deec615422cae53e0c260364f63db0cd14d58218033ff019'
PREP=ROOT/'results/agent_p1_nonlinear_suffix_library_20261028_v2_prepare_MAIN01.json'
PREP_SHA='3da0299d73a865621f60b7bbd804a0f757044455dad30e562986635c1093eb4d'
SOURCE_SHA='0a3e475af9271a0769a3829b4e714fa4e2d6f00eaed38ec8c3ea5d7b9a163fbc'
STEPS=(.01,.001,.0001)


def contract():
    return dict(panels=4,known_roots=2,coordinates=10,steps=list(STEPS),signs=[-1,1],signed_requests=240,pairs=120,
        anchor='original_generation_winner.index ONLY; iterate0 commands, gradients[0].raw_gradient, predicted mid full3',
        forward='original v2 Backend.forward; full3; original input normalization; frozen native float32',
        objective='native float32 ((endpoint-goal)**2).mean(), identity192; NOT saved float64 selector Q',
        FD='float64(Qplus_native32-Qminus_native32)/(2*h); actual float32 spacing separate',
        missing='outside raw [-1,1] cube or collapsed float32 step: retained, never clamp/drop/replace',
        order='panel, h, coordinate, sign(-1,+1)',max_new_compute=src.caps(240,0),
        new_AD=0,new_encoder=0,new_physics=0,new_training=0,new_selection=0,retry=False,
        wall_seconds=1800,output_bytes=128*1024**2,final_reserve_bytes=16*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',FLOPs=None,equal_FLOPs=False,
        AD_correctness_certified=False,P1_admission=False,freshness=False,scientific_admission=False)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def packet(value,shape=(5,2)):
    a=np.asarray(value,np.float64)
    require(a.shape==shape and np.isfinite(a).all(),'finite exact-shape literal')
    b=a.astype(np.float32)
    require(np.isfinite(b).all() and np.array_equal(a,b.astype(np.float64)),'no lossy float32 literal repair')
    return b


def same(a,b):return raw(a)==raw(b)


def perturb(value,coordinate,sign,h,index):
    a=packet(value);require(type(coordinate) is int and 0<=coordinate<10 and sign in (-1,1) and h>0,'fixed coordinate/sign/step')
    b=a.copy().reshape(10);old=float(b[coordinate]);b[coordinate]=np.float32(old+sign*h)
    delta=float(b[coordinate])-old
    reason='outside_raw_cube' if np.any(abs(b)>1) else ('float32_step_collapsed' if delta==0 else None)
    return dict(index=index,label=f'h{STEPS.index(h)}_c{coordinate}_{"minus" if sign<0 else "plus"}' if h in STEPS else 'test_only',
        h=h,coordinate=coordinate,sign=sign,commands=b.reshape(5,2).tolist(),raw_bits=b.tobytes().hex(),
        nominal_delta=sign*h,actual_delta=delta,legal=reason is None,missing_reason=reason)


def requests(point):
    return [perturb(point['raw5'],c,s,h,i) for i,(h,c,s) in enumerate((h,c,s) for h in STEPS for c in range(10) for s in (-1,1))]


def summarize(point,schedule,values):
    require(same(schedule,requests(point)) and len(values)==60,'all fixed60 signed slots')
    g=packet(point['raw_gradient']).astype(float).reshape(10);steps=[]
    for k,h in enumerate(STEPS):
        pairs=[]
        for c in range(10):
            j=k*20+2*c;minus,plus=values[j:j+2];a,b=schedule[j:j+2]
            complete=a['legal'] and b['legal'] and minus is not None and plus is not None
            fd=(float(plus)-float(minus))/(2*h) if complete else None
            spacing=float(packet(b['commands']).flat[c])-float(packet(a['commands']).flat[c])
            secant=(float(plus)-float(minus))/spacing if complete else None
            pairs.append(dict(coordinate=c,minus_index=j,plus_index=j+1,minus_nativeQ=minus,plus_nativeQ=plus,
                nominal_spacing=2*h,actual_float32_spacing=spacing,nominal_FD=fd,actual_spacing_FD=secant,
                saved_scalar_raw_gradient=float(g[c]),signed_error=None if fd is None else fd-float(g[c]),
                abs_error=None if fd is None else abs(fd-float(g[c])),
                missing_reasons=[a['missing_reason'],b['missing_reason']],complete=bool(complete)))
        complete=all(x['complete'] for x in pairs);metrics=None
        if complete:
            v=np.asarray([x['nominal_FD'] for x in pairs]);err=v-g;ng=float(np.linalg.norm(g));nv=float(np.linalg.norm(v))
            metrics=dict(L2_error=float(np.linalg.norm(err)),Linf_error=float(abs(err).max()),
                relative_L2=None if ng==0 else float(np.linalg.norm(err))/ng,
                cosine=None if ng*nv==0 else float(v@g)/(ng*nv),gradient_L2=ng,FD_L2=nv)
        steps.append(dict(step_index=k,h=h,pairs=pairs,complete=complete,metrics_nominal=metrics))
    return dict(pair_denominator=30,signed_denominator=60,steps=steps,AD_correctness_certified=False,
        best_h_selected=False,pass_threshold=None)


def check_objective(arrays,endpoint,goal):
    core.keys(arrays,('endpoint','goal','residual','square','loss'))
    a={k:signal.unblob(v) for k,v in arrays.items()}
    require(all(v.dtype==np.float32 and np.isfinite(v).all() and v.shape==((1,) if k=='loss' else (192,)) for k,v in a.items()),'native objective float32 shapes')
    y,g=packet(endpoint,(192,)),packet(goal,(192,))
    def eq(x,z):return x.shape==z.shape and x.tobytes()==z.tobytes()
    require(eq(a['endpoint'],y) and eq(a['goal'],g) and eq(a['residual'],y-g)
        and eq(a['square'],a['residual']*a['residual']),'independent endpoint/goal/residual/square buffers')
    # A standard conservative floating reduction bound, not a fitted tolerance
    # or an FD acceptance gate. CUDA mean need not equal NumPy reduction bits.
    exact_square_mean=float(np.sum(a['square'],dtype=np.float64)/192)
    u=float(np.finfo(np.float32).eps)/2;gamma=193*u/(1-193*u)
    bound=gamma*exact_square_mean+193*float(np.nextafter(np.float32(0),np.float32(1)))
    q=float(a['loss'][0]);require(q>=0 and abs(q-exact_square_mean)<=bound,'native mean arithmetic reduction bound')
    return dict(native_float32_Q=q,float64_endpoint_Q=float(np.mean((y.astype(float)-g.astype(float))**2)),
        numpy_float32_mean=float(a['square'].mean()),square_mean_float64=exact_square_mean,
        native_reduction_difference=q-exact_square_mean,reduction_roundoff_bound=bound,bitwise_GPU_mean_replayed=False)


def new_row(index):
    return dict(index=index,status='not_started',evaluations=[],resource_receipt=None,native_journal=None,probe_journal=None)


def evaluate(point,schedule,backend,row,persist):
    require(same(schedule,requests(point)) and row['index']==point['index'] and row['status']=='not_started','fresh fixed point')
    row['evaluations']=[dict(index=r['index'],status='not_started',request=deepcopy(r),context=None,objective=None) for r in schedule]
    row['status']='running'
    for r,e in zip(schedule,row['evaluations'],strict=True):
        persist(dict(kind='probe_intent',point=point['index'],request=deepcopy(r),context_sha256=point['context_sha256']))
        if not r['legal']:
            e['status']='missing';persist(dict(kind='probe_missing',point=point['index'],index=r['index'],reason=r['missing_reason']));continue
        e['status']='started'
        after=backend.step(deepcopy(point['context']),deepcopy(r['commands']),r['label'])
        arrays=backend.objective(after)
        diagnostic=check_objective(arrays,after['latent_history'][-1],point['goal'])
        e.update(status='recorded',context=deepcopy(after),objective=dict(arrays=arrays,diagnostic=diagnostic))
        persist(dict(kind='probe_return',point=point['index'],evaluation=deepcopy(e)))
    row['status']='recorded'


def probe_events(point,schedule,row):
    out=[]
    for r,e in zip(schedule,row['evaluations'],strict=True):
        out.append(dict(kind='probe_intent',point=point['index'],request=deepcopy(r),context_sha256=point['context_sha256']))
        out.append(dict(kind='probe_return',point=point['index'],evaluation=deepcopy(e)) if r['legal'] else
            dict(kind='probe_missing',point=point['index'],index=r['index'],reason=r['missing_reason']))
    return out


def audit_native_bits(point,request,phase,events,end):
    local=[e for e in events if e.get('phase')==phase]
    def arr(kind,key):
        return signal.unblob(next(e['arrays'][key] for e in local if e['kind']==kind and key in e.get('arrays',{})))
    def eq(x,y):require(x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes(),'native full3 wiring bit identity '+phase)
    z=packet(point['context']['latent_history'],(3,192))[None]
    history=packet(point['context']['action_history'],(2,10))[None]
    a=packet(request['commands']);mean,std=(packet(point['task'][k],(2,)) for k in ('action_mean','action_std'))
    token=((a-mean)/std).reshape(10);outgoing=np.concatenate((history,token[None,None]),axis=1)
    for k,v in dict(latent=z,history=history,raw=a,mean=mean,std=std,token=token,outgoing=outgoing).items():eq(arr('program_packet',k),v)
    eq(arr('io_input','action_encoder:0'),outgoing)
    for key in ('predict:latent','predictor:0'):eq(arr('io_input',key),z)
    for key in ('predict:act_emb','predictor:1'):eq(arr('io_input',key),arr('io_return','action_encoder:output'))
    eq(arr('io_input','pred_proj:0'),arr('io_return','predictor:output').reshape(3,192))
    predicted=arr('io_return','predict:output');eq(predicted,arr('io_return','pred_proj:output').reshape(1,3,192))
    z=np.concatenate((z[:,1:],predicted[:,-1:]),axis=1)
    eq(arr('program_boundary','latent'),z);eq(arr('program_boundary','history'),outgoing[:,1:])
    eq(packet(end['context']['latent_history'],(3,192)),z[0])
    eq(packet(end['context']['action_history'],(2,10)),outgoing[0,1:])


def validate_point(point,schedule,row,receipt,native_events):
    require(same(schedule,requests(point)) and row['index']==point['index'] and row['status']=='recorded'
        and len(row['evaluations'])==60 and digest(point['context'])==point['context_sha256'],'fixed complete point')
    allowed={'forward_started','forward_finished','program_packet','program_boundary','io_input','io_return','resource_attempt','resource_return'}
    require(all(e['kind'] in allowed for e in native_events),'FD-only native event allowlist, no reverse or bookends')
    legal=[r for r in schedule if r['legal']]
    native_schedule=[(r['label'],point['context'],r['commands']) for r in legal]
    _,ends,_=src.audit_native(point['task'],receipt,native_events,native_schedule)
    starts=[e for e in native_events if e['kind']=='forward_started']
    for st,en,r in zip(starts,ends,legal,strict=True):
        require(same(st['input'],point['context']) and same(st['commands'],r['commands']),'bit-exact full3/context/raw incl signed zero')
        audit_native_bits(point,r,st['phase'],native_events,en)
    it=iter(ends);values=[]
    for r,e in zip(schedule,row['evaluations'],strict=True):
        require(e['index']==r['index'] and same(e['request'],r),'all literal request indexes/commands')
        if not r['legal']:
            require(e['status']=='missing' and e['context'] is None and e['objective'] is None,'missing retained, never substituted');values.append(None);continue
        end=next(it)
        require(e['status']=='recorded' and same(e['context'],end['context']),'same native terminal full context')
        obj=e['objective'];v=check_objective(obj['arrays'],end['context']['latent_history'][-1],point['goal'])
        require(same(v,obj['diagnostic']),'saved objective diagnostic recomputation');values.append(v['native_float32_Q'])
    return summarize(point,schedule,values)


def extract_points(p,r,output):
    points=[]
    require(len(p['slots'])==4 and len(r['rows'])==168,'full fixed upstream roster')
    for i,s in enumerate(p['slots']):
        j=s['original_generation_winner']['index'];require(type(j) is int and 0<=j<42,'original generation index')
        row=r['rows'][42*i+j];c=s['candidates'][j];res=row['result'];v=res['iterates'][0];g=res['gradients'][0]
        require(row['slot']==i and row['prefix_index']==j and row['status']=='sealed','original-winner fixed candidate')
        events=src.load_journal(output,row);src.audit_candidate(s,c,res,row['resource_receipt'],events)
        start=next(e for e in events if e['kind']=='forward_started' and e['phase']=='f002')
        require(start['label']=='gradient_1' and same(start['input'],res['mid']) and same(start['commands'],v['commands'])
            and same(g['context'],v['context']),'original gradient_1 is iterate0 at own predicted mid')
        ins=next(e for e in events if e['kind']=='io_input' and e.get('phase')=='f002' and 'grad:loss' in e['arrays'])
        outs=next(e for e in events if e['kind']=='io_return' and e.get('phase')=='f002' and 'grad:output' in e['arrays'])
        cb=next(e for e in events if e['kind']=='raw_backward' and e.get('phase')=='f002')
        aa=signal.unblob(ins['arrays']['grad:raw']);gg=signal.unblob(outs['arrays']['grad:output'])
        require(aa.tobytes()==packet(v['commands']).tobytes() and gg.tobytes()==packet(g['raw_gradient']).tobytes()
            and gg.tobytes()==signal.unblob(cb['arrays']['value']).tobytes(),'actual rawleaf, returned scalar gradient and callback exact bits')
        require(np.array_equal(np.asarray(s['task']['projection']),np.eye(192)),'original identity192, no fitted Q')
        point=dict(index=i,root_slot=s['root_slot'],condition=s['condition'],prefix_index=j,task=deepcopy(s['task']),
            context=deepcopy(res['mid']),context_sha256=digest(res['mid']),raw5=aa.tolist(),raw_gradient=gg.tolist(),
            goal=deepcopy(s['task']['goal']),anchor_native_float32_loss=g['native_float32_loss'],
            original_float64_endpoint_Q=v['Q'],original_prefix=c['prefix'],original_candidate_sha256=digest(c),
            authority=dict(report_sha256=UP_SHA,journal_file=row['journal_file'],journal_sha256=row['journal_sha256'],
                phase='f002',label='gradient_1',gradient_input_digest=digest(ins),gradient_output_digest=digest(outs),callback_digest=digest(cb)))
        packet(point['goal'],(192,));points.append(point)
    return points


def build():
    require(sha(src.__file__)==SOURCE_SHA and sha(UP/'report.json')==UP_SHA and sha(STRICT)==STRICT_SHA and sha(PREP)==PREP_SHA,'fixed original source/report/strict/prepare SHA')
    authority=src.consume(UP,UP_SHA)
    require(same(authority,signal.read(STRICT,64*1024**2)) and authority['complete'] is True,'entire original strict rebuilt, not status-only')
    r=signal.read(UP/'report.json',64*1024**2);p=signal.read(PREP,16*1024**2)
    require(r['prepared']==str(PREP) and r['prepare_sha256']==PREP_SHA,'original execution binding')
    points=extract_points(p,r,UP);bindings=dict(p['bindings'])
    bindings.update({str(UP/'report.json'):UP_SHA,str(STRICT):STRICT_SHA,str(PREP):PREP_SHA})
    for name,h in r['artifact_sha256'].items():bindings[str(UP/name)]=h
    for f in own_files():bindings[str(f)]=sha(f)
    installed=src.installed_class_sources();require(same(installed,p['installed_class_sources']),'same installed model class sources')
    src.check_bindings(bindings)
    return dict(schema=NAME+'.prepare',contract=contract(),points=points,schedules=[requests(x) for x in points],
        bindings=bindings,installed_class_sources=installed,policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        original_model_before=r['model_before'],original_model_after=r['model_after'],
        upstream_full_strict_sha256=STRICT_SHA,upstream_actual_compute=authority['actual_new_compute'],
        planned_signed_requests=240,planned_legal_forwards=sum(q['legal'] for x in points for q in requests(x)),
        actual_new_compute=None)


def verify(prepared,expected):
    core.hash_value(expected);require(sha(prepared)==expected,'own prepared SHA')
    p=signal.read(prepared,16*1024**2)
    require(same(p,build()) and sha(prepared)==expected,'full original strict, fixed points and schedules reconstruct exactly')
    return p


def write(path,value):
    path=Path(path).absolute();data=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve()==path and not path.exists()
        and len(data)<=16*1024**2 and shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(data),'exclusive bounded own metadata')
    with path.open('xb') as f:require(f.write(data)==len(data),'complete metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


class Store(src.transport.Store):
    def __init__(self,path,started):
        self.path=Path(path).absolute();self.started=started;self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve()==self.path
            and not self.path.exists() and shutil.disk_usage(self.path.parent).free>=contract()['output_bytes']+contract()['disk_floor_bytes'],'exclusive128MiB plus200MiB floor')
        self.path.mkdir();signal.fsync_dir(self.path.parent)
    def namespace(self,name):
        require(name in {f'p{i}_{kind}' for i in range(4) for kind in ('native','probe')},'fixed journal namespace')
        p=self.path/name;p.mkdir(exist_ok=False);signal.fsync_dir(self.path);return p
    def check(self,size=4096):
        require(self.finalizing or time.monotonic()-self.started<contract()['wall_seconds'],'whole-stage1800s wall cap')
        used=sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size<=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
            and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,'output reserve and disk cap')
    def put(self,name,value,final=False):return self.write(name,raw(value),final=final)


class Backend:
    """Composition exposes only original forward; scalar objective is CUDA, no AD."""
    def __init__(self,model,device,task,journal,ledger,check):
        self.inner=src.Backend(model,device,task,journal,ledger,check)
        self.torch=self.inner.torch;self.y=None
    def observe(self):return self.inner.observe()
    def step(self,c,a,label):
        with self.torch.no_grad():self.y,after=self.inner.forward(c,self.inner.tensor(a),label)
        return after
    def objective(self,after):
        t=self.torch
        require(self.y is not None and self.y.dtype==t.float32 and tuple(self.y.shape)==(192,),'actual native terminal tensor')
        require(self.y.detach().cpu().numpy().tobytes()==packet(after['latent_history'][-1],(192,)).tobytes(),'native endpoint bits before objective')
        with t.no_grad():
            goal=self.inner.tensor(self.inner.task['goal']);residual=self.y-goal;square=residual**2;loss=square.mean()
        require(not loss.requires_grad and loss.ndim==0,'native scalar float32 mean, no AD graph')
        return {k:signal.blob(v.detach().cpu().numpy()) for k,v in
            dict(endpoint=self.y,goal=goal,residual=residual,square=square,loss=loss.reshape(1)).items()}


def journal_ref(j,output):return dict(file=str(j.path.relative_to(output)),sha256=sha(j.path),tail=j.tail)


def dispatch(p,store,rows,check,factory):
    for point,schedule,row in zip(p['points'],p['schedules'],rows,strict=True):
        i=point['index'];nj=pj=ledger=None
        try:
            nj=src.transport.NativeJournal(store,f'p{i}_native',check)
            pj=src.transport.NativeJournal(store,f'p{i}_probe',check)
            ledger=core.Ledger(src.caps(sum(r['legal'] for r in schedule),0),sink=nj.event)
            backend=factory(point['task'],nj,ledger)
            with backend.observe():evaluate(point,schedule,backend,row,pj.event)
        finally:
            if ledger is not None:row['resource_receipt']=ledger.receipt()
            for j,key in ((nj,'native_journal'),(pj,'probe_journal')):
                if j is not None:j.close();row[key]=journal_ref(j,store.path)
            store.put(f'p{i}_row.json',row,final=True)


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA authorization')
    started=time.monotonic();p=verify(prepared,expected);store=Store(output,started)
    rows=[new_row(i) for i in range(4)];errors=[];model=None;before=after=host=flags=None;sources_ok=False
    def check():
        store.check()
        require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'CUDA memory engineering caps')
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32 and not torch.is_autocast_enabled('cuda'),'no TF32 or autocast')
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
        before=signal.model_hash(model)
        require(before==p['original_model_after']==p['original_model_before'] and before['original_state_dict']==p['expected_model_hash'],'same original model all buffers')
        for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'bound installed model class source')
        with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            dispatch(p,store,rows,check,lambda task,j,l:Backend(model,device,task,j,l,check))
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        try:
            if model is not None:
                src.source.overlay.moving.low.frozen(model);after=signal.model_hash(model);require(before==after,'all model tensors unchanged')
            src.check_bindings(p['bindings']);sources_ok=True
        except BaseException as exc:errors.append(dict(stage='final_closure',error=repr(exc)))
        wall=time.monotonic()-started
        if wall>=contract()['wall_seconds']:errors.append(dict(stage='wall',error='whole-stage wall cap'))
        peaks=None if model is None else [torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        good=not errors and all(r['status']=='recorded' for r in rows) and before==after and before is not None and sources_ok
        artifacts={str(f.relative_to(store.path)):sha(f) for f in store.path.rglob('*') if f.is_file()}
        report=dict(schema=NAME+'.report',status='fixed240_recorded_not_certified' if good else 'partial_failclosed',
            contract=contract(),prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,
            fixed_signed_denominator=240,fixed_pair_denominator=120,host=host,model_before=before,model_after=after,
            source_verified=sources_ok,source_manifest_sha256=digest(p['bindings']),wall_seconds=wall,CUDA_peak_bytes=peaks,
            artifact_sha256=artifacts,AD_correctness_certified=False,P1_admission=False,scientific_admission=False)
        store.put('report.json',report,final=True)
    return dict(output=str(store.path),report_sha256=sha(store.path/'report.json'),status=report['status'],errors=errors)


def load_journal(output,ref):
    require(set(ref)=={'file','sha256','tail'} and sha(output/ref['file'])==ref['sha256'],'own full journal SHA')
    es,tail=signal.journal(output/ref['file']);require(tail==ref['tail'],'own complete journal hashchain');return es


def consume(output,expected):
    output=Path(output).absolute();require(output.resolve()==output and sha(output/'report.json')==expected,'actual report SHA/path')
    r=signal.read(output/'report.json',16*1024**2);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and same(r['contract'],contract()) and len(r['rows'])==4
        and r['fixed_signed_denominator']==240 and r['fixed_pair_denominator']==120
        and all(r[k] is False for k in ('AD_correctness_certified','P1_admission','scientific_admission')),'full denominators and no admission')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'own execution binding')
    actual={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
    require(actual==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in output.rglob('*') if f.is_file())<=contract()['output_bytes'],'whole artifact census/cap')
    for name,h in r['artifact_sha256'].items():
        f=Path(name);require(not f.is_absolute() and all(v not in ('.','..') for v in f.parts) and (output/f).resolve()==output/f and sha(output/f)==h,'all artifact paths and SHA')
    panels=[];receipts=[];complete=r['status']=='fixed240_recorded_not_certified' and not r['errors']
    for i,(point,schedule,row) in enumerate(zip(p['points'],p['schedules'],r['rows'],strict=True)):
        require(row['index']==i,'fixed4 panel order')
        if row['resource_receipt'] is None:
            require(row==new_row(i),'only untouched remaining points');complete=False;continue
        require(same(signal.read(output/f'p{i}_row.json',4*1024**2),row),'durable row')
        require(row['native_journal']['file']==f'p{i}_native/events.jsonl' and row['probe_journal']['file']==f'p{i}_probe/events.jsonl','fixed journal routing')
        es=load_journal(output,row['native_journal']);ps=load_journal(output,row['probe_journal'])
        src.transport.audit_resources(row['resource_receipt'],es);receipts.append(row['resource_receipt'])
        if row['status']!='recorded':complete=False;continue
        summary=validate_point(point,schedule,row,row['resource_receipt'],es)
        require(same(ps,probe_events(point,schedule,row)),'all exact chronological probe intent/return/missing events')
        panels.append(dict(index=i,prefix_index=point['prefix_index'],root_slot=point['root_slot'],condition=point['condition'],
            context_sha256=point['context_sha256'],raw_bits=src.bits(point['raw5']),summary=summary))
    counts={kind:{k:sum(q[kind][k] for q in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    if complete:
        require(len(panels)==4 and counts['attempted']==counts['completed']==src.caps(p['planned_legal_forwards'],0),'exact legal forward counts; no reverse')
        require(r['model_before']==r['model_after']==p['original_model_after'] and r['source_verified'] is True
            and r['source_manifest_sha256']==digest(p['bindings']),'same original and before/after full model/source identity')
        require(0<=r['wall_seconds']<contract()['wall_seconds'] and len(r['CUDA_peak_bytes'])==2
            and all(0<=v<=contract()[k] for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'actual wall/CUDA caps')
    src.check_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report seal')
    return dict(schema=NAME+'.consume',status='fixed240_verified_not_certified' if complete else 'partial_failclosed',complete=complete,
        report_sha256=expected,signed_denominator=240,pair_denominator=120,panel_denominator=4,panels=panels,
        actual_new_compute=counts,planned_signed_requests=240,planned_legal_forwards=p['planned_legal_forwards'],
        FLOPs=None,AD_correctness_certified=False,P1_admission=False,scientific_admission=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha','receipt','authorization'):q.add_argument('--'+k)
    a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build())
    elif a.command=='verify':v=dict(verified=True,schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:
        require(a.receipt,'durable own saved receipt path');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),flush=True)


if __name__=='__main__':main()
