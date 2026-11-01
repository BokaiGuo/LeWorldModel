"""Fixed eight A-history native predictions; MAIN authorized HOST CUDA only.

Pinned MAIN compat authority, no extra R replay, encoder, AD, physics or fitting.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect as pyinspect
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import main_p3_history_preconditions_compat_20261029_v1 as compat
from experiments.spectral_reachability import agent_p3_native_program_20261028_v1 as native

resolver, saved = compat.resolver, compat.resolver.saved
signal, core = native.signal, native.core
require, canonical, digest = resolver.require, resolver.canonical, resolver.digest
ROOT = resolver.ROOT
NAME = 'agent_p3_same_action_actual_history_cuda_20261029_v1'
AUTH = 'MAIN_SAME_ACTION_ACTUAL_HISTORY8_HOST_CUDA_ONLY'
PRECONDITIONS_SHA = 'd51467dabc0fb2625016ff21229f33b04fde93b966e4a5c8e77408d9cef675c8'
COMPAT_SHA = 'f909d207b880d0fc5e4dc23b1f4f378f99b68f932edc6c26a6f4858be6be3ba1'
CAP, FINAL, FLOOR, WALL, META = 64*1024**2, 8*1024**2, 200*1024**2, 600, 128*1024**2
REPORT_CAP = 8*1024**2


def file_sha(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and path.is_file() and
            not any(q.is_symlink() for q in (path,*path.parents)), 'canonical regular bound file')
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(1024**2):h.update(data)
    return h.hexdigest()


def read(path,h,cap=META):return saved.decode(saved.checked_bytes(path,h,cap))


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def caps():
    c=dict.fromkeys(core.AXES,0)
    c.update(predict_API=8,predict_samples=8,predict_context_tokens=24,
        action_encoder_calls=8,action_encoder_samples=8,action_encoder_context_tokens=24,
        predictor_calls=8,predictor_samples=8,predictor_context_tokens=24,
        pred_proj_calls=8,pred_proj_rows=24)
    return c


def contract():
    return dict(schema=NAME+'.contract',fixed_denominator=8,known_roots=2,fresh_roots=0,
        preconditions_schema=compat.NAME+'.prepare',preconditions_sha256=PRECONDITIONS_SHA,
        source_receipt='MAIN complete compat eight R IO and A history; not rebuilt here',
        starts=[10,15],ends=[15,20],R_new_calls=0,A_predict=8,batch=1,input_token_frames=24,
        resource_caps=caps(),raw_normalization='original float32 subtract/divide once; no clip',
        native_tensor_bytes=206720,encoder=0,projector=0,AD=0,physics=0,training=False,
        no_retry=True,no_refund=True,no_dedup=True,full8=True,labels_after_model_seal=True,
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        metadata_cap_bytes=META,cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        TF32=False,autocast=False,SDP='MATH',total_FLOPs=None,equal_total_compute=False,
        P1_admission=False,P3_admission=False,C3=False,physical_release=False)


def verify_bindings(bindings,sizes=None):
    for f,h in bindings.items():
        require(file_sha(f)==h,'complete bound file SHA: '+f)
        if sizes is not None:require(Path(f).stat().st_size==sizes[f],'bound file size: '+f)


def material(cp,pp):
    require(cp['schema']==compat.NAME+'.prepare' and cp['contract']==resolver.contract() and
            cp['status']=='all8_saved_preconditions_ready_NOT_execution' and
            cp['ready']==8 and cp['blocked']==0 and cp['fixed_denominator']==8,
            'actual complete compat authority, no ready subset')
    require(cp['compatibility']==dict(unchanged_resolver_sha256=compat.PIN,
            planner_report_cap=256*1024**2,other_input_caps_unchanged=True,
            R_IO_math_resource_checks_unchanged=True,original_shared_panel_report_reserve_bytes=256*1024**2),
            'only reviewed planner report input ABI compatibility')
    require(pp['schema']==resolver.PLANNER+'.prepare' and len(pp['tasks'])==2,
            'original model/task preparation')
    require(all(cp[k] is False for k in ('P1_admission','P3_admission','C3','physical_release')),
            'engineering authority, no scientific admission')
    require(len(cp['jobs'])==8,'complete eight fixed jobs')
    jobs=[]
    for i,j in enumerate(cp['jobs']):
        b,k=divmod(i,2);slot=b//2;bank=('endpoints','midpoints')[b%2];start=(10,15)[k]
        require(type(j['index']) is int and type(j['plan_index']) is int and type(j['root_slot']) is int
                and type(j['start']) is int and type(j['end']) is int and
                (j['index'],j['plan_index'],j['root_slot'],j['bank'],j['start'],j['end'])==
                (i,b,slot,bank,start,start+5) and j['status']=='ready' and j['blocked_reasons']==[]
                and j['R_reuse_verified'] is True and j['new_R_calls']==0 and j['new_A_calls_cap']==1,
                'fixed authenticated8 R eligibility')
        base=core.validate_task(pp['tasks'][slot]);task=deepcopy(base);task['root_context']=deepcopy(j['A_context'])
        state,hist=resolver.context(task['root_context']);_,rh=resolver.context(j['R_context'])
        resolver.same(hist,rh,'A/R action history byte identity')
        require(j['A_context_sha256']==digest(task['root_context']) and j['model_hash']==pp['expected_model_hash']
                and j['task_binding']==base['binding'] and canonical(j['goal192'])==canonical(base['goal']),
                'original frozen model/context/goal identity')
        require(j['normalizer']==dict(mean=base['action_mean'],std=base['action_std'],dtype='float32',applications=1),
                'original once-only normalizer')
        raw=resolver.f32(j['future_raw5'],(5,2));history=resolver.f32(j['causal_raw10'],(10,2))
        full=resolver.f32(j['full20'],(20,2))
        require(np.abs(raw).max()<=1 and raw.tobytes().hex()==j['future_raw5_bits'] and
                history.tobytes().hex()==j['causal_raw10_bits'] and full.tobytes().hex()==j['full20_rawbits'],
                'fixed same action rawbits')
        resolver.same(raw,full[start:start+5],'literal selected future suffix')
        resolver.same(hist,resolver.normalization(base,history),'original actual history tokens')
        token=((raw-resolver.f32(base['action_mean'],(2,)))/resolver.f32(base['action_std'],(2,))).reshape(10)
        out=np.concatenate((hist,token[None]),axis=0)[None]
        require(canonical(j['A_predict_input'])==canonical(dict(latent=state[None].tolist(),
                outgoing_actions=out.tolist(),raw5=raw.tolist())),'full A native input recipe')
        local_identity=dict(schema=NAME+'.actual_context_binding',index=i,start=start,end=start+5,
            selection_digest=j['selection_digest'],context_sha256=digest(task['root_context']))
        task['binding'].update(context_sha256=digest(task['root_context']),
            task_id=digest([base['binding']['task_id'],local_identity]),
            task_sha256=digest([base,task['root_context'],local_identity]),
            task_contract_sha256=digest([NAME,local_identity]))
        core.validate_task(task)
        # No endpoint labels or R outputs are passed to the neural callback.
        jobs.append(dict(index=i,plan_index=b,root_slot=slot,bank=bank,start=start,end=start+5,
            task_id=j['task_id'],selection_digest=j['selection_digest'],witness_sha256=j['witness_sha256'],
            task=task,original_task_binding=deepcopy(base['binding']),local_identity=local_identity,
            raw5=raw.tolist(),raw5_bits=raw.tobytes().hex(),context_sha256=digest(task['root_context'])))
    return jobs


def build(preconditions,preconditions_sha):
    require(preconditions_sha==PRECONDITIONS_SHA,'pinned MAIN actual complete compat SHA; no substitute')
    require(file_sha(Path(compat.__file__).resolve())==COMPAT_SHA,'frozen MAIN compat source')
    cp=read(preconditions,preconditions_sha)
    args=cp['arguments']
    pp=read(args['planner_prepared'],args['planner_prepared_sha'])
    ep=read(args['encoder_prepared'],args['encoder_prepared_sha'])
    jobs=material(cp,pp)
    bindings={}
    def merge(items):
        for f,h in items.items():
            require(f not in bindings or bindings[f]==h,'nonconflicting authority closure')
            bindings[f]=h
    merge(cp['bindings']);merge(pp['sources']);merge(ep['bindings'])
    merge({str(Path(preconditions).resolve()):preconditions_sha,
           str(Path(compat.__file__).resolve()):COMPAT_SHA})
    for f in own_files()+[Path(native.__file__).resolve(),Path(signal.__file__).resolve(),Path(core.__file__).resolve(),
                         ROOT/'experiments/checkpoint_compat.py',ROOT/'experiments/gpu_preflight.py']:
        merge({str(f):file_sha(f)})
    require(all(bindings[str(Path(f).resolve())]==h for f,h in resolver.PINS.items()), 'original pinned sources closure')
    policy=Path(pp['policy']).resolve()
    require(str(policy/'weights.pt') in bindings and str(policy/'config.json') in bindings,
            'original checkpoint and config in immutable closure')
    verify_bindings(bindings);sizes={f:Path(f).stat().st_size for f in bindings}
    return dict(schema=NAME+'.prepare',arguments=dict(preconditions=str(Path(preconditions).resolve()),
        preconditions_sha=preconditions_sha),contract=contract(),jobs=jobs,bindings=bindings,binding_sizes=sizes,
        policy=str(policy),expected_model_hash=pp['expected_model_hash'],inherited_resources=cp['inherited_resources'],
        inherited_authority=dict(compat_schema=cp['schema'],compat_sha256=preconditions_sha,
            full_upstream_strict_rebuilt_here=False,new_R_calls=0),P1_admission=False,P3_admission=False,
        C3=False,physical_release=False)


def write_metadata(path,value):
    path=Path(path).absolute();data=canonical(value)+b'\n'
    require(path.name.startswith(NAME+'_') and not path.is_symlink() and len(data)<=META and
            shutil.disk_usage(path.parent).free>=FLOOR+len(data),'exclusive bounded metadata')
    with path.open('xb') as f:
        require(f.write(data)==len(data),'full metadata');f.flush();os.fsync(f.fileno())
    syncdir(path.parent)
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest())


def syncdir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def verify(path,expected):
    p=read(path,expected)
    require(p['schema']==NAME+'.prepare' and p['contract']==contract(),'own prepare schema and contract')
    # Rebuild own saved material only, never compat.build or upstream consume.
    require(canonical(p)==canonical(build(**p['arguments'])),'entire own prepare reconstruction')
    return p


class Store:
    """Exclusive durable bounded journal+metadata; failed finalization bypasses wall only."""
    def __init__(self,output,started,clock=time.monotonic,free=None):
        self.path=Path(output).absolute();require(self.path.name.startswith(NAME+'_') and
            not self.path.exists() and not self.path.is_symlink(),'new own output directory')
        self.clock=clock;self.started=started;self.free=free or (lambda:shutil.disk_usage(self.path).free)
        self.used=0;self.index=0;self.tail='0'*64;self.files={};self.finalizing=False
        self.path.mkdir();syncdir(self.path.parent)
        with (self.path/'events.jsonl').open('xb') as f:f.flush();os.fsync(f.fileno())
        self.files['events.jsonl']=None;syncdir(self.path)

    def check(self,size=0,final=False):
        final=final or self.finalizing
        require(self.used+size<=CAP-(0 if final else FINAL),'whole output bytes/final reserve')
        require(self.free()>=FLOOR+size,'disk floor')
        if not final:require(self.clock()-self.started<=WALL,'whole stage wall budget')

    def event(self,payload):
        body=dict(index=self.index,previous=self.tail,payload=deepcopy(payload))
        h=digest(body);data=canonical(dict(body,sha256=h))+b'\n'
        require(len(data)<=resolver.LINE_CAP,'bounded event')
        metadata=deepcopy(payload)
        if 'arrays' in metadata:
            for b in metadata['arrays'].values():b.pop('data',None)
        require(len(canonical(metadata))<=64*1024,'per-event metadata bound')
        self.check(len(data))
        with (self.path/'events.jsonl').open('ab') as f:
            require(f.write(data)==len(data),'full journal append');f.flush();os.fsync(f.fileno())
        self.used+=len(data);self.index+=1;self.tail=h

    def put(self,name,value,final=False):
        require(type(name) is str and Path(name).name==name and name not in self.files,'exclusive flat artifact')
        data=canonical(value)+b'\n';self.check(len(data),final)
        with (self.path/name).open('xb') as f:
            require(f.write(data)==len(data),'full saved file');f.flush();os.fsync(f.fileno())
        self.used+=len(data);self.files[name]=hashlib.sha256(data).hexdigest();syncdir(self.path)
        return self.files[name]

    def census(self):
        require({f.name for f in self.path.iterdir()}==set(self.files),'whole output artifact census')
        require(sum(f.stat().st_size for f in self.path.iterdir())==self.used,'whole file bytes')
        return {n:file_sha(self.path/n) for n in self.files}


def precision(torch):
    return dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
        autocast=torch.is_autocast_enabled('cuda'),math=torch.backends.cuda.math_sdp_enabled(),
        flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())


def precision_contract():
    return dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)


def inventory(model,bindings):
    out=[]
    for module in model.modules():
        path=str(Path(pyinspect.getfile(type(module))).resolve())
        require(path in bindings and file_sha(path)==bindings[path],'presealed loaded model class source')
        out.append(dict(type=type(module).__module__+'.'+type(module).__qualname__,path=path,sha256=bindings[path]))
    require(out,'complete module source inventory')
    return out


class NoAD:
    """Temporary runtime-only forbidden API wrappers, restored in finally; no old globals patched."""
    def __init__(self,torch,sink):self.torch=torch;self.sink=sink;self.saved=[];self.events=[]
    def __enter__(self):
        t=self.torch
        try:
            for owner,name in [(t.autograd,'grad'),(t.autograd,'backward'),(t.Tensor,'backward'),
                (t.autograd.functional,'jacobian'),(t.autograd.functional,'hessian'),
                (t.func,'jacrev'),(t.func,'jacfwd'),(t.func,'grad'),(t.func,'hessian'),
                (t.linalg,'svd'),(t.linalg,'solve')]:
                original=getattr(owner,name)
                def forbidden(*args,_name=name,**kwargs):
                    e=dict(kind='forbidden_API',resource=_name);self.events.append(e);self.sink(e)
                    raise ValueError('zero AD/solve/SVD budget: '+_name)
                self.saved.append((owner,name,original));setattr(owner,name,forbidden)
        except BaseException:
            self.__exit__();raise
        return self
    def __exit__(self,*exc):
        for owner,name,fn in reversed(self.saved):setattr(owner,name,fn)


def execute(p,store,ledger,factory,rows,check):
    """Same fixed denominator loop for HOST native or explicit test callbacks; no labels passed."""
    require(len(p['jobs'])==len(rows)==8,'all8 execution denominator')
    for job,row in zip(p['jobs'],rows,strict=True):
        check();require(row['index']==job['index'] and row['status']=='not_started','once-only fixed job')
        row['status']='attempted';ledger.scope=dict(job=job['index'],phase='A')
        store.event(dict(kind='A_intent',index=job['index'],job_sha256=digest(job)))
        try:
            path=factory(job).replay(job['task']['root_context'],job['raw5'],False,ledger)
            require(len(path)==1,'exact one native A step')
            resolver.context(path[0]);row.update(status='returned',after=path[0])
            store.event(dict(kind='A_return',index=job['index'],after=path[0]))
        except BaseException as exc:
            row.update(status='failed',error=type(exc).__name__+': '+str(exc));raise


def score(p,rows):
    """Called only after durable prediction/model-after seal; no selection or fitting."""
    cp=read(p['arguments']['preconditions'],PRECONDITIONS_SHA);out=[]
    for job,row,old in zip(p['jobs'],rows,cp['jobs'],strict=True):
        require(row['status']=='returned','all8 completed before scoring')
        actual=old['label192'];goal=old['goal192']
        A=saved.decompose_point(row['after']['latent_history'][-1],actual,goal)
        R=saved.decompose_point(old['R_proof']['predicted192'],actual,goal)
        out.append(dict(index=job['index'],plan_index=job['plan_index'],root_slot=job['root_slot'],
            bank=job['bank'],start=job['start'],end=job['end'],A=A,R=R,
            A_minus_R_MSE=A['error_mse']-R['error_mse'],
            A_minus_R_absolute_Q_error=A['absolute_Q_error']-R['absolute_Q_error'],
            A_minus_R_signed_Q_error=A['model_minus_actual_Q']-R['model_minus_actual_Q'],
            same_physical_label=True,physical_benefit_claim=False))
    return out


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA authorization before run I/O')
    started=time.monotonic();p=verify(prepared,expected)
    require(time.monotonic()-started<WALL,'verification charged to whole stage wall')
    store=Store(output,started);rows=[dict(index=i,status='not_started',after=None,error=None) for i in range(8)]
    ledger=core.Ledger(caps(),sink=store.event);errors=[];model=None;before=after=None
    sources_before=sources_after=None;numeric_before=numeric_after=None;peak=None;forbidden=[];seal=None;scores=None;guard=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    store.put('initial_jobs.json',p['jobs'])
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original model state')
        sources_before=inventory(model,p['bindings'])
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            def check():
                store.check();require(precision(torch)==precision_contract(),'real native precision')
                require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and
                        torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'real CUDA memory caps')
                require(not torch.is_grad_enabled(),'zero AD inference context')
            numeric_before=precision(torch);check()
            guard=NoAD(torch,store.event)
            with guard:
                execute(p,store,ledger,lambda j:native.NativeProgram(model,device,j['task'],store),rows,check)
            forbidden=guard.events
            torch.cuda.synchronize(device);check()
            after=signal.model_hash(model);sources_after=inventory(model,p['bindings']);numeric_after=precision(torch)
            peak=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
            require(before==after and sources_before==sources_after and numeric_before==numeric_after==precision_contract()
                    and not forbidden and ledger.attempted==ledger.completed==caps(),'whole eight original frozen model/cost')
            verify_bindings(p['bindings'],p['binding_sizes']);check()
            seal=store.put('prediction_seal.json',dict(rows=rows,resource_receipt=ledger.receipt(),
                model_before=before,model_after=after,inventory_before=sources_before,inventory_after=sources_after,
                precision_before=numeric_before,precision_after=numeric_after,forbidden=forbidden))
            store.event(dict(kind='predictions_sealed',sha256=seal))
            store.event(dict(kind='labels_unlocked',preconditions_sha256=PRECONDITIONS_SHA))
            scores=score(p,rows);store.put('scores.json',scores)
            verify_bindings(p['bindings'],p['binding_sizes']);check()
    except BaseException as exc:errors.append(dict(stage='run',error=type(exc).__name__+': '+str(exc)))
    finally:
        if guard is not None:forbidden=deepcopy(guard.events)
        elapsed=time.monotonic()-started
        if elapsed>WALL:errors.append(dict(stage='wall',error='whole stage wall exceeded'))
        complete=not errors and scores is not None and seal is not None and all(r['status']=='returned' for r in rows)
        store.finalizing=True
        report=dict(schema=NAME+'.report',status='all8_A_recorded_NOT_admitted' if complete else 'partial_or_failed',
            contract=contract(),prepared=str(Path(prepared).resolve()),prepare_sha256=expected,rows=rows,
            errors=errors,resource_receipt=ledger.receipt(),forbidden=forbidden,model_before=before,model_after=after,
            inventory_before=sources_before,inventory_after=sources_after,precision_before=numeric_before,
            precision_after=numeric_after,CUDA_peak_bytes=peak,elapsed_seconds=elapsed,prediction_seal_sha256=seal,
            scores_sha256=None if scores is None else store.files.get('scores.json'),
            P1_admission=False,P3_admission=False,C3=False,physical_release=False)
        store.event(dict(kind='run_finished',status=report['status']))
        report.update(event_count=store.index,journal_tail=store.tail,artifact_sha256=store.census())
        store.put('report.json',report,final=True)
    return dict(output=str(store.path),report_sha256=file_sha(store.path/'report.json'),status=report['status'])


def audit_A(job,row,events,operations):
    """Independent exact byte checking of actual native lifecycle for one A step."""
    require(len(events)==10 and row['status']=='returned','all10 native A events')
    groups=[];scope=dict(job=job['index'],phase='A')
    for e,(kind,keys) in zip(events,resolver.SEQUENCE,strict=True):
        require(set(e)=={'kind','scope','phase','step','operation_index','arrays'} and e['kind']==kind
                and e['scope']==scope and e['phase']=='whole' and type(e['step']) is int and e['step']==0
                and set(e['arrays'])==set(keys),'exact native A lifecycle schema')
        groups.append({k:resolver.unblob(e['arrays'][k]) for k in keys})
    packet,ae_in,ae_out,pr_in,inner_in,inner_out,proj_in,proj_out,pr_out,boundary=groups
    state,hist=resolver.context(job['task']['root_context']);task=job['task']
    raw=resolver.f32(job['raw5'],(5,2));mean=resolver.f32(task['action_mean'],(2,));std=resolver.f32(task['action_std'],(2,))
    token=((raw-mean)/std).reshape(10);out=np.concatenate((hist,token[None]),axis=0)[None]
    for k,v in dict(latent=state[None],history=hist[None],raw=raw,mean=mean,std=std,token=token,outgoing=out).items():
        resolver.same(packet[k],v,'A actual native packet '+k)
    require(events[0]['operation_index'] is events[-1]['operation_index'] is None,'packet/boundary no operation')
    ids=set()
    for name,positions,shape in [('action_encoder',(1,2),[1,3,10]),('predict',(3,8),[1,3,192]),
                               ('predictor',(4,5),[1,3,192]),('pred_proj',(6,7),[3,192])]:
        a,b=(events[i]['operation_index'] for i in positions)
        require(type(a) is int and a==b and a not in ids,'unique matched A resource ID');ids.add(a)
        ops=[op for op in operations if op['index']==a]
        require(len(ops)==1 and ops[0]['scope']==scope and ops[0]['resource']==name
                and ops[0]['status']=='returned' and ops[0]['error'] is None and ops[0]['shape']==shape,
                'native A returned resource identity')
    resolver.same(ae_in['action_encoder:0'],out,'A action encoder actual input')
    encoded=ae_out['action_encoder:output'];require(encoded.shape==(1,3,192),'A encoded actions full3')
    resolver.same(pr_in['predict:act_emb'],encoded,'A actual action embedding')
    resolver.same(pr_in['predict:latent'],state[None],'A complete input history')
    resolver.same(inner_in['predictor:0'],state[None],'A predictor history')
    resolver.same(inner_in['predictor:1'],encoded,'A predictor actions')
    hidden=inner_out['predictor:output'];require(hidden.shape==(1,3,192),'full3 predictor output')
    resolver.same(proj_in['pred_proj:0'],hidden.reshape(3,192),'A project reshape')
    full=pr_out['predict:output'];require(full.shape==(1,3,192),'full3 native output')
    resolver.same(proj_out['pred_proj:output'],full.reshape(3,192),'A native output')
    z=np.concatenate((state[1:],full[0,-1:]),axis=0);h=out[0,1:]
    resolver.same(boundary['latent'],z[None],'A exact shifted context')
    resolver.same(boundary['history'],h[None],'A exact shifted token history')
    expected=dict(task['root_context'],latent_history=z.tolist(),action_history=h.tolist())
    require(canonical(row['after'])==canonical(expected),'returned native full context, no endpoint substitution')
    require(sum(a.nbytes for g in groups for a in g.values())==25840,'fixed full IO byte budget')
    return ids


def validate_events(p,report,events,seal):
    require(len(report['rows'])==8 and [r['index'] for r in report['rows']]==list(range(8)), 'all8 report rows')
    rr=seal['resource_receipt'];core.audit_ledger(rr)
    require(rr['caps']==rr['attempted']==rr['completed']==caps(),'exact eight native resource axes')
    ops=rr['events'];require(len(ops)==32,'all32 actual module/API operations')
    require([op['index'] for op in ops]==list(range(32)),'whole native operation IDs')
    # Reconstruct full resource attempt/return + actual IO lifecycle, no unknown event filtering.
    cursor=0;allids=set()
    for job,row in zip(p['jobs'],report['rows'],strict=True):
        require(events[cursor]==dict(kind='A_intent',index=job['index'],job_sha256=digest(job)), 'literal fixed intent')
        cursor+=1;local=[]
        # Exact runtime nested resource order between the ten tensor events.
        layout=[('native',0),('attempt',0),('native',1),('native',2),('return',0),
                ('attempt',1),('native',3),('attempt',2),('native',4),('native',5),('return',2),
                ('attempt',3),('native',6),('native',7),('return',3),('native',8),('return',1),('native',9)]
        for kind,k in layout:
            e=events[cursor];cursor+=1
            if kind=='native':local.append(e)
            else:
                op=deepcopy(ops[job['index']*4+k])
                if kind=='attempt':op.update(status='attempted',ended_ns=None,error=None)
                require(e==dict(kind='resource_'+('attempt' if kind=='attempt' else 'return'),entry=op),
                        'whole nested resource/event chronology')
        ids=audit_A(job,row,local,ops);require(not ids & allids,'no cross-job operation splice');allids|=ids
        require(events[cursor]==dict(kind='A_return',index=job['index'],after=row['after']),'exact literal return');cursor+=1
    require(events[cursor:]==[dict(kind='predictions_sealed',sha256=report['prediction_seal_sha256']),
        dict(kind='labels_unlocked',preconditions_sha256=PRECONDITIONS_SHA),
        dict(kind='run_finished',status='all8_A_recorded_NOT_admitted')], 'seals before labels; no unknown extra call/event')
    require(seal['rows']==report['rows'] and seal['forbidden']==report['forbidden']==[], 'entire prelabel prediction rows')
    for k in ('caps','attempted','completed','events','inclusive_operation_wall_ns'):
        require(report['resource_receipt'][k]==rr[k],'report/seal actual resource ledger')


def consume(output,expected):
    output=Path(output).absolute();r=read(output/'report.json',expected,REPORT_CAP)
    p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['status']=='all8_A_recorded_NOT_admitted'
            and r['errors']==[] and all(r[k] is False for k in ('P1_admission','P3_admission','C3','physical_release')),
            'complete own A run only')
    require(type(r['elapsed_seconds']) in (int,float) and 0<=r['elapsed_seconds']<=WALL,
            'whole run wall includes own verify')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash']
            and r['inventory_before']==r['inventory_after'] and r['inventory_before']
            and r['precision_before']==r['precision_after']==precision_contract(),'actual frozen model and precision')
    for inv in r['inventory_before']:
        require(p['bindings'][inv['path']]==inv['sha256'],'original class source inventory')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2 and
            all(type(v) is int and 0<=v<=contract()[k] for v,k in zip(r['CUDA_peak_bytes'],
                ('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'actual CUDA memory caps')
    require(all(f.is_file() and not f.is_symlink() for f in output.iterdir()) and
            {f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
            sum(f.stat().st_size for f in output.iterdir())<=CAP,'full bounded artifact census')
    require(set(r['artifact_sha256'])=={'events.jsonl','binding.json','initial_jobs.json','prediction_seal.json','scores.json'},
            'exact complete own artifact vocabulary')
    for n,h in r['artifact_sha256'].items():require(file_sha(output/n)==h,'every actual artifact hash')
    binding=read(output/'binding.json',r['artifact_sha256']['binding.json'])
    require(binding==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
            read(output/'initial_jobs.json',r['artifact_sha256']['initial_jobs.json'])==p['jobs'],'literal prepared initial roster')
    # Parser authenticates every chain record, not a filtered subset.
    resolver.read_selected_journal(output/'events.jsonl',r['artifact_sha256']['events.jsonl'],
        r['journal_tail'],None)
    # scope=None picks control events only; load all already-authenticated payloads independently below.
    full=[]
    with (output/'events.jsonl').open('rb') as f:
        for line in f:full.append(saved.decode(line)['payload'])
    require(len(full)==r['event_count'],'whole event count')
    seal=read(output/'prediction_seal.json',r['prediction_seal_sha256'])
    require(r['artifact_sha256']['prediction_seal.json']==r['prediction_seal_sha256'],'prelabel seal file hash')
    for k in ('model_before','model_after','inventory_before','inventory_after','precision_before','precision_after'):
        require(seal[k]==r[k],'sealed before labels model evidence')
    validate_events(p,r,full,seal)
    scores=score(p,r['rows'])
    require(r['scores_sha256']==r['artifact_sha256']['scores.json'] and
            canonical(scores)==canonical(read(output/'scores.json',r['scores_sha256'])),'independent saved pair arithmetic')
    verify_bindings(p['bindings'],p['binding_sizes'])
    for n,h in r['artifact_sha256'].items():require(file_sha(output/n)==h,'all artifacts unchanged after strict')
    require(file_sha(output/'report.json')==expected,'report unchanged')
    return dict(schema=NAME+'.consume',status='all8_same_action_actual_history_verified_NOT_admitted',
        report_sha256=expected,prepare_sha256=r['prepare_sha256'],fixed_denominator=8,known_roots=2,fresh_roots=0,
        rows=scores,resource_receipt=r['resource_receipt'],R_new_calls=0,consume_new_neural=0,consume_new_physics=0,
        inherited_resources=p['inherited_resources'],full_upstream_strict_rebuilt_here=False,
        P1_admission=False,P3_admission=False,C3=False,physical_release=False,equal_total_compute=False,total_FLOPs=None)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=['prepare','verify','run','consume'])
    for k in ('preconditions','preconditions-sha','prepared','prepare-sha','output','report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-host-cuda',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write_metadata(a.output,build(a.preconditions,a.preconditions_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_saved_inputs',slots=len(p['jobs']))
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN explicit authorization flag');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'exclusive own receipt path required');v=write_metadata(a.receipt,consume(a.output,a.report_sha))
    print(canonical(v).decode())


if __name__=='__main__':main()
