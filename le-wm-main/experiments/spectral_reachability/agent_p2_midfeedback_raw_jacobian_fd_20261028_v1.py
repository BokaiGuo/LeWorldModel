"""Fixed raw central FD at all8 sealed v2 incumbent anchors. MAIN CUDA only.

No optimization, new AD, physics, images, outcome labels or admission threshold.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v2 as source

NAME='agent_p2_midfeedback_raw_jacobian_fd_20261028_v1'
ROOT=source.ROOT
F,bridge,physical,kernel=source.F,source.bridge,source.physical,source.kernel
require,sha,raw=source.require,source.sha,source.raw
SOURCE_SHA='e459cdc5de507353873346d6f86bc223afed291982904c4c4c6ce9bdfb7eaeab'
COMPILE_SHA='363925ff3714c6c9a5fbd2ec6c59e86dbcb63fe685cde1945d292ec1a9196117'
STEPS=(1e-2,1e-3,1e-4)
SIGNS=(-1,1)
AUTH='MAIN_HOST_CUDA_RAW_JACOBIAN_FD_ONLY'
CAP,FINAL,FLOOR,WALL=128*1024**2,8*1024**2,200*1024**2,900


def contract():
    return dict(points=8,known_roots=2,scientific_slots=4,steps=list(STEPS),coordinates=list(range(10)),outputs=192,
        signed_probe_denominator=480,central_pair_denominator=240,sealed_anchor_references=8,anchor_replays=0,max_predict=480,
        max_modules=dict(action_encoder=480,predict=480,pred_proj=480),predict_API_inclusive=480,
        new_AD=0,new_visual=0,new_physics=0,new_solve=0,new_selection=0,
        variable='literal float32 raw5x2, time-major; no tanh, logits, inverse or quantized-tanh derivative',
        perturbation='float32(float64(anchor_coordinate)+sign*nominal_h), all other bits untouched',
        domain='raw public action cube [-1,1]; derivative diagnostic, NOT feasible joint-logit planner candidates',
        missing='out-of-cube/collapsed probes retained null, never clamp/pad/drop; no subset matrix summary',
        FD='primary (zplus-zminus)/(2*nominal_h); separate actual-float32-spacing secant',
        resources='actual module and inclusive API attempts/returns, no nested-axis addition; inherited compile separate',
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',batch=1,total_FLOPs=None,
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        scientific_threshold=None,P1_admission=False,P2_admission=False,physical_release=False,
        AD_correctness_certified=False,physical_benefit_claim=False,training=False,Qfit=False,retries=0)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def packet(x):
    a=np.asarray(x)
    require(a.shape==(5,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and np.array_equal(a,a.astype(np.float32)),'literal finite rawfloat32 packet, no cast repair')
    return a.astype(np.float32)


def requests(point):
    anchor=packet(point['raw5']);require((np.abs(anchor)<=1).all(),'legal original raw anchor')
    result=[]
    for hindex,coordinate,sign in [(k,c,s) for k in range(3) for c in range(10) for s in SIGNS]:
        a=anchor.copy();h=STEPS[hindex]
        a.reshape(10)[coordinate]=np.float32(float(anchor.reshape(10)[coordinate])+sign*h)
        delta=float(a.reshape(10)[coordinate])-float(anchor.reshape(10)[coordinate])
        reason='outside_raw_cube' if (np.abs(a)>1).any() else 'float32_step_collapsed' if delta==0 else None
        label=f'h{hindex}_c{coordinate}_{"minus" if sign==-1 else "plus"}'
        result.append(dict(index=len(result),label=label,step_index=hindex,h=h,coordinate=coordinate,sign=sign,
            commands=a.astype(float).tolist(),raw_bits=a.astype('<f4').tobytes().hex(),actual_delta=delta,
            legal=reason is None,missing_reason=reason))
    require(len(result)==60,'fixed60 signed requests, no new anchor forward');return result


def extract(index,b,row,plan,output):
    """Compiler input whitelist: no selected suffix, native outcome or terminal RGB."""
    require(row['index']==index and row['branch_id']==b['branch_id'],'source row identity')
    jac=source.old.load_npy(output,row['arrays']['jacobian']);e=row['evaluations'][0]
    require(jac.dtype==np.float64 and jac.shape==(192,10) and np.isfinite(jac).all(),'literal saved raw AD 192x10')
    require(e['stage']=='incumbent' and source.suffix_bits(e['commands'])==source.suffix_bits(b['old_suffix_commands']),'literal original incumbent, not selected candidate')
    return dict(index=index,root_slot=plan['root_slot'],condition=plan['condition'],arm=plan['arm'],task_id=plan['task_id'],
        branch_id=b['branch_id'],parent_selection_digest=plan['parent_selection_digest'],
        context=deepcopy(b['observed_context']),context_sha256=F.core.digest(b['observed_context']),
        raw_history10=deepcopy(b['observed_raw_history']),raw5=deepcopy(b['old_suffix_commands']),goal=deepcopy(b['goal']),
        action_mean=deepcopy(b['action_mean']),action_std=deepcopy(b['action_std']),
        anchor_context=deepcopy(e['context']),anchor_context_sha256=F.core.digest(e['context']),anchor_Q=e['cost'],jacobian=jac.tolist(),
        jacobian_file=str(Path(output)/row['arrays']['jacobian']['file']),jacobian_sha256=row['arrays']['jacobian']['sha256'],
        incumbent_logits=deepcopy(b['incumbent_logits']),incumbent_domain=deepcopy(b['incumbent_domain']))


def build(compile_output,compile_sha):
    require(compile_sha==COMPILE_SHA and sha(source.__file__)==SOURCE_SHA,'fixed actual v2 compile/source authority')
    output=Path(compile_output).resolve();p,plans=source.consume_compile(output,compile_sha)
    report=physical.read(output/'report.json');points=[extract(i,b,r,plan,output) for i,(b,r,plan) in enumerate(zip(p['branches'],report['rows'],plans,strict=True))]
    require(len(points)==8,'all8 anchors, no subset')
    schedule=[requests(point) for point in points];bindings=dict(p['bindings'])
    physical.merge(bindings,{str(f):sha(f) for f in output.iterdir()})
    physical.merge(bindings,{str(f):sha(f) for f in own_files()})
    bridge.verify_bindings(bindings)
    return dict(schema=NAME+'.prepare',arguments=dict(compile_output=str(output),compile_sha=compile_sha),contract=contract(),
        points=points,schedule=schedule,expected_predict=[sum(r['legal'] for r in rs) for rs in schedule],bindings=bindings,
        policy=p['policy'],expected_model_hash=p['expected_model_hash'],inherited_compiler_contract=p['contract'],
        inherited_compiler_resources=[r['resources'] for r in report['rows']],inherited_AD_API=[r['AD_API_resources'] for r in report['rows']])


def verify(path,expected):
    require(sha(path)==expected,'own prepare SHA');p=physical.read(path)
    require(raw(p)==raw(build(**p['arguments'])),'whole fixed source/arrays/schedule reconstruction');return p


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(payload)<=32*1024**2
        and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'new own bounded metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    kernel.paired.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


class Store(kernel.Store):
    def __init__(self,path,started):
        require(Path(path).name.startswith(NAME+'_') and shutil.disk_usage(Path(path).parent).free>=FLOOR+CAP,'new own128MiB +200MiB floor')
        super().__init__(path);self.started=started;self.branch=None
    @property
    def output(self):return self.path
    def check(self,size=4096,final=False):
        require(final or time.monotonic()-self.started<WALL,'whole-stage900s including verification, not watchdog')
        used=sum(f.stat().st_size for f in self.path.iterdir())
        require(used+size<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+size,'128MiB cap/8MiB final/200MiB floor')
    def _write(self,n,b):return self.write(n,b)
    def event(self,e):self.emit('native_event',branch=self.branch,event=e)


def task(point):return dict(action_mean=point['action_mean'],action_std=point['action_std'])


class Modules(F.Modules):
    """Frozen complete-array module hooks with predeclared FD forward schedule."""
    def __init__(self,model,store,stem,meter,schedule,sync=lambda:None):
        super().__init__(model,store,stem,meter,sync);self.schedule=[r for r in schedule if r['legal']]
    def pre(self,kind):
        def enter(module,args):
            n=len(self.entries);k=n//3
            require(not self.failed and not self.pending and k<len(self.schedule) and kind==self.ORDER[n%3]
                and self.meter.stage==self.schedule[k]['label'],'fixed bounded native FD order, no extra calls')
            e=dict(index=n,resource=kind,stage=self.meter.stage,completed=False,inputs=[])
            self.entries.append(e);self.pending[module]=e;self.store.event(dict(kind='module_attempt',entry=deepcopy(e)))
            require(len(args)==(2 if kind=='predict' else 1),'exact native argument arity')
            for j,x in enumerate(args):
                a=self.array(x)
                if kind=='action_encoder':require(a.shape==(1,3,10) and np.array_equal(a[0],self.meter.expected_input['action_history']),'literal full3 normalized raw tokens')
                elif kind=='predict':require((j==0 and a.shape==(1,3,192) and np.array_equal(a[0],self.meter.expected_input['latent_history'])) or
                    (j==1 and np.array_equal(a,self.action_embedding)),'actual sealed context/action embeddings')
                else:require(a.shape==(3,192) and np.array_equal(a,self.predictor_output.reshape(3,192)),'whole native predictor->projection')
                e['inputs'].append(F.save_array(self.store,f'{self.stem}.m{n:03d}.i{j}.npy',a))
            self.store.event(dict(kind='module_inputs_saved',entry=deepcopy(e)))
        return enter


class PredictAPI(source.PredictAPI):
    def __init__(self,model,meter,sink,sync,schedule):
        super().__init__(model,meter,sink,sync);self.schedule=[r for r in schedule if r['legal']]
    def __enter__(self):
        self.had='predict' in self.model.__dict__;self.original_local=self.model.__dict__.get('predict');self.original=self.model.predict
        def call(*a,**kw):
            n=len(self.entries)
            require(n<len(self.schedule) and all(e['completed'] for e in self.entries) and self.meter.stage==self.schedule[n]['label'],'fixed no-retry native predict API')
            e=dict(index=n,stage=self.meter.stage,completed=False);self.entries.append(e);self.sink.event(dict(kind='predict_API_attempt',entry=deepcopy(e)))
            out=self.original(*a,**kw);self.sync();e['completed']=True;self.sink.event(dict(kind='predict_API_return',entry=deepcopy(e)));return out
        self.model.predict=call;return self


class NoAD(F.blind.ADTrace):
    def dispatch(self,kind,fn,args,kwargs):
        self.failed=True;self.sink(dict(kind='forbidden_AD_API',resource=kind));raise ValueError('FD forward only, zero AD budget')


def evaluate(point,schedule,backend,meter,store,row):
    require(schedule==requests(point),'literal predeclared all60 requests')
    row.update(status='attempted',evaluations=[dict(request=deepcopy(r),status='not_started') for r in schedule])
    for e in row['evaluations']:
        r=e['request']
        if not r['legal']:
            e['status']='missing_domain';store.event(dict(kind='probe_missing',receipt=deepcopy(e)));continue
        meter.stage=r['label'];meter.last_native=None;meter.expected_input=F.planner.incoming(task(point),point['context'],r['commands'])
        e['status']='attempted';store.event(dict(kind='evaluation_intent',request=r,expected_input=meter.expected_input))
        after=backend.step(deepcopy(point['context']),packet(r['commands']));F.planner.advance(task(point),point['context'],r['commands'],after)
        require(np.array_equal(meter.last_native,after['latent_history'][-1]),'returned actual full native endpoint')
        stem=f'p{point["index"]}_r{r["index"]}'
        e.update(status='completed',context_sha256=F.core.digest(after),
            latent_array=F.save_array(store,stem+'.latent.npy',np.asarray(after['latent_history'],np.float32)),
            token_array=F.save_array(store,stem+'.tokens.npy',np.asarray(after['action_history'],np.float32)))
        store.event(dict(kind='evaluation_return',receipt=deepcopy(e)))
    row['status']='recorded';return row


def check_point(point,schedule,row,events,load):
    require(row['index']==point['index'] and row['status']=='recorded' and schedule==requests(point)
        and len(row['evaluations'])==60 and F.core.digest(point['anchor_context'])==point['anchor_context_sha256'],
        'fixed full point denominator and original anchor seal')
    valid=[r for r in schedule if r['legal']];n=len(valid)
    F.blind.exact_counts(row['resources'],{k:n for k in Modules.ORDER})
    F.blind.exact_counts(row['predict_API_resources'],dict(predict=n));F.blind.exact_counts(row['AD_API_resources'],dict(grad=0,jacobian=0))
    require(row['AD_API_ledger']==[] and row['predict_API_ledger']==[dict(index=i,stage=r['label'],completed=True) for i,r in enumerate(valid)],'actual no-AD/native API census')
    require(len(row['module_ledger'])==3*n,'all actual native module entries')
    want=[];module_cursor=api_cursor=0;terminals=[]
    for request,e in zip(schedule,row['evaluations'],strict=True):
        require(e['request']==request,'each signed request kept exact')
        if not request['legal']:
            require(e==dict(request=request,status='missing_domain'),'invalid action never executed or padded')
            want.append(dict(kind='probe_missing',receipt=e));terminals.append(None);continue
        require(e['status']=='completed','valid probe returned, pending not complete')
        incoming=F.planner.incoming(task(point),point['context'],request['commands'])
        want.append(dict(kind='evaluation_intent',request=request,expected_input=incoming));ins=[];outs=[]
        api=row['predict_API_ledger'][api_cursor];api_cursor+=1
        for j,kind in enumerate(Modules.ORDER):
            m=row['module_ledger'][module_cursor];module_cursor+=1
            require(m['index']==module_cursor-1 and m['resource']==kind and m['stage']==request['label'] and m['completed'] is True,'complete literal module sequence')
            if kind=='predict':want.append(dict(kind='predict_API_attempt',entry=dict(api,completed=False)))
            want.append(dict(kind='module_attempt',entry=dict(index=m['index'],resource=kind,stage=request['label'],completed=False,inputs=[])))
            want.append(dict(kind='module_inputs_saved',entry={k:v for k,v in m.items() if k!='output'}|dict(completed=False)))
            want.append(dict(kind='module_return',entry=m))
            if kind=='pred_proj':want.append(dict(kind='predict_API_return',entry=api))
            ins.append([load(a) for a in m['inputs']]);outs.append(load(m['output']))
            require(all(a.dtype==np.float32 and np.isfinite(a).all() for a in ins[-1]+[outs[-1]]),'literal native float32 arrays')
        require(len(ins[0])==1 and ins[0][0].shape==(1,3,10) and np.array_equal(ins[0][0][0],incoming['action_history'])
            and outs[0].ndim==3 and outs[0].shape[:2]==(1,3),'full3 actual normalized action input')
        require(len(ins[1])==2 and ins[1][0].shape==(1,3,192) and np.array_equal(ins[1][0][0],incoming['latent_history'])
            and np.array_equal(ins[1][1],outs[0]) and outs[1].shape==(1,3,192),'sealed history->native predictor')
        require(len(ins[2])==1 and np.array_equal(ins[2][0],outs[1].reshape(3,192)) and outs[2].shape==(3,192),'full native projected outputs')
        lat,tok=load(e['latent_array']),load(e['token_array'])
        require(lat.dtype==tok.dtype==np.float32 and lat.shape==(3,192) and tok.shape==(2,10),'complete shifted history buffers')
        after=dict(point['context'],latent_history=lat.astype(float).tolist(),action_history=tok.astype(float).tolist())
        F.planner.advance(task(point),point['context'],request['commands'],after)
        require(F.core.digest(after)==e['context_sha256'] and np.array_equal(lat[-1],outs[2][-1]),'exact returned context/native endpoint')
        want.append(dict(kind='evaluation_return',receipt=e));terminals.append(lat[-1].astype(np.float64))
    require(events==want,'whole native/evaluation event ABI, unknown/drop/reorder never filtered')
    return summarize(point,schedule,terminals)


def measures(ad,fd):
    ad,fd=np.asarray(ad,np.float64),np.asarray(fd,np.float64)
    require(ad.shape==fd.shape and np.isfinite(ad).all() and np.isfinite(fd).all(),'finite aligned derivative arrays')
    d=fd-ad;na,nf=float(np.linalg.norm(ad)),float(np.linalg.norm(fd));nd=float(np.linalg.norm(d))
    return dict(absolute_l2=nd,relative_l2_to_AD=None if na==0 else nd/na,
        symmetric_relative_l2=None if max(na,nf)==0 else nd/max(na,nf),
        direction_cosine=None if na==0 or nf==0 else float(np.sum(ad*fd)/(na*nf)),
        max_abs=float(np.max(np.abs(d))),RMSE=float(np.sqrt(np.mean(d*d))),AD_norm=na,FD_norm=nf)


def summarize(point,schedule,terminals):
    require(len(schedule)==len(terminals)==60,'all60 signed slots, sealed nominal reference')
    j=np.asarray(point['jacobian'],np.float64);z0=np.asarray(point['anchor_context']['latent_history'][-1],np.float64)
    require(j.shape==(192,10) and np.isfinite(j).all(),'original raw literal AD')
    by_step=[]
    for si,h in enumerate(STEPS):
        columns=[];actual_columns=[];pairs=[]
        for c in range(10):
            mi=20*si+2*c;pi=mi+1;minus,plus=terminals[mi],terminals[pi]
            missing=minus is None or plus is None
            item=dict(coordinate=c,minus_request_index=mi,plus_request_index=pi,missing=missing,AD=j[:,c].tolist(),
                minus_missing_reason=schedule[mi]['missing_reason'],plus_missing_reason=schedule[pi]['missing_reason'],
                nominal_fd=None,actual_spacing_fd=None,nominal_metrics=None,actual_spacing_metrics=None,
                nominal_signed_error=None,nominal_absolute_error=None,relative_element_to_AD=None,curvature=None)
            if not missing:
                dm=-schedule[mi]['actual_delta'];dp=schedule[pi]['actual_delta'];require(dm>0 and dp>0,'strict positive actual spacing, no inverse rounding')
                fd=(plus-minus)/(2*h);actual=(plus-minus)/(dp+dm);d=fd-j[:,c]
                second=2*((plus-z0)/dp-(z0-minus)/dm)/(dp+dm)
                plus_res=plus-z0-j[:,c]*dp;minus_res=minus-z0+j[:,c]*dm
                item.update(nominal_fd=fd.tolist(),actual_spacing_fd=actual.tolist(),nominal_metrics=measures(j[:,c],fd),
                    actual_spacing_metrics=measures(j[:,c],actual),nominal_signed_error=d.tolist(),nominal_absolute_error=np.abs(d).tolist(),
                    relative_element_to_AD=[None if a==0 else float(abs(x)/abs(a)) for a,x in zip(j[:,c],d,strict=True)],
                    curvature=dict(actual_minus_step=dm,actual_plus_step=dp,actual_span=dp+dm,midpoint_shift=(dp-dm)/2,
                        second_difference192=second.tolist(),second_difference_l2=float(np.linalg.norm(second)),
                        plus_AD_linearization_residual192=plus_res.tolist(),minus_AD_linearization_residual192=minus_res.tolist(),
                        plus_residual_l2=float(np.linalg.norm(plus_res)),minus_residual_l2=float(np.linalg.norm(minus_res))))
                columns.append(fd);actual_columns.append(actual)
            pairs.append(item)
        complete=len(columns)==10
        by_step.append(dict(h=h,columns=pairs,full_matrix_available=complete,
            nominal_FD_matrix=np.column_stack(columns).tolist() if complete else None,
            actual_spacing_FD_matrix=np.column_stack(actual_columns).tolist() if complete else None,
            nominal_matrix_metrics=measures(j,np.column_stack(columns)) if complete else None,
            actual_spacing_matrix_metrics=measures(j,np.column_stack(actual_columns)) if complete else None))
    return dict(index=point['index'],root_slot=point['root_slot'],condition=point['condition'],arm=point['arm'],
        context_sha256=point['context_sha256'],jacobian_sha256=point['jacobian_sha256'],steps=by_step,
        fixed_signed_denominator=60,fixed_pair_denominator=30,AD_correctness_certified=False,scientific_threshold=None)


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA before I/O')
    started=time.monotonic();p=verify(prepared,expected);store=Store(output,started);store.check()
    rows=[dict(index=i,status='not_started',evaluations=[],module_ledger=[],predict_API_ledger=[],AD_API_ledger=[]) for i in range(8)]
    errors=[];model=None;before=after=None;flags=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original fixed model')
        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            for i,(point,schedule,row) in enumerate(zip(p['points'],p['schedule'],rows,strict=True)):
                store.branch=i;store.check();store.emit('point_intent',branch=i,expected_predict=p['expected_predict'][i])
                meter=SimpleNamespace(stage=None,expected_input=None,last_native=None)
                sync=lambda:torch.cuda.synchronize(device)
                modules=Modules(model,store,f'p{i}',meter,schedule,sync);api=PredictAPI(model,meter,store,sync,schedule)
                ad=NoAD(torch,F.blind.ARMS[1],store.event,sync)
                try:
                    with ad,modules,api:evaluate(point,schedule,F.d2p.TorchBackend(model,device),meter,store,row)
                finally:
                    row.update(module_ledger=deepcopy(modules.entries),predict_API_ledger=deepcopy(api.entries),AD_API_ledger=deepcopy(ad.events))
                # Strict saved validation is deferred to consume, not extra forward calls.
                store.emit('point_return',branch=i,status=row['status'])
    except BaseException as e:errors.append(kernel.error('FD',e))
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');store.check()
        except BaseException as e:errors.append(kernel.error('final_closure_model_limits',e))
        summaries=[]
        for i,row in enumerate(rows):
            row['resources']={k:dict(attempted=sum(e['resource']==k for e in row['module_ledger']),
                completed=sum(e['resource']==k and e['completed'] for e in row['module_ledger'])) for k in Modules.ORDER}
            row['predict_API_resources']=dict(predict=dict(attempted=len(row['predict_API_ledger']),completed=sum(e['completed'] for e in row['predict_API_ledger'])))
            row['AD_API_resources']={k:dict(attempted=sum(e['resource']==k for e in row['AD_API_ledger']),
                completed=sum(e['resource']==k and e['completed'] for e in row['AD_API_ledger'])) for k in ('grad','jacobian')}
            filename=f'point_{i}.json';h=store.put(filename,row,final=True)
            summaries.append(dict(index=i,status=row['status'],file=filename,sha256=h,resources=row['resources'],predict_API_resources=row['predict_API_resources']))
        complete=not errors and before==after==p['expected_model_hash'] and all(r['status']=='recorded' for r in rows)
        status='raw_FD_recorded_diagnostic_only' if complete else 'partial_or_failed'
        store.emit('run_finished',status=status)
        r=dict(schema=NAME+'.report',status=status,contract=contract(),rows=summaries,errors=errors,
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,model_before=before,model_after=after,
            elapsed_seconds=time.monotonic()-started,journal_chain_tail=store.previous,
            artifact_sha256={f.name:sha(f) for f in store.path.iterdir()},new_AD=0,new_physics=0,P1_admission=False,P2_admission=False)
        store.put('report.json',r,final=True)
    require(complete,'partial full8 retained, no retry: '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual FD report SHA')
    r=physical.read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='raw_FD_recorded_diagnostic_only' and r['contract']==p['contract']==contract()
        and r['errors']==[] and r['model_before']==r['model_after']==p['expected_model_hash'] and 0<=r['elapsed_seconds']<WALL
        and r['new_AD']==r['new_physics']==0 and r['P1_admission'] is r['P2_admission'] is False and len(r['rows'])==8,'complete actual diagnostic only')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in output.iterdir())<=CAP,'whole output census/cap')
    for n,h in r['artifact_sha256'].items():require(sha(bridge.rgb.flat(output,n))==h,'all actual file SHAs')
    require(physical.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'actual prepared execution binding')
    events=physical.journal(output/'journal.jsonl',r['journal_chain_tail'])
    require(events[-1]['kind']=='run_finished' and events[-1]['status']==r['status'] and
        all(e['kind'] in ('point_intent','native_event','point_return') and type(e['branch']) is int and e['branch'] in range(8) for e in events[:-1])
        and [e['branch'] for e in events[:-1]]==sorted(e['branch'] for e in events[:-1]),'entire ordered actual journal')
    rows=[];counts={k:dict(attempted=0,completed=0) for k in Modules.ORDER};api=dict(attempted=0,completed=0)
    for i,(point,schedule,summary) in enumerate(zip(p['points'],p['schedule'],r['rows'],strict=True)):
        require(summary['index']==i and summary['status']=='recorded' and summary['file']==f'point_{i}.json'
            and sha(output/summary['file'])==summary['sha256'],'fixed8 actual point row files')
        row=physical.read(output/summary['file']);segment=[e for e in events[:-1] if e['branch']==i]
        require(segment[0]['kind']=='point_intent' and segment[0]['expected_predict']==p['expected_predict'][i] and
            segment[-1]['kind']=='point_return' and segment[-1]['status']=='recorded' and all(e['kind']=='native_event' for e in segment[1:-1]),'durable reserve/entry/return for all8 points')
        require(all(summary[k]==row[k] for k in ('resources','predict_API_resources')),'top-level actual resource counters')
        rows.append(check_point(point,schedule,row,[e['event'] for e in segment[1:-1]],lambda a:source.old.load_npy(output,a)))
        for k in counts:
            for state in counts[k]:counts[k][state]+=row['resources'][k][state]
        for state in api:api[state]+=row['predict_API_resources']['predict'][state]
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'report unchanged after all arrays')
    return dict(schema=NAME+'.consume',status='full8_raw_FD_saved_verified_diagnostic_only',report_sha256=expected,
        compile_report_sha256=p['arguments']['compile_sha'],rows=rows,resources=counts,predict_API_resources=dict(predict=api),
        signed_probe_denominator=480,central_pair_denominator=240,sealed_anchor_denominator=8,anchor_replays=0,known_roots=2,
        inherited_compiler_contract=p['inherited_compiler_contract'],inherited_compiler_resources=p['inherited_compiler_resources'],
        inherited_AD_API=p['inherited_AD_API'],total_FLOPs=None,new_AD=0,new_physics=0,consume_model_calls=0,
        scientific_threshold=None,P1_admission=False,P2_admission=False,physical_release=False,AD_correctness_certified=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','compile-output','compile-sha','report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-host-cuda',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.compile_output,a.compile_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',predict=sum(p['expected_predict']))
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN explicit CUDA flag');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'own saved receipt');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()
