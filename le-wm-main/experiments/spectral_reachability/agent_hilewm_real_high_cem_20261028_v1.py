"""Installed CEM -> owned HiLeWM root.get_cost(high), MAIN HOST CUDA only.

Integration inputs only, not a task/bank/official HiLeWM performance experiment.
v2 is diagnostic evidence, NOT full-base certification. No mutable v3 import.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
from experiments.spectral_reachability import agent_hilewm_solver_cuda_seam_20261028_v1 as seam
from experiments.spectral_reachability import agent_hilewm_trusted_cuda_validation_20261028_v2 as diag

TAG='agent_hilewm_real_high_cem_20261028_v1'
ROOT=seam.ROOT
CAP,FLOOR,WALL,RESERVE=32*1024**2,200*1024**2,600,4*1024**2
need,encode,digest=seam.require,seam.encode,seam.digest
V2=ROOT/'results/agent_hilewm_trusted_cuda_validation_20261028_v2_prepare_MAIN01/manifest.json'
V2_SHA='eadbb1d11383a46b02866a1966588b482dccbb45d6684a55c812d81925782c72'
V2_REPORT=ROOT/'results/agent_hilewm_trusted_cuda_validation_20261028_v2_cuda_MAIN01/report.json'
V2_REPORT_SHA='8f19346c99aba5c3cc5dd6efec878356938c24bcb53023444c37791f94ff6771'
V2_RECEIPT=ROOT/'results/agent_hilewm_trusted_cuda_validation_20261028_v2_consume_MAIN01/receipt.json'
V2_RECEIPT_SHA='0c2b3055c91cd01101774511e8dff24693ec46caa3a91a7ef43b495eeba0a4d9'
PINS={str(Path(seam.__file__).resolve()):'dbf9662e669554b7a57f73e6caad63b30d56993e599c1fa7b10e5a3b88c46622',
    str(Path(diag.__file__).resolve()):'35836e271ac09e3b393e33f01adf3f39d5c2b828aa0f2734acb11d30d15102f0',
    str(ROOT/'tests/test_agent_hilewm_trusted_cuda_validation_20261028_v2.py'):'155aa858acc46540c7063e9794b01ed81ea2ad134e76aa5b08a50200109970f5',
    str(ROOT/'innerdocs/agent_hilewm_trusted_cuda_validation_20261028_v2.md'):'5b92d489e593b8930eea3f30e6c347e75aa918c58410327cd13e2a939055147c'}
KINDS=('macro_to_condition','high_predictor','high_pred_proj')
CHECKS=('topk_vals','topk_inds','gather','mean','sample_std','mean_candidate')


def protocol():
    return dict(schema=TAG+'.protocol',batches=[5,2],chunk=2,samples=8,iterations=3,topk=3,
        horizon=2,action_block=2,macro_dim=32,dim=64,latent=192,var_scale=1.,seed_base=1729,
        input_recipe='float32((arange(numel)%31)-15+1729+B+role)/64; z role0, goal role13',
        initial_action='full B,H,64 zero CUDA; no actor',bounds=[-1.,1.],
        bounds_semantics='Box metadata only: installed CEM does NOT clamp sampled candidates',
        interpretation='synthetic integration ONLY; not official bank/task/OOD/performance',
        cost='unchanged owned root.get_cost(planner_level=high); no analytic substitution',
        cost_attempts=12,candidate_rows=168,high_predictor_calls=48,high_predictor_sample_rows=672,
        high_predictor_context_tokens=1680,pixel_encoder=0,macro_encoder=0,AD=0,physics=0,
        source_authority='v2 exact-layout diagnostic only, NOT fullbase cert',
        run_precondition='MAIN explicitly confirms frozen v3 allfour passed; no v3 bytes read here',
        saved_float_rtol=seam.protocol()['saved_float_rtol'],saved_float_atol=seam.protocol()['saved_float_atol'],
        saved_tolerance_scope='CEM NumPy reduction rounding ONLY, never neural model parity',
        cost_rounding=dict(dimension=192,unit_roundoff=float(np.finfo(np.float32).eps/2),gamma_operations=194,
            underflow_absolute='194 * smallest_float32_subnormal',scope='saved f32 subtraction/square/sum arithmetic, not neural parity'),
        output_cap_bytes=CAP,final_reserve_bytes=RESERVE,floor_bytes=FLOOR,wall_seconds=WALL,
        CUDA_only=True,CPU_fallback=False,training=False,official_parity=False,baseline_runnable=False,
        scientific_admission=False)


def inputs(batch):
    need(type(batch) is int and batch in (5,2),'fixed B5,B2')
    return {k:diag.old.array_input([batch,192],1729+batch+offset) for k,offset in (('z_init',0),('z_goal',13))}


def read(path,h=None):
    p=Path(path);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'bounded regular JSON')
    body=p.read_bytes();need(h is None or digest(body)==h,'actual JSON SHA')
    return json.loads(body,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def sources():
    for path,h in PINS.items():need(digest(Path(path).read_bytes())==h,'frozen source pin '+path)
    files=[Path(__file__).resolve(),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')]
    out=seam.sources();out.update(diag.closure());out.update(PINS)
    out.update({str(p):digest(p.read_bytes()) for p in files})
    need(all(digest(Path(p).read_bytes())==h for p,h in out.items()),'all own/installed/source bytes')
    return out


def authority():
    result=diag.consume(V2,V2_SHA,V2_REPORT,V2_REPORT_SHA)
    need(result==read(V2_RECEIPT,V2_RECEIPT_SHA),'entire actual v2 diagnostic receipt reconstructed')
    parent,_=diag.upstream()
    return dict(v2_manifest=str(V2),v2_manifest_sha256=V2_SHA,v2_report=str(V2_REPORT),
        v2_report_sha256=V2_REPORT_SHA,v2_receipt=str(V2_RECEIPT),v2_receipt_sha256=V2_RECEIPT_SHA,
        diagnostic_only=True,export={k:parent[k] for k in
            ('export_report','export_report_sha256','export_receipt','export_receipt_sha256')})


def build():
    own=sources();a=authority();b=seam.bridge_module()
    return dict(schema=TAG+'.prepare',protocol=protocol(),sources=own,authority=a,
        owner_evidence=b.source_evidence(),neural_run_approved=False,fullbase_certificate=False)


def scoped(path):
    p=Path(path)
    need(p.is_absolute() and p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(TAG+'_'),
        'new absolute own-stem results path');return p


def write_new(path,body):
    p=Path(path);need(not p.exists() and not p.is_symlink() and len(body)<=CAP
        and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'exclusive bytes/floor')
    with p.open('xb') as f:need(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())


def prepare(output):
    value=build();p=scoped(output);need(shutil.disk_usage(p.parent).free>=CAP+FLOOR,'full cap+floor')
    p.mkdir();write_new(p/'manifest.json',encode(value));return dict(manifest=str(p/'manifest.json'),sha256=digest(encode(value)))


def verify(path,h):
    m=read(path,h);need(m==build(),'whole manifest/diagnostic/source/export drift')
    need(digest(Path(path).read_bytes())==h,'prepare unchanged');return m


class Store:
    def __init__(self,path,started):
        self.path=scoped(path);self.started=started;self.used=0;self.arrays={}
        need(shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'32MiB+floor before output')
        self.path.mkdir()
    def check(self,size=0,final=False):
        need(final or time.monotonic()-self.started<=WALL,'whole600s cooperative wall')
        need(self.used+size<=CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free>=FLOOR+size,'actual cap/reserve/floor')
    def put(self,name,body,final=False):
        need(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat new file')
        self.check(len(body),final);write_new(self.path/name,body);self.used+=len(body)
    def array(self,name,a):
        a=np.asarray(a);need(a.dtype in (np.dtype('<f4'),np.dtype('<i8')) and np.isfinite(a).all(),'finite native f32/i64 only')
        need(name not in self.arrays and name.endswith('.npy'),'exclusive NPY')
        s=io.BytesIO();np.save(s,a,allow_pickle=False);body=s.getvalue();self.put(name,body)
        self.arrays[name]=dict(file_sha256=digest(body),file_bytes=len(body),data_sha256=digest(a.tobytes()),
            dtype=a.dtype.str,shape=list(a.shape));return name


def resources(cases):
    calls=[a for c in cases for a in c.get('cost_attempts',[])];native=[e for c in cases for e in c.get('native',[])]
    return dict(cost=dict(attempted=len(calls),returned=sum(a['returned'] for a in calls)),
        bridge_candidate_rows=sum(c.get('bridge_receipt',{}).get('candidate_rows_attempted',0) for c in cases),
        modules={k:dict(attempted=sum(e['kind']==k and e['phase']=='attempt' for e in native),
            returned=sum(e['kind']==k and e['phase']=='return' for e in native),
            sample_rows=sum(e['batch'] for e in native if e['kind']==k and e['phase']=='attempt'),
            context_tokens=sum(e['batch']*e['context'] for e in native if e['kind']==k and e['phase']=='attempt')) for k in KINDS},
        AD=0,pixel_encoder=0,macro_encoder=0,physics=0)


def precision(t):
    value=dict(deterministic=t.are_deterministic_algorithms_enabled(),tf32_matmul=t.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=t.backends.cudnn.allow_tf32,autocast=t.is_autocast_enabled('cuda'),grad=t.is_grad_enabled())
    need(value==dict(deterministic=True,tf32_matmul=False,tf32_cudnn=False,autocast=False,grad=False),'actual CUDA inference precision')
    return value


def run_case(t,CEM,Box,root,batch,store,case):
    p=protocol();device=next(root.parameters()).device;b=seam.bridge_module();abi=seam.cuda_abi(t,device)
    info={'planner_level':'high',**{k:t.from_numpy(v).to(device) for k,v in inputs(batch).items()}}
    case.update(events=[],cost_attempts=[],native=[],resets=0,starts=0,ends=0,outputs={},blocked=[])
    current=dict(chunk=-1,attempt=None,last_projection=None);handles=[];pending={}
    def save(label,x):
        if t.is_tensor(x):
            need(x.dtype in (t.float32,t.int64) and not x.requires_grad,'original native dtype/inference')
            x=x.detach().cpu().numpy().copy()
        return store.array(f'B{batch}_{label}.npy',x)
    def pre(kind):
        def hook(module,args):
            store.check();precision(t);a=current['attempt'];need(a is not None and not t.is_grad_enabled(),'scoped native high cost')
            ordinal=sum(e['phase']=='attempt' and e['kind']==kind for e in case['native'])
            step=ordinal%4;need(ordinal//4==a['index'],'four real calls per cost, before dispatch cap')
            n=args[0].shape[0];context=(1 if kind=='high_pred_proj' else args[0].shape[1])
            event=dict(phase='attempt',kind=kind,cost_index=a['index'],step=step,batch=n,context=context,
                inputs=[save(f'n{len(case["native"])}_in{j}',v) for j,v in enumerate(args)],
                input_layouts=[diag.layout(v) for v in args])
            case['native'].append(event);pending[kind]=event
        return hook
    def post(kind):
        def hook(module,args,out):
            event=pending.pop(kind);t.cuda.synchronize();store.check()
            case['native'].append(dict(event,phase='return',output=save(f'n{len(case["native"])}_out',out),output_layout=diag.layout(out)))
            if kind=='high_pred_proj' and event['step']==3:current['last_projection']=out
        return hook
    def blocked(name):
        def hook(*args):case['blocked'].append(name);raise ValueError('forbidden pixel/macro/low forward '+name)
        return hook
    class ObservedHigh(b.HighBatchBridge):
        def get_cost(self,expanded,candidates):
            store.check();index=len(case['cost_attempts']);offset=current['chunk']*2;n=min(2,batch-offset)
            a=dict(index=index,chunk=current['chunk'],returned=False,candidate_rows=n*8)
            case['cost_attempts'].append(a);current['attempt']=a
            need(index<((batch+1)//2)*3,'fixed cost dispatch cap')
            for k in ('z_init','z_goal'):
                expected=info[k][offset:offset+n,None].expand(n,8,192)
                need(tuple(expanded[k].shape)==(n,8,192) and expanded[k].stride(1)==0
                    and t.equal(expanded[k],expected),'actual expanded source/tail lineage')
                a[k]=save(f'cost{index}_{k}',expanded[k])
            a['candidates']=save(f'cost{index}_candidates',candidates)
            value=super().get_cost(expanded,candidates)  # direct root.get_cost, no replacement cost
            terminal=current['last_projection'].reshape(n*8,4,192)[:,-1].reshape(n,8,192)
            check=(terminal-expanded['z_goal']).square().sum(dim=-1)
            a['native_cost_reduction_exact']=t.equal(value,check)
            need(a['native_cost_reduction_exact'],'actual root cost terminal/reduction join')
            a['cost']=save(f'cost{index}_value',value);a['returned']=True;current['attempt']=None
            return value
    facade=ObservedHigh.bind_high_model(info,root,tensor_batch=batch,chunk_size=2,num_samples=8,abi=abi,
        max_cost_calls=((batch+1)//2)*3,max_candidate_rows=batch*8*3)
    class Recorder:
        output_key=TAG
        def __init__(self):self.history=[]
        def reset(self):case['resets']+=1
        def start_batch(self):case['starts']+=1;current['chunk']+=1
        def end_solve(self):case['ends']+=1
        def __call__(self,**state):
            store.check();index=len(case['events']);a=case['cost_attempts'][index]
            event=dict(step=state['step'],chunk=current['chunk'],attempt=index,
                arrays={k:save(f'event{index}_{k}',state[k]) for k in seam.KEYS})
            case['events'].append(event)
            vals,inds=t.topk(state['costs'],k=3,dim=1,largest=False)
            elite=state['candidates'][t.arange(len(vals),device=device)[:,None].expand(-1,3),inds]
            event['cuda_exact_checks']=dict(topk_vals=t.equal(vals,state['topk_vals']),topk_inds=t.equal(inds,state['topk_inds']),
                gather=t.equal(elite,state['topk_candidates']),mean=t.equal(elite.mean(dim=1),state['mean']),
                sample_std=t.equal(elite.std(dim=1),state['var']),mean_candidate=t.equal(state['candidates'][:,0],state['prev_mean']))
            need(a['returned'] and all(event['cuda_exact_checks'].values()),'native cost return/CEM exact update')
            self.history.append(dict(chunk=current['chunk'],step=state['step']))
    callback=Recorder()
    try:
        for kind in KINDS:
            module=getattr(root,kind);handles.extend((module.register_forward_pre_hook(pre(kind)),module.register_forward_hook(post(kind))))
        for name in ('encoder','latent_action_encoder','low_predictor','low_pred_proj'):
            if hasattr(root,name):handles.append(getattr(root,name).register_forward_pre_hook(blocked(name)))
        solver=CEM(model=facade,batch_size=2,num_samples=8,var_scale=1.,n_steps=3,topk=3,
            device=device,seed=1729+batch,callbacks=[callback])
        solver.configure(action_space=Box(-1.,1.,shape=(batch,32),dtype=np.float32),n_envs=batch,
            config=SimpleNamespace(horizon=2,action_block=2))
        initial=t.zeros((batch,2,64),device=device,dtype=t.float32);case['initial']=save('initial',initial)
        with t.inference_mode():out=solver.solve(dict(facade.solver_info),init_action=initial)
        t.cuda.synchronize()
        case['outputs']={k:save('output_'+k,out[k] if k=='actions' else out[k][0]) for k in ('actions','mean','var')}
        case['outputs']['costs']=save('output_costs',np.asarray(out['costs'],np.float32))
        case['callback_output']=out['callbacks'][TAG]
    finally:
        for h in handles:h.remove()
        case['hooks_removed']=len(handles);case['bridge_receipt']=facade.receipt()


def run_cuda(manifest,h,output,*,main_authorized=False,main_v3_allfour_approved=False):
    need(main_authorized is True and main_v3_allfour_approved is True,'MAIN authorization AND frozen v3 allfour approval required before reads/imports')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','original deterministic CUBLAS configuration')
    start=time.monotonic();m=verify(manifest,h);store=Store(output,start)
    r=dict(schema=TAG+'.report',manifest_path=str(Path(manifest).resolve()),manifest_sha256=h,protocol=protocol(),
        sources=m['sources'],main_v3_allfour_approved=True,cases=[dict(batch=b,status='not_started') for b in (5,2)],
        arrays=store.arrays,errors=[],completed=False,model_before=None,model_after=None,construction=None,runtime=None)
    root=None
    try:
        t,CEM,Box=seam.load_runtime();t.use_deterministic_algorithms(True)
        t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
        a=m['authority']['export']
        root,r['construction']=diag.loader.construct_cuda(a['export_report'],a['export_report_sha256'],
            a['export_receipt'],a['export_receipt_sha256'],main_authorized=True)
        r['model_before']=diag.old.weights(root)
        need(r['model_before']==r['construction']['tensor_value_hashes'],'all original export tensors before CEM')
        r['runtime']=dict(device=str(next(root.parameters()).device),torch=str(t.__version__),cuda=t.version.cuda,
            deterministic=t.are_deterministic_algorithms_enabled(),tf32_matmul=False,tf32_cudnn=False,autocast=False)
        with t.inference_mode(),t.autocast('cuda',enabled=False):
            r['precision_before']=precision(t)
            for c in r['cases']:
                c['status']='attempted';run_case(t,CEM,Box,root,c['batch'],store,c);c['status']='returned'
            r['precision_after']=precision(t)
        r['model_after']=diag.old.weights(root);need(r['model_before']==r['model_after'],'all weights/buffers unchanged')
        need(verify(manifest,h)==m,'final sources/export/diagnostic unchanged');store.check();r['completed']=True
    except BaseException as e:
        r['errors'].append(dict(type=type(e).__name__,message=str(e)[:2000]))
    finally:
        if root is not None and r['model_after'] is None:
            try:r['model_after']=diag.old.weights(root)
            except BaseException as e:r['errors'].append(dict(type=type(e).__name__,message=str(e)[:2000]))
        r.update(resources=resources(r['cases']),elapsed_seconds=time.monotonic()-start,
            baseline_runnable=False,official_parity=False,scientific_admission=False)
        store.put('report.json',encode(r),final=True)
    need(r['completed'] and not r['errors'],'partial run retained, no retry/refund')
    return dict(report=str(store.path/'report.json'),sha256=digest((store.path/'report.json').read_bytes()))


def exact(a,b,message):
    need(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),message)


def check_cost(actual,terminal,goal):
    """No neural recompute: bound f32 (subtract, square,192-term sum).

    Three relative rounding factors per squared residual, at most191 for a
    sequential sum: gamma194 bounds any shorter pairwise reduction as well.
    Add conservative gradual-underflow absolute error; no fitted tolerance.
    """
    need(actual.dtype==terminal.dtype==goal.dtype==np.float32 and terminal.shape==goal.shape
        and terminal.shape[-1]==192 and actual.shape==terminal.shape[:-1],'native cost f32 ABI')
    expected=np.square(terminal.astype(np.float64)-goal.astype(np.float64)).sum(axis=-1)
    u=np.finfo(np.float32).eps/2;k=194;gamma=k*u/(1-k*u)
    bound=gamma*expected+k*float(np.nextafter(np.float32(0),np.float32(1)))
    need(np.isfinite(actual).all() and np.isfinite(expected).all()
        and np.all(np.abs(actual.astype(np.float64)-expected)<=bound)
        and np.all(actual[expected==0]==0),'terminal-goal native cost arithmetic bound (not neural parity)')


def checked_layout(layout,a,device):
    need(layout['shape']==list(a.shape) and layout['dtype']=='torch.float32' and layout['device']==device
        and type(layout['storage_offset']) is int and layout['storage_offset']>=0
        and len(layout['stride'])==a.ndim and all(type(x) is int and x>=0 for x in layout['stride']),
        'actual native layout/device/dtype')
    stride=1;contiguous=True
    for size,s in zip(a.shape[::-1],layout['stride'][::-1]):
        if size>1:contiguous &= s==stride;stride*=size
    need(layout['contiguous'] is contiguous,'native contiguity arithmetic')


def validate_native(case,attempt,candidates,z_init,arrays,device):
    index=attempt['index'];n=candidates.shape[0];bs=n*8
    events=[e for e in case['native'] if e['cost_index']==index]
    need(len(events)==24,'four native high steps, three module axes attempt+return')
    history=z_init.reshape(bs,1,192);actions=candidates.reshape(n,8,4,32)
    for step in range(4):
        values={}
        for j,kind in enumerate(KINDS):
            pre,post=events[step*6+j*2:step*6+j*2+2]
            need(pre['phase']=='attempt' and post['phase']=='return' and pre['kind']==post['kind']==kind
                and pre['step']==post['step']==step
                and {k:v for k,v in post.items() if k not in ('output','output_layout')}==dict(pre,phase='return'),
                'native paired full chronology')
            ins=[arrays[f] for f in pre['inputs']];out=arrays[post['output']]
            need(all(a.dtype==np.float32 and np.isfinite(a).all() for a in ins+[out]),'native high float32 buffers')
            need(len(pre['input_layouts'])==len(ins),'every native input layout')
            for l,a in zip(pre['input_layouts'],ins,strict=True):checked_layout(l,a,device)
            checked_layout(post['output_layout'],out,device)
            want_batch=bs*(step+1) if kind=='high_pred_proj' else bs
            want_context=1 if kind=='high_pred_proj' else step+1
            need(pre['batch']==want_batch and pre['context']==want_context,'actual module sample/context axes')
            values[kind]=(ins,out)
        mi,mo=values['macro_to_condition'];hi,ho=values['high_predictor'];pi,po=values['high_pred_proj']
        need(len(mi)==len(pi)==1 and len(hi)==2 and mo.shape==ho.shape==(bs,step+1,192)
            and po.shape==(bs*(step+1),192),'native chain full shapes')
        sequence=np.stack([actions[:,:,k].reshape(bs,32) for k in range(step+1)],axis=1)
        exact(mi[0],sequence,'original source macro assembly; H2 is four macros, not child signals')
        exact(hi[0],history,'actual autoregressive high history')
        exact(hi[1],mo,'macro projection -> native high condition')
        exact(pi[0],ho.reshape(-1,192),'native predictor -> projection input')
        history=np.concatenate((history,po.reshape(bs,step+1,192)[:,-1:]),axis=1)
    # Neural outputs are NOT recomputed/tolerance-certified by this NumPy path.
    return history[:,-1].reshape(n,8,192)


def validate_case(case,arrays,device):
    p=protocol();batch=case['batch'];need(type(batch) is int and batch in (5,2),'fixed batch')
    count=((batch+1)//2)*3
    need(case['status']=='returned' and len(case['events'])==len(case['cost_attempts'])==count
        and (case['resets'],case['starts'],case['ends'])==(1,(batch+1)//2,1)
        and not case['blocked'] and case['hooks_removed']>=6,'all chunks/tail/iterations and native cleanup')
    initial=arrays[case['initial']];exact(initial,np.zeros((batch,2,64),np.float32),'full zero initial no actor')
    receipt=case['bridge_receipt'];witness=receipt['input_witness']
    need(receipt['schema']==seam.bridge_module().TAG and receipt['backend']=='actual_torch_CUDA_float32'
        and receipt['tensor_batch']==batch and receipt['cost_attempts']==receipt['cost_returns']==count
        and receipt['cost_failures']==0 and receipt['candidate_rows_attempted']==batch*8*3
        and len(receipt['calls'])==count and receipt['predict_API_calls'] is receipt['predictor_samples'] is None
        and receipt['scientific_admission'] is receipt['official_unadapted_reproduction'] is False,'complete bridge counters, axes not invented')
    need(witness['original_keys']==['planner_level','z_init','z_goal'] and witness['original_metadata']=={'planner_level':'high'}
        and witness['solver_keys']==['z_init','z_goal'] and witness['original_tensors']==witness['solver_tensors'],
        'original high metadata retained, tensors only sliced by CEM')
    for k,a in inputs(batch).items():seam.check_fingerprint(witness['original_tensors'][k],a)
    final_mean=[];final_std=[];final_cost=[];history=[]
    for index,(event,attempt,call) in enumerate(zip(case['events'],case['cost_attempts'],receipt['calls'],strict=True)):
        chunk,step=divmod(index,3);n=min(2,batch-chunk*2)
        need((event['chunk'],event['step'],event['attempt'])==(chunk,step,index)
            and (attempt['index'],attempt['chunk'])==(index,chunk) and attempt['returned'] is True
            and attempt['candidate_rows']==n*8 and call['attempt']==index+1 and call['batch']==n
            and call['candidate_rows']==n*8 and call['metadata']=='high' and call['returned'] is True,
            'complete cost/tail/iteration native lineage')
        need(set(event['arrays'])==set(seam.KEYS) and set(event['cuda_exact_checks'])==set(CHECKS)
            and all(v is True for v in event['cuda_exact_checks'].values()),'all native CUDA CEM checks')
        x={k:arrays[f] for k,f in event['arrays'].items()}
        shapes=dict(candidates=(n,8,2,64),costs=(n,8),topk_vals=(n,3),topk_inds=(n,3),topk_candidates=(n,3,2,64),
            **{k:(n,2,64) for k in ('mean','var','prev_mean','prev_var')})
        need(all(a.shape==shapes[k] and a.dtype==(np.int64 if k=='topk_inds' else np.float32) for k,a in x.items()),'saved CEM native ABI')
        c=x['candidates'];exact(arrays[attempt['candidates']],c,'root cost candidate -> callback bits')
        exact(arrays[attempt['cost']],x['costs'],'actual full root neural cost -> callback bits')
        seam.check_fingerprint(call['candidates'],c);seam.check_fingerprint(call['cost'],x['costs'])
        for k,a in inputs(batch).items():
            expected=np.repeat(a[chunk*2:chunk*2+n,None],8,axis=1);actual=arrays[attempt[k]]
            exact(actual,expected,'original row and tail expanded source bits');seam.check_fingerprint(call['expanded'][k],actual,expanded=True)
        terminal=validate_native(case,attempt,c,arrays[attempt['z_init']],arrays,device)
        need(attempt['native_cost_reduction_exact'] is True,'native CUDA cost reduction exact check')
        check_cost(x['costs'],terminal,arrays[attempt['z_goal']])
        inds=x['topk_inds'];need(((inds>=0)&(inds<8)).all() and all(len(set(row))==3 for row in inds),'distinct valid elites')
        ix=np.arange(n)[:,None];elites=c[ix,inds]
        exact(elites,x['topk_candidates'],'elite gather');exact(x['costs'][ix,inds],x['topk_vals'],'elite actual costs')
        exact(np.sort(x['costs'],axis=1)[:,:3],x['topk_vals'],'topk smallest, ties not respecified')
        seam.close_float(x['mean'],elites.astype(np.float64).mean(axis=1),'CEM mean rounding')
        seam.close_float(x['var'],elites.astype(np.float64).std(axis=1,ddof=1),'CEM SAMPLE std, not variance')
        if step==0:
            exact(x['prev_mean'],np.zeros((n,2,64),np.float32),'zero chunk initial')
            exact(x['prev_var'],np.ones((n,2,64),np.float32),'original initial std')
        else:
            previous=case['events'][index-1]['arrays']
            exact(x['prev_mean'],arrays[previous['mean']],'mean update chain');exact(x['prev_var'],arrays[previous['var']],'std update chain')
        exact(c[:,0],x['prev_mean'],'first sample mean')
        history.append(dict(chunk=chunk,step=step))
        if step==2:final_mean.append(x['mean']);final_std.append(x['var']);final_cost.append(x['topk_vals'].astype(np.float64).mean(axis=1))
    need(case['callback_output']==history and set(case['outputs'])=={'actions','mean','var','costs'},'complete callback and output')
    for k,a in (('actions',np.concatenate(final_mean)),('mean',np.concatenate(final_mean)),('var',np.concatenate(final_std))):
        exact(arrays[case['outputs'][k]],a,'final CEM '+k)
    seam.close_float(arrays[case['outputs']['costs']],np.concatenate(final_cost),'elite average logging rounding')
    need(len(case['native'])==count*24,'no unknown/extra native events')


def consume(manifest,h,report_path,rh):
    m=verify(manifest,h);p=Path(report_path);r=read(p,rh)
    need(r['schema']==TAG+'.report' and r['manifest_path']==str(Path(manifest).resolve()) and r['manifest_sha256']==h
        and r['protocol']==protocol() and r['sources']==m['sources'] and r['completed'] is True and not r['errors']
        and r['main_v3_allfour_approved'] is True and 0<=r['elapsed_seconds']<=WALL,'complete MAIN-authorized integration')
    need([c['batch'] for c in r['cases']]==[5,2] and all(r[k] is False for k in
        ('baseline_runnable','official_parity','scientific_admission')),'all2 denominator; no scientific release')
    need(r['model_before']==r['model_after']==r['construction']['tensor_value_hashes']
        and r['construction']['all_parameters_frozen'] is True
        and r['construction']['tensor_copy_attempts']==r['construction']['tensor_copy_returns']==425,'all export tensors unchanged')
    a=m['authority']['export']
    _,spec,_,export=diag.loader.authorities(a['export_report'],a['export_report_sha256'],a['export_receipt'],a['export_receipt_sha256'])
    expected={e['key']:row['array']['value_sha256'] for row,e in zip(export['rows'],spec['entries'],strict=True) if row['array'] is not None}
    need(r['model_before']==expected and r['construction']['export_report_sha256']==a['export_report_sha256']
        and r['construction']['export_receipt_sha256']==a['export_receipt_sha256']
        and r['construction']['ir_sha256']==diag.loader.IR_SHA,'original export/IR exact byte identity')
    device=r['runtime']['device'];need(device.startswith('cuda:') and r['construction']['device']==device
        and r['runtime']['torch']==r['construction']['torch'] and r['runtime']['deterministic'] is True
        and r['runtime']['tf32_matmul'] is r['runtime']['tf32_cudnn'] is r['runtime']['autocast'] is False,'original precision CUDA no CPU fallback')
    need(r['precision_before']==r['precision_after']==dict(deterministic=True,tf32_matmul=False,tf32_cudnn=False,autocast=False,grad=False),
        'actual precision guards before/after all solves')
    files=list(p.parent.iterdir());need(all(f.is_file() and not f.is_symlink() for f in files)
        and {f.name for f in files}==set(r['arrays'])|{p.name} and sum(f.stat().st_size for f in files)<=CAP,'whole artifact census/cap')
    arrays={}
    for name,s in r['arrays'].items():
        need(Path(name).name==name and name.endswith('.npy'),'flat NPY path')
        body=(p.parent/name).read_bytes();need(len(body)==s['file_bytes'] and digest(body)==s['file_sha256'],'all saved NPY bytes')
        stream=io.BytesIO(body);a=np.load(stream,allow_pickle=False)
        need(stream.tell()==len(body) and a.dtype in (np.dtype('<f4'),np.dtype('<i8')) and np.isfinite(a).all()
            and list(a.shape)==s['shape'] and a.dtype.str==s['dtype'] and digest(a.tobytes())==s['data_sha256'],'full native array payload no trailing')
        arrays[name]=a
    for c in r['cases']:validate_case(c,arrays,device)
    used=set()
    def refs(v):
        if isinstance(v,dict):
            for x in v.values():refs(x)
        elif isinstance(v,list):
            for x in v:refs(x)
        elif isinstance(v,str) and v.endswith('.npy'):used.add(v)
    refs(r['cases']);need(used==set(arrays),'all referenced arrays, none omitted/extra')
    actual=resources(r['cases']);need(actual==r['resources'] and actual['cost']==dict(attempted=12,returned=12)
        and actual['bridge_candidate_rows']==168 and actual['modules']['high_predictor']==dict(attempted=48,returned=48,sample_rows=672,context_tokens=1680),
        'separate real module/sample/context/cost resources')
    need(verify(manifest,h)==m and digest(p.read_bytes())==rh,'sources/weights/export/report final closure')
    return dict(schema=TAG+'.consume',status='real_high_CEM_integration_saved_verified_NOT_baseline',report_sha256=rh,
        manifest_sha256=h,resources=actual,batches=[5,2],v2_diagnostic_only=True,neural_parity_tolerance_used=False,
        fullbase_certificate=False,baseline_runnable=False,official_parity=False,scientific_admission=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('manifest','manifest-sha256','output','report','report-sha256','receipt'):p.add_argument('--'+k)
    p.add_argument('--main-authorized',action='store_true');p.add_argument('--main-v3-allfour-approved',action='store_true');a=p.parse_args(argv)
    if a.command=='prepare':v=prepare(a.output)
    elif a.command=='verify':verify(a.manifest,a.manifest_sha256);v=dict(status='model_free_prepare_verified_not_authorized')
    elif a.command=='run':v=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized,main_v3_allfour_approved=a.main_v3_allfour_approved)
    else:
        v=consume(a.manifest,a.manifest_sha256,a.report,a.report_sha256)
        target=scoped(a.receipt);write_new(target,encode(v))
    print(encode(v).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())
