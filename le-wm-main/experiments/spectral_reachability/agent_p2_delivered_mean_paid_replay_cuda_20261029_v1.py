"""Fixed delivered means: paid CUDA actual-token replay, never raw release."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace

import numpy as np
from experiments.spectral_reachability import agent_p2_delivered_mean_raw_domain_20261029_v1 as raw

TAG='agent_p2_delivered_mean_paid_replay_cuda_20261029_v1'
ROOT=raw.ROOT
controller=raw.controller
tracker=controller.tracker
core=tracker.core
low=core.low
need,sha,read,encode=raw.need,raw.sha,raw.read,raw.encode
ACTIVE=controller.ACTIVE
AUTH=ROOT/'results/agent_p2_delivered_mean_raw_domain_20261029_v1_prepare_MAIN01.json'
AUTH_SHA='9df31325d7d2bf149b60db0f6d21ce40e6284060c4bb3af29c1699e1ee056c56'
CAP,RESERVE,FLOOR,WALL,SLOT_WALL=512*1024**2,64*1024**2,200*1024**2,900,180
RAW_PINS={
    Path(raw.__file__):'3d5fa8ac4d0561dd8775105399ce46e7d0f01c92760b6058267c48a28d7dfb88',
    ROOT/'tests/test_agent_p2_delivered_mean_raw_domain_20261029_v1.py':'b71e4194324837cee10d514731e66f6b2d5b04f4d0a3fbcdb889b6e2458f7cf3',
    ROOT/'innerdocs/agent_p2_delivered_mean_raw_domain_20261029_v1.md':'1fc0dfb41278b62181ea1eee8a74f84485433efd3f3ba3da62b649e64487b27e'}


def protocol():
    return dict(schema=TAG+'.protocol',active=list(ACTIVE),missing=[2,5],denominator=6,
        H=3,S=1,segments=[1,2],search=False,reselection=False,clip=False,tanh=False,
        arithmetic=['mul_float32','add_float32','sub_float32','div_float32'],
        raw_identity='120 bytes including signed zero',score_after='all_predictions_durably_sealed',
        objective='identity192_terminal_meanMSE_float64',
        planned=dict(predict_API=24,predictor_samples=24,action_encoder=24,low_projection=24,
                     AE_tokens=96,predictor_context_tokens=72,projection_rows=72),
        encoder=0,high=0,AD_primal=0,J=0,VJP=0,backward=0,physics=0,FLOPs=None,
        output_cap=CAP,reserve=RESERVE,floor=FLOOR,wall=WALL,per_slot_wall=SLOT_WALL,
        trace_payload_cap=64*1024**2,cuda_allocated=3*1024**3,cuda_reserved=4*1024**3,
        cost_reduction_bound='8*192*eps64*max(abs(fsum_reference),tiny64)',
        original_prediction_cost_reuse_authorized=False,raw_deployment_authorized=False,
        original_Hi_scaler_certified=False,scientific_admission=False,fresh=False,
        fair_comparison=False,prior_physical_time=None)


def closure():
    out=controller.closure()
    for p,h in RAW_PINS.items():need(sha(p)==h,'frozen raw triple');out[str(p)]=h
    for folder,name in (('experiments/spectral_reachability',TAG+'.py'),('tests','test_'+TAG+'.py'),('innerdocs',TAG+'.md')):
        p=ROOT/folder/name;out[str(p)]=sha(p)
    return out


def payloads(d,cm):
    values=controller.select_payloads(cm['delivery_artifact'])
    need(len(d['rows'])==6 and d['denominator']==dict(total=6,checked=4,missing=2),'all six raw checks')
    for i,row in enumerate(d['rows']):
        need((row['slot'],row['root'],row['arm'])==(i,i//3,controller.ARMS[i%3]),'fixed six slot identities')
        if i not in ACTIVE:
            need(row['status']=='missing' and row['checks'] is None,'arbitrary remains missing');continue
        checks=row['checks']
        mean=np.frombuffer(bytes.fromhex(checks['array_bits']['normalized']),'<f4').copy().reshape(1,3,10)
        mean_xy=np.frombuffer(bytes.fromhex(d['normalizer']['mean_bits']),'<f4').copy()
        std=np.frombuffer(bytes.fromhex(d['normalizer']['std_bits']),'<f4').copy()
        need(raw.check_mean(mean,mean_xy,std)==checks,'entire sealed inverse equation/coordinates')
        need(core.digest(mean.tobytes())==row['mean_source']['normalized_value_sha256'],'original delivered mean identity')
        sealed=np.frombuffer(bytes.fromhex(checks['array_bits']['raw']),'<f4').copy().reshape(15,2)
        values[i].update(mean=mean,mean_xy=mean_xy,std=std,sealed_raw=sealed)
    return values


def build():
    d=read(AUTH,AUTH_SHA);need(d==raw.build(),'whole actual saved raw-domain artifact; no full strict')
    a=d['authorities']['prepare'];cm=read(a['path'],a['sha256']);payloads(d,cm)
    sources=dict(cm['sources'])
    for p,h in {**d['sources'],**closure(),str(AUTH):AUTH_SHA}.items():
        need(p not in sources or sources[p]==h,'consistent entire closure');sources[p]=h
    need(all(sha(p)==h for p,h in sources.items()),'all source/model/input bytes')
    return dict(schema=TAG+'.prepare',protocol=protocol(),authority=dict(path=str(AUTH),sha256=AUTH_SHA),
        raw_artifact=d,controller_prepare=cm,export=cm['export'],expected_state=cm['expected_state'],
        expected_construction=cm['expected_construction'],sources=sources)


def own_output(path):
    p=Path(path);need(p.parent.resolve()==ROOT/'results' and p.name.startswith(TAG+'_') and not p.is_symlink(),'exclusive own results child');return p


def prepare(output):
    m=build();b=encode(m);p=own_output(output)
    need(len(b)<=32*1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'bounded prepare/floor')
    tracker.durable(p,b);return dict(path=str(p.resolve()),sha256=sha(p))


def verify(path,h):
    m=read(path,h);need(m==build(),'entire prepare reconstructed');return m


class Store(tracker.Store):
    def __init__(self,path):
        self.path=own_output(path);self.start=time.monotonic();self.used=0;self.root_used={0:0,1:0}
        self.files={};self.journal=[]
        need(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'exclusive new output/floor')
        self.path.mkdir()

    def guard(self,n=0,root=None,footer=False,final=False):
        need(final or footer or time.monotonic()-self.start<=WALL,'whole900s cooperative guard')
        need(self.used+n<=CAP-(0 if final or footer else RESERVE),'512MiB incl64MiB footer')
        need(shutil.disk_usage(self.path).free>=FLOOR+n,'floor200MiB')
        need(root is None or root in (0,1),'fixed root accounting')


class Trace(core.Trace):
    """Reuse pure capture/debit arithmetic, NOT CEM execution/spec/consumer."""
    def __init__(self,capture,backend):
        self.spec=SimpleNamespace(wall_seconds=SLOT_WALL,array_bytes_cap=64*1024**2,predict_api_cap=6,predict_sample_cap=6)
        self.backend=backend;self.capture=capture;self.started=time.monotonic()
        self.arrays={};self.used=0;self.capture_wall=0.;self.events=[];self.api=self.samples=0
        self.report=dict(schema=TAG+'.trace',backend=backend,completed=False,errors=[],events=self.events,
                         arithmetic=[],replays=[],blocked=None)

    def call(self,kind,args,fn):
        layout=getattr(self,'layout',None);index=len(self.events)
        try:
            out=super().call(kind,args,fn)
            if layout is not None:self.events[index]['output_layout']=layout(out)
            return out
        finally:
            if layout is not None and len(self.events)>index:
                self.events[index]['input_layouts']=[layout(x) for x in args]


def arithmetic(tr,inputs,ops):
    tr.report['inputs']={k:tr.save(v) for k,v in inputs.items()}
    mean=inputs['mean'].reshape(15,2)
    def call(kind,args,fn):
        e=dict(kind=kind,inputs=[tr.save(a) for a in args],returned=False,
               layouts=[ops.layout(a) for a in args]);tr.report['arithmetic'].append(e)
        out=fn();e.update(output=tr.save(out),layout=ops.layout(out),returned=True);return out
    product=call('mul',[mean,inputs['std']],lambda:ops.mul(mean,inputs['std']))
    actual_raw=call('add',[product,inputs['mean_xy']],lambda:ops.add(product,inputs['mean_xy']))
    a=tr.capture(actual_raw);sealed=tr.capture(inputs['sealed_raw'])
    tr.report['raw_byte_identity']=core.exact(a,sealed)
    tr.report['domain_ok']=bool(np.isfinite(a).all() and (np.abs(a)<=1).all())
    if not tr.report['domain_ok']:tr.report['blocked']='blocked_native_domain';return None
    if not tr.report['raw_byte_identity']:tr.report['blocked']='blocked_raw_byte_identity';return None
    centered=call('sub',[actual_raw,inputs['mean_xy']],lambda:ops.sub(actual_raw,inputs['mean_xy']))
    tokens=call('div',[centered,inputs['std']],lambda:ops.div(centered,inputs['std']))
    tr.report['actual_tokens']=tr.save(tokens.reshape(1,1,3,10))
    return tokens.reshape(1,1,3,10)


def execute(inputs,kernels,ops,backend):
    """No scoring here: all slots must first pass a durable prediction barrier."""
    tr=Trace(ops.capture,backend)
    tr.layout=ops.layout
    try:
        future=arithmetic(tr,inputs,ops)
        if future is not None:
            k=core.traced_kernels(kernels,tr)
            z,a=inputs['z'][:,None],inputs['a'][:,None]
            for segmented in (False,True):
                row=dict(kind='segmented' if segmented else 'whole',first_event=len(tr.events),returned=False)
                tr.report['replays'].append(row)
                r=(low.segmented_rollout(z,a,future,segments=(1,2),kernels=k) if segmented else low.rollout(z,a,future,kernels=k))
                row.update(rollout=core.save_roll(tr,r),end_event=len(tr.events),returned=True)
            low.check_replay(rwhole := saved_roll(tr,0),saved_roll(tr,1),exact=core.exact)
            need(rwhole.prediction.shape==(1,1,3,192),'complete H3 prediction')
        for key,v in inputs.items():need(core.exact(tr.arrays[tr.report['inputs'][key]],ops.capture(v)),'no mutation of input')
        tr.guard();tr.report['completed']=True
    except BaseException as ex:tr.report['errors'].append(dict(type=type(ex).__name__,message=str(ex)[:3000]))
    return tr.finish()


def saved_roll(tr,index):
    d=tr.report['replays'][index]['rollout'];a=tr.arrays
    return low.Rollout(a[d['prediction']],low.FlatContext(**{k:a[v] for k,v in d['final'].items()}),
        tuple(low.FlatContext(**{k:a[v] for k,v in c.items()}) for c in d['checkpoints']))


def token_differences(old,new):
    need(old.shape==new.shape==(15,2) and old.dtype==new.dtype==np.float32,'token comparison ABI')
    rows=[]
    for j,(x,y) in enumerate(zip(old.flat,new.flat,strict=True)):
        u,v=int(raw.bits(np.asarray(x)).item()),int(raw.bits(np.asarray(y)).item())
        rows.append(dict(step=j//2+1,axis=j%2,before_bits=raw.bithex(u),after_bits=raw.bithex(v),
            bitexact=u==v,ULP=abs(raw.ordered(u)-raw.ordered(v)),
            zero_signs=[bool(np.signbit(q)) if q==0 else None for q in (x,y)]))
    return rows


def equal(a,b,message):need(core.exact(a,b),message)


def validate_trace(tr,arrays,inputs,*,host=True):
    need(tr['schema']==TAG+'.trace' and tr['backend']==('host_cuda_actual_tokens' if host else 'ram_fixture') and
         tr['completed'] is True and tr['errors']==[],'complete actual backend trace')
    need(tr['arrays']=={k:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=core.digest(a.tobytes())) for k,a in arrays.items()},'all trace array metadata')
    need(tr['array_payload_bytes']==sum(a.nbytes for a in arrays.values())<=64*1024**2,'payload bytes')
    need(tr['resources']==dict(J=0,VJP=0,backward=0,physics=0,encoder=0,FLOPs=None,AD_primal=0),'no encoding/AD/physics')
    need(0<=tr['whole_core_wall_seconds']<=SLOT_WALL and 0<=tr['capture_sync_wall_seconds']<=tr['whole_core_wall_seconds'],'trace wall scopes')
    A=lambda k:arrays[k]
    def layout(d,value):
        need(d['shape']==list(value.shape) and len(d['stride'])==value.ndim and
             all(type(v) is int and v>=0 for v in d['stride']) and d['requires_grad'] is False and
             d['dtype']==('torch.float32' if host else 'numpy.float32') and
             d['device']==('cuda:0' if host else 'ram_fixture'),'actual layout/device/nonAD witness')
    for e in tr['arithmetic']:
        need(len(e['layouts'])==len(e['inputs']),'all arithmetic layouts')
        for d,key in zip(e['layouts'],e['inputs'],strict=True):layout(d,A(key))
        layout(e['layout'],A(e['output']))
    need(set(tr['inputs'])==set(inputs),'all sealed input keys')
    for k,v in inputs.items():equal(A(tr['inputs'][k]),v,'input identity '+k)
    ar=tr['arithmetic'];need(len(ar) in (2,4),'arithmetic operation count')
    norm=inputs['mean'].reshape(15,2);product=np.multiply(norm,inputs['std'],dtype=np.float32)
    rr=np.add(product,inputs['mean_xy'],dtype=np.float32)
    for e,kind,args,result in zip(ar[:2],('mul','add'),((norm,inputs['std']),(product,inputs['mean_xy'])),(product,rr),strict=True):
        need(e['kind']==kind and e['returned'] is True and len(e['inputs'])==2,'separate CUDA inverse')
        for key,value in zip(e['inputs'],args,strict=True):equal(A(key),value,'inverse input')
        equal(A(e['output']),result,'exact sealed inverse arithmetic')
    domain=bool(np.isfinite(rr).all() and (np.abs(rr)<=1).all());identity=core.exact(rr,inputs['sealed_raw'])
    need(tr['domain_ok']==domain and tr['raw_byte_identity']==identity,'domain/raw bit guards recomputed')
    blocked='blocked_native_domain' if not domain else 'blocked_raw_byte_identity' if not identity else None
    need(tr['blocked']==blocked,'blocked status not fallback')
    if blocked:
        need(len(ar)==2 and tr['events']==tr['replays']==[] and tr['predict_API_debited']==tr['predictor_samples_debited']==0,'blocked no model work')
        need(all(all(v==0 for v in d.values()) for d in tr['modules'].values()),'blocked zero module work')
        return dict(blocked=blocked,predict_API=0,predictor_samples=0,action_encoder=0,low_projection=0,
                    AE_tokens=0,predictor_context_tokens=0,projection_rows=0)
    need(len(ar)==4 and [x['kind'] for x in ar]==['mul','add','sub','div'],'actual normalize operations')
    centered=np.subtract(rr,inputs['mean_xy'],dtype=np.float32)
    equal(A(ar[2]['inputs'][0]),rr,'sub raw');equal(A(ar[2]['inputs'][1]),inputs['mean_xy'],'sub stats')
    equal(A(ar[2]['output']),centered,'sub exact f32')
    equal(A(ar[3]['inputs'][0]),centered,'div centered');equal(A(ar[3]['inputs'][1]),inputs['std'],'div stats')
    # CUDA division is authenticated saved work, not assumed equal to NumPy division.
    tokens=A(tr['actual_tokens']);equal(tokens,A(ar[3]['output']).reshape(1,1,3,10),'actual div delivery')
    need(tokens.shape==(1,1,3,10) and tokens.dtype==np.float32 and np.isfinite(tokens).all(),'actual finite grouped tokens')
    need(all(e['returned'] for e in ar),'all arithmetic returns')
    events=tr['events'];need(len(events)==18 and len(tr['replays'])==2,'full replay event denominator')
    cursor=0
    for ri,replay in enumerate(tr['replays']):
        need(replay['kind']==('whole','segmented')[ri] and replay['first_event']==cursor and replay['returned'],'paid replay order')
        z=inputs['z'].copy();a=inputs['a'].copy();d=replay['rollout'];ys=[]
        need(len(d['checkpoints'])==3,'all H3 checkpoints')
        for step,checkpoint in enumerate(d['checkpoints']):
            es=events[cursor:cursor+3]
            need([e['kind'] for e in es]==['action_encoder','low_predictor','low_pred_proj'],'source module order')
            for j,e in enumerate(es):
                need(e['index']==cursor+j and all(e[k] is True for k in ('dispatched','module_returned','returned')),'actual module return')
                need(e['shapes']==[list(A(v).shape) for v in e['inputs']],'module input shape ledger')
                need(e['budget_debited']==(e['kind']=='low_predictor'),'predict debit before dispatch')
                need(len(e['input_layouts'])==len(e['inputs']),'full module input layouts')
                for d0,key in zip(e['input_layouts'],e['inputs'],strict=True):layout(d0,A(key))
                layout(e['output_layout'],A(e['output']))
            need([len(e['inputs']) for e in es]==[1,2,1],'module arity')
            a=np.concatenate((a,tokens[:,:,step,:].reshape(1,1,10)),axis=1)
            equal(A(es[0]['inputs'][0]),a,'full causal AE history')
            emb=A(es[0]['output']);need(emb.shape==(1,step+3,192) and emb.dtype==np.float32,'full AE output')
            equal(A(es[1]['inputs'][0]),z[:,-3:],'predict z last3');equal(A(es[1]['inputs'][1]),emb[:,-3:],'predict a last3')
            hidden=A(es[1]['output']);need(hidden.shape==(1,3,192) and hidden.dtype==np.float32,'all predictor slots')
            equal(A(es[2]['inputs'][0]),hidden.reshape(3,192),'flatten low projection')
            p=A(es[2]['output']);need(p.shape==(3,192) and p.dtype==np.float32,'all projection slots')
            y=p.reshape(1,3,192)[:,-1:];z=np.concatenate((z,y),axis=1);ys.append(y[:,0])
            equal(A(checkpoint['z']),z,'full causal z');equal(A(checkpoint['a']),a,'full causal a');cursor+=3
        equal(A(d['prediction']),np.stack(ys,axis=1).reshape(1,1,3,192),'full prediction')
        equal(A(d['final']['z']),z,'final z');equal(A(d['final']['a']),a,'final a')
        need(replay['end_event']==cursor,'replay end event')
    fake=SimpleNamespace(report=tr,arrays=arrays)
    low.check_replay(saved_roll(fake,0),saved_roll(fake,1),exact=core.exact)
    need(tr['predict_API_debited']==tr['predictor_samples_debited']==6,'six paid predict calls/samples')
    for kind,rows,tokens_count in (('action_encoder',6,24),('low_predictor',6,18),('low_pred_proj',18,0)):
        need(tr['modules'][kind]==dict(attempts=6,dispatched=6,module_returns=6,captured_returns=6,rows_dispatched=rows,tokens_dispatched=tokens_count),'strict full-history work census')
    return dict(blocked=None,predict_API=6,predictor_samples=6,action_encoder=6,low_projection=6,
        AE_tokens=24,predictor_context_tokens=18,projection_rows=18,
        old_to_actual=token_differences(norm,tokens.reshape(15,2)),
        numpy_roundtrip_to_actual=token_differences(np.divide(centered,inputs['std'],dtype=np.float32),tokens.reshape(15,2)))


def persist_trace(store,row,tr,arrays):
    # Metadata/debits first: a subsequent cap/timeout must not erase attempts.
    row['trace']=store.put(f"slot{row['slot']}_trace.json",encode(tr),row['root'],footer=True)
    row['arrays']={}
    for k,a in arrays.items():row['arrays'][k]=store.array(f"slot{row['slot']}_{k}.npy",a,row['root'])


def seal_predictions(store,rows):
    need(all(rows[i]['status'] in ('predictions_sealed','blocked') for i in ACTIVE),'no score before all four terminal predictions')
    value=dict(schema=TAG+'.prediction_seal',slots=[dict(slot=i,status=rows[i]['status'],trace=rows[i]['trace'],arrays=rows[i]['arrays']) for i in ACTIVE])
    return store.put('prediction_seal.json',encode(value))


def run_cuda(manifest,mh,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN authorization before production IO/torch')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','fixed CUBLAS workspace')
    store=Store(output);caps=tracker.GPUResourceGuard(store);graph=t=None;precision=handler=None
    r=dict(schema=TAG+'.report',manifest=str(Path(manifest).resolve()),manifest_sha256=mh,protocol=protocol(),completed=False,errors=[],
        rows=[dict(slot=i,root=i//3,arm=controller.ARMS[i%3],status='not_started' if i in ACTIVE else 'missing',
                   debits=dict(known=True,predict_API=0,predictor_samples=0)) for i in range(6)],
        files=store.files,write_journal=store.journal,construction=None,model_before=None,model_after=None,sources_after=None,
        prediction_seal=None,raw_deployment_authorized=False,scientific_admission=False)
    def timeout(signum,frame):raise TimeoutError('whole900s alarm; no CUDA kernel preemption')
    try:
        need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'standalone no existing alarm')
        handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,WALL)
        m=verify(manifest,mh);store.put('dispatch_manifest.json',encode(m));values=payloads(m['raw_artifact'],m['controller_prepare'])
        for i in ACTIVE:
            r['rows'][i]['inputs']={k:store.array(f'slot{i}_array_{9000+j}.npy',v,i//3) for j,(k,v) in enumerate(values[i].items())}
        import torch as t
        need(Path(t.__file__).resolve()==ROOT/'.venv/lib/python3.12/site-packages/torch/__init__.py' and t.cuda.is_available(),'pinned venv HOST CUDA only')
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.nn.attention import sdpa_kernel,SDPBackend
        precision=(t.are_deterministic_algorithms_enabled(),t.backends.cuda.matmul.allow_tf32,t.backends.cudnn.allow_tf32)
        t.use_deterministic_algorithms(True);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
        t.cuda.set_device(0);t.cuda.reset_peak_memory_stats(0)
        def gpu():caps.observe(int(t.cuda.max_memory_allocated(0)),int(t.cuda.max_memory_reserved(0)))
        class Bounds(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                gpu();out=func(*args,**({} if kwargs is None else kwargs));gpu();return out
        class Ops:
            mul=staticmethod(t.mul);add=staticmethod(t.add);sub=staticmethod(t.sub);div=staticmethod(t.div)
            def capture(self,x):
                need(t.is_tensor(x) and x.device==t.device('cuda:0') and not x.requires_grad,'CUDA frozen non-AD capture')
                t.cuda.synchronize();return x.detach().cpu().numpy().copy()
            def layout(self,x):return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),device=str(x.device),requires_grad=x.requires_grad)
        ops=Ops();pending={}
        with t.enable_grad(),t.autocast('cuda',enabled=False),Bounds(),sdpa_kernel(SDPBackend.MATH):
            e=m['export'];graph,r['construction']=tracker.parity.rgb.loader.construct_cuda(e['report'],e['report_sha256'],e['receipt'],e['receipt_sha256'],main_authorized=True)
            tracker.validate_construction(r['construction'],m)
            r['model_before']=tracker.state_hashes(graph,t);need(r['model_before']==m['expected_state'],'all425 before')
            facade=core.diff.wrap(graph,t,main_authorized=True);kernels=low.cuda_kernels(t,facade,main_authorized=True)
            for i in ACTIVE:
                row=r['rows'][i];start=time.monotonic();row['status']='attempted'
                # Set even before return so a preserved partial has its actual scope.
                row['slot_started_whole_seconds']=start-store.start
                row['model_before']=tracker.state_hashes(graph,t);need(row['model_before']==m['expected_state'],'slot425 before')
                xs={k:t.from_numpy(v.copy()).to('cuda:0') for k,v in values[i].items()}
                row['debits']=dict(known=False,predict_API=None,predictor_samples=None)
                tr,arrays=execute(xs,kernels,ops,'host_cuda_actual_tokens')
                row['debits']=dict(known=True,predict_API=tr['predict_API_debited'],predictor_samples=tr['predictor_samples_debited'])
                persist_trace(store,row,tr,arrays)
                row['local_check']=validate_trace(tr,arrays,values[i]);row['model_after']=tracker.state_hashes(graph,t)
                need(row['model_after']==row['model_before'],'slot425 after')
                row['wall_seconds']=time.monotonic()-start;need(row['wall_seconds']<=SLOT_WALL,'slot180s including persistence')
                row['status']='blocked' if tr['blocked'] else 'predictions_sealed';pending[i]=(tr,arrays)
            r['prediction_seal']=seal_predictions(store,r['rows'])
            # All prediction arrays and the exclusive barrier file exist BEFORE scoring.
            for i in ACTIVE:
                row=r['rows'][i]
                if row['status']=='blocked':continue
                score_start=time.monotonic()
                tr,arrays=pending[i];row['scores']=[]
                target=t.from_numpy(values[i]['target'].copy()).to('cuda:0').to(t.float64)
                for j in (0,1):
                    pred=t.from_numpy(arrays[tr['replays'][j]['rollout']['prediction']][:,:,-1,:].copy()).to('cuda:0').to(t.float64)
                    cost=(pred-target[:,None]).square().mean(dim=-1)
                    row['scores'].append(store.array(f'slot{i}_array_{8000+j}.npy',ops.capture(cost),row['root']))
                row['status']='returned'
                row['score_wall_seconds']=time.monotonic()-score_start
                row['wall_seconds']+=row['score_wall_seconds'];need(row['wall_seconds']<=SLOT_WALL,'slot180s incl score scope; excludes other slots/barrier wait')
            gpu()
        r['model_after']=tracker.state_hashes(graph,t);need(r['model_after']==r['model_before']==m['expected_state'],'all425 after')
        r['sources_after']={p:sha(p) for p in m['sources']};need(r['sources_after']==m['sources'],'all source/input after')
        store.guard();r['completed']=True
    except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)[:3000]))
    finally:
        if handler is not None:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
        if graph is not None and r['model_after'] is None:
            try:r['model_after']=tracker.state_hashes(graph,t)
            except BaseException as ex:r['errors'].append(dict(stage='final_model',message=str(ex)[:1000]))
        if 'm' in locals() and r['sources_after'] is None:
            try:r['sources_after']={p:sha(p) for p in m['sources']}
            except BaseException as ex:r['errors'].append(dict(stage='final_sources',message=str(ex)[:1000]))
        if precision is not None:
            t.use_deterministic_algorithms(precision[0]);t.backends.cuda.matmul.allow_tf32=precision[1];t.backends.cudnn.allow_tf32=precision[2]
        r['cuda_resources']=caps.snapshot();r['whole_wall_seconds']=time.monotonic()-store.start
        r['debits']=controller.totals(r['rows'])
        r['file_bytes_before_report']=store.used;r['root_file_bytes']=store.root_used
        store.put('report.json',encode(r),final=True)
    return dict(report=str((store.path/'report.json').resolve()),sha256=sha(store.path/'report.json'),completed=r['completed'])


def validate_files(r,folder):
    files=r['files'];need({p.name for p in Path(folder).iterdir()}==set(files)|{'report.json'},'exact file census')
    for name,d in files.items():
        p=Path(folder)/name
        need(Path(name).name==name and d['file']==name and sha(p)==d['file_sha256'] and p.stat().st_size==d['file_bytes'],'all files size/SHA')
    need(sum(d['file_bytes'] for d in files.values())==r['file_bytes_before_report'] and
         r['file_bytes_before_report']+(Path(folder)/'report.json').stat().st_size<=CAP,'whole512MiB')
    need(len(r['write_journal'])==len(files) and {e['name']:e for e in r['write_journal']}==
         {n:dict(name=n,root=d['root'],bytes=d['file_bytes'],returned=True) for n,d in files.items()},'exact durable journal census')
    for root in (0,1):need(sum(d['file_bytes'] for d in files.values() if d['root']==root)==r['root_file_bytes'][str(root)],'root file census')


def analyze_saved(m,r,folder):
    need(r['schema']==TAG+'.report' and r['protocol']==protocol() and r['completed'] is True and r['errors']==[],'complete actual paid replay')
    need(r['raw_deployment_authorized'] is False and r['scientific_admission'] is False,'not raw/science release')
    tracker.validate_construction(r['construction'],m)
    need(r['model_before']==r['model_after']==m['expected_state'] and r['sources_after']==m['sources'],'entire guards')
    need(0<=r['whole_wall_seconds']<=WALL,'whole wall scope')
    c=r['cuda_resources'];need(c['max_allocated']<=3*1024**3 and c['max_reserved']<=4*1024**3,'CUDA limits')
    dispatch=r['files']['dispatch_manifest.json']
    need(read(Path(folder)/dispatch['file'],dispatch['file_sha256'])==m,'whole dispatch manifest identity')
    values=payloads(m['raw_artifact'],m['controller_prepare']);need(len(r['rows'])==6,'six denominator')
    def member(d):need({k:d[k] for k in ('file','file_bytes','file_sha256','root')}==r['files'][d['file']],'descriptor inventory')
    def load(d):member(d);return controller.load_array(folder,d)
    checks=[];seal_slots=[];score_files=[];referenced={'dispatch_manifest.json','prediction_seal.json'}
    for i,row in enumerate(r['rows']):
        need((row['slot'],row['root'],row['arm'])==(i,i//3,controller.ARMS[i%3]),'fixed full slots')
        if i not in ACTIVE:
            need(row==dict(slot=i,root=i//3,arm='arbitrary',status='missing',debits=dict(known=True,predict_API=0,predictor_samples=0)),'missing no invented data');continue
        need(row['status'] in ('returned','blocked') and 0<=row['wall_seconds']<=SLOT_WALL,'slot outcome/wall')
        for d in [*row['inputs'].values(),*row['arrays'].values(),row['trace'],*row.get('scores',[])]:
            need(d['root']==i//3 and d['file'].startswith(f'slot{i}_'),'same exact slot/source, never crossbind')
            referenced.add(d['file'])
        need(row['model_before']==row['model_after']==m['expected_state'],'per-slot425 guard')
        need(set(row['inputs'])==set(values[i]),'all fixed input fields')
        for k,v in values[i].items():equal(load(row['inputs'][k]),v,'sealed runtime inputs')
        member(row['trace']);tr=read(Path(folder)/row['trace']['file'],row['trace']['file_sha256'])
        arrays={k:load(d) for k,d in row['arrays'].items()};lc=validate_trace(tr,arrays,values[i]);need(row['local_check']==lc,'full local check rebuilt')
        checks.append(dict(slot=i,**lc));blocked=lc['blocked'] is not None
        need(row['debits']==dict(known=True,predict_API=lc['predict_API'],predictor_samples=lc['predictor_samples']),'slot paid debit rebuilt')
        need((row['status']=='blocked')==blocked,'status and trace block identity')
        seal_slots.append(dict(slot=i,status='blocked' if blocked else 'predictions_sealed',trace=row['trace'],arrays=row['arrays']))
        if blocked:need('scores' not in row,'no blocked score');continue
        need(len(row['scores'])==2,'both replay costs')
        for j,d in enumerate(row['scores']):
            cost=load(d);need(cost.shape==(1,1) and cost.dtype==np.float64,'actual float64 cost')
            prediction=arrays[tr['replays'][j]['rollout']['prediction']][0,0,-1].astype(np.float64)
            delta=prediction-values[i]['target'][0].astype(np.float64)
            ref=math.fsum(float(v)*float(v) for v in delta)/192
            need(abs(float(cost[0,0])-ref)<=8*192*np.finfo(np.float64).eps*max(abs(ref),np.finfo(np.float64).tiny),'independent original target meanMSE')
            score_files.append(d['file'])
        equal(load(row['scores'][0]),load(row['scores'][1]),'whole segmented cost exact')
    member(r['prediction_seal']);seal=r['prediction_seal']
    need(read(Path(folder)/seal['file'],seal['file_sha256'])==dict(schema=TAG+'.prediction_seal',slots=seal_slots),'full all-slot prediction barrier')
    order=[e['name'] for e in r['write_journal']];barrier=order.index(seal['file'])
    need(all(order.index(d['file'])<barrier for x in seal_slots for d in [x['trace'],*x['arrays'].values()]) and
         all(order.index(n)>barrier for n in score_files),'predictions persisted before any scoring')
    need(r['debits']==controller.totals(r['rows']),'aggregate exact debits')
    need(set(r['files'])==referenced,'all files accounted, no unexplained work')
    return dict(status='all_saved_paid_replays_verified_with_explicit_blocked_slots',local_checks=checks,
        denominator=dict(total=6,returned=sum(x['blocked'] is None for x in checks),blocked=sum(x['blocked'] is not None for x in checks),missing=2),
        predict_API=sum(x['predict_API'] for x in checks),predictor_samples=sum(x['predictor_samples'] for x in checks),
        module_resources={k:sum(x[k] for x in checks) for k in ('action_encoder','low_projection','AE_tokens','predictor_context_tokens','projection_rows')},
        original_Hi_scaler_certified=False,raw_deployment_authorized=False,scientific_admission=False)


def consume(manifest,mh,report,rh,output):
    m=verify(manifest,mh);r=read(report,rh)
    need(r['manifest_sha256']==mh and Path(r['manifest']).resolve()==Path(manifest).resolve(),'exact report/manifest join')
    validate_files(r,Path(report).parent);out=analyze_saved(m,r,Path(report).parent)
    out.update(manifest_sha256=mh,report_sha256=rh)
    tracker.durable(own_output(output),encode(out));return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('protocol','prepare','verify','run','consume'))
    for key in ('manifest','manifest-sha256','report','report-sha256','output'):p.add_argument('--'+key)
    p.add_argument('--main-authorized',action='store_true');a=p.parse_args(argv)
    if a.command=='protocol':out=protocol()
    elif a.command=='prepare':out=prepare(a.output)
    elif a.command=='verify':verify(a.manifest,a.manifest_sha256);out=dict(verified=True)
    elif a.command=='run':out=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else:out=consume(a.manifest,a.manifest_sha256,a.report,a.report_sha256,a.output)
    print(encode(out).decode())


if __name__=='__main__':main()
