"""MAIN CUDA-only fixed Hi normalized low probe; worker uses saved/RAM tests.

Independent source-formula reference, no pickle execution, training, physical
actions, target selection, P2 coordinate mapping, or original-class parity.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_hilewm_normalized_transition_20261029_v1 as seam
from experiments.spectral_reachability import agent_hilewm_differentiable_low_20261029_v1 as diff

TAG='agent_hilewm_transition_cuda_probe_20261029_v2'
ROOT=Path(__file__).resolve().parents[2]
CAP,FLOOR,RESERVE,WALL=64*1024**2,200*1024**2,8*1024**2,600
SEAM_SHA='56c88a29048833546654a47257e2a34b681d103f66bfd94ccdedb0bd4763c1b2'
PARENT=ROOT/'results/agent_hilewm_low_cem_20261029_v1_prepare_MAIN01/manifest.json'
REPORT=ROOT/'results/agent_hilewm_low_cem_20261029_v1_cuda_MAIN01/report.json'
REPORT_SHA='68d23f06b88114d3f3a3df227cefe7ca247cb399ba452defbf5456dbda9f32d0'
RECEIPT=ROOT/'results/agent_hilewm_low_cem_20261029_v1_consume_MAIN01.json'
RECEIPT_SHA='96be2b84aadabc4ca23a8e84f2f54f889b86367d41d4d344e0d7d56b46e3b43e'
ROUTES=('reference','whole','segmented','fd_plus','fd_minus')
KINDS=('action_encoder','low_predictor','low_pred_proj')


def need(ok,message):
    if not ok:raise ValueError(message)


def encode(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()


def body(path,cap=CAP):
    p=Path(path);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'regular bounded file')
    b=p.read_bytes();need(len(b)<=cap,'file cap');return b


def sha(path):return digest(body(path,128*1024**2))


def read(path,h=None):
    b=body(path);need(h is None or digest(b)==h,'actual byte SHA')
    def pairs(xs):
        d={}
        for k,v in xs:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite JSON'))


def owner():
    return importlib.import_module('experiments.spectral_reachability.agent_hilewm_low_cem_20261029_v1')


def protocol():
    return dict(schema=TAG+'.protocol',root=0,condition='off',known_roots=1,fresh=False,
        initial='paid root0/off Hi z_init; single z, explicit zero incoming action history',
        B=1,S=1,H=3,segments=[1,1,1],width=10,latent=192,
        action='float32((arange(30)%11)-5)/32 reshape(1,1,3,10)',
        direction='float32((arange(30)%7)-3)/16 reshape(1,1,3,10)',
        terminal_weights='float32((arange(192)%13)-6)/128',
        scalar='float64 sum(float64(terminal)*float64(weights)); no goal/Q fit',
        forward='bitwise dtype/shape/value; exact recorded layout equality across reference/whole/segmented',
        gradient='one scalar VJP per route; complete30 float32 gradient bitwise equality across all3',
        finite_difference=dict(epsilon=2.**-8,atol=2.e-4,rtol=2.e-2,
            rule='abs(AD_dot-FD)<=atol+rtol*max(abs(AD_dot),abs(FD))',
            scope='single fixed direction central difference of float32 neural forward; no sweep/retry'),
        new_resources=dict(low_predictor=15,action_encoder=15,low_pred_proj=15,
            predictor_sample_rows=15,predictor_context_tokens=30,action_context_tokens=45,
            projection_rows=30,explicit_scalar_VJP=3,finite_difference_forward_rollouts=2,
            encoder=0,projector=0,high_predictor=0,physics=0,predict_API=0),
        AD_primal='the reference/whole/segmented nine low forward calls; inclusive, do not add again',
        internal_backward_calls=None,FLOPs=None,attempts_failed_refunded=False,
        wall_seconds=WALL,output_cap_bytes=CAP,floor_bytes=FLOOR,final_reserve_bytes=RESERVE,
        official_parity=False,P2_pairing=False,raw_inverse=False,scientific_admission=False)


def fixed_arrays():
    return dict(actions=((np.arange(30,dtype=np.float32)%11-5)/32).reshape(1,1,3,10),
        direction=((np.arange(30,dtype=np.float32)%7-3)/16).reshape(1,1,3,10),
        weights=(np.arange(192,dtype=np.float32)%13-6)/128)


def closure(low):
    seam.verify_sources();need(sha(seam.__file__)==SEAM_SHA,'frozen transition source')
    need(sha(diff.__file__)=='dd22dc0bb15adc2978e8fe385b59113d2c7e1515539e80fdfdf778229cde8904','frozen differentiable facade')
    out=low.closure();out.update(diff.source_closure())
    files=[Path(__file__).resolve(),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md'),
           Path(seam.__file__).resolve(),ROOT/'tests'/('test_'+seam.TAG+'.py'),ROOT/'innerdocs'/(seam.TAG+'.md')]
    files += [ROOT/'.venv/lib/python3.12/site-packages/torch'/x for x in
              ('autograd/__init__.py','autograd/grad_mode.py','_tensor.py','nn/functional.py','nn/modules/module.py')]
    out.update({str(p):sha(p) for p in files});return out


def build():
    low=owner();sources=closure(low)
    r=read(REPORT,REPORT_SHA);ph=r['manifest_sha256']
    need(Path(r['manifest']).resolve()==PARENT,'exact actual low manifest path')
    accepted=low.consume(PARENT,ph,REPORT,REPORT_SHA)
    need(accepted==read(RECEIPT,RECEIPT_SHA),'entire persisted actual low consume equality')
    m=low.verify(PARENT,ph);slot=m['slots'][0];row=r['rows'][0]
    need((slot['root_slot'],slot['condition'],row['root_slot'],row['condition'])==(0,'off',0,'off'),
         'fixed root0/off, no outcome selection')
    z=low.load_array(REPORT.parent,row['z_init'],r['arrays'][row['z_init']])
    need(z.dtype==np.float32 and z.shape==(1,192) and
         z.tobytes()==np.asarray(slot['buffers']['z_init']['values'],np.float32).tobytes(),'paid Hi z exact')
    export=m['export'];_,spec,_,er=low.base.loader.authorities(export['export_report'],export['export_report_sha256'],
        export['export_receipt'],export['export_receipt_sha256'])
    expected={}
    for e,rr in zip(spec['entries'],er['rows'],strict=True):
        if rr['array'] is not None:
            need(e['include_in_literal_state_dict'],'all probe weights in literal state_dict')
            expected[e['key']]=rr['array']['value_sha256']
            p=Path(export['export_report']).parent/rr['array']['file']
            sources[str(p)]=rr['array']['file_sha256']
    need(len(expected)==425,'all425 existing tensor values, no low-only subset')
    for p,h in ((PARENT,ph),(REPORT,REPORT_SHA),(RECEIPT,RECEIPT_SHA),
        (export['export_report'],export['export_report_sha256']),(export['export_receipt'],export['export_receipt_sha256'])):
        sources[str(p)]=h
    source_z=REPORT.parent/row['z_init'];sources[str(source_z)]=r['arrays'][row['z_init']]['file_sha256']
    need(all(sha(p)==h for p,h in sources.items()),'all source/input/export bytes')
    return dict(schema=TAG+'.prepare',protocol=protocol(),sources=sources,export=export,
        parent=dict(manifest=str(PARENT),manifest_sha256=ph,report=str(REPORT),report_sha256=REPORT_SHA,
            receipt=str(RECEIPT),receipt_sha256=RECEIPT_SHA,full_strict_receipt=accepted),
        z_init=dict(values=z.tolist(),file=str(source_z),descriptor=r['arrays'][row['z_init']]),
        root_source=slot['RGB_source'],expected_model_hashes=expected,
        construction_expected=dict(module_count=spec['module_count'],entry_count=len(spec['entries']),
            state_key_count=len(expected),ir_sha256=low.base.loader.IR_SHA),scientific_admission=False)


def scoped(path):
    p=Path(path);need(p.is_absolute() and p.resolve()==p and p.parent==ROOT/'results' and
        p.name.startswith(TAG+'_'),'exclusive own-prefix direct results path');return p


def write(path,b):
    need(not Path(path).exists(),'exclusive file')
    with Path(path).open('xb') as f:need(f.write(b)==len(b),'full write');f.flush();os.fsync(f.fileno())


def prepare(output):
    m=build();p=scoped(output);need(shutil.disk_usage(p.parent).free>=CAP+FLOOR,'cap/floor')
    b=encode(m);need(len(b)<=RESERVE,'bounded prepare metadata')
    p.mkdir();write(p/'manifest.json',b);return dict(manifest=str(p/'manifest.json'),sha256=sha(p/'manifest.json'))


def verify(path,h):
    m=read(path,h);need(m==build() and sha(path)==h,'entire authority/manifest/source reconstruction');return m


class Store:
    def __init__(self,path,start):
        self.path=scoped(path);self.start=start;self.used=0;self.arrays={}
        need(shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'whole cap/floor');self.path.mkdir()
    def guard(self,n=0,final=False):
        need(final or time.monotonic()-self.start<=WALL,'cooperative wall')
        need(self.used+n<=CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free>=FLOOR+n,'cap/reserve/floor')
    def put(self,name,b,final=False):
        need(Path(name).name==name and name not in ('','.','..'),'flat output');self.guard(len(b),final)
        write(self.path/name,b);self.used+=len(b)
    def array(self,x):
        a=x if isinstance(x,np.ndarray) else x.detach().cpu().numpy()
        need(a.dtype.str in ('<f4','<f8') and np.isfinite(a).all(),'original finite f32/f64, no coercion')
        stream=io.BytesIO();np.save(stream,a,allow_pickle=False);b=stream.getvalue()
        name=f'array_{len(self.arrays):05d}.npy';self.put(name,b)
        self.arrays[name]=dict(file_sha256=digest(b),file_bytes=len(b),data_sha256=digest(a.tobytes()),
            shape=list(a.shape),dtype=a.dtype.str)
        return name


def layout(x):
    return dict(shape=list(x.shape),stride=list(x.stride()),storage_offset=x.storage_offset(),
        contiguous=x.is_contiguous(),dtype=str(x.dtype),device=str(x.device),requires_grad=x.requires_grad)


def expected_precision(grad):
    return dict(deterministic=True,tf32_matmul=False,tf32_cudnn=False,autocast=False,grad=grad,inference=False,
        math_sdp=True,flash_sdp=False,efficient_sdp=False,cudnn_sdp=False)


def precision(t,grad):
    p=dict(deterministic=t.are_deterministic_algorithms_enabled(),tf32_matmul=t.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=t.backends.cudnn.allow_tf32,autocast=t.is_autocast_enabled('cuda'),grad=t.is_grad_enabled(),
        inference=t.is_inference_mode_enabled(),math_sdp=t.backends.cuda.math_sdp_enabled(),
        flash_sdp=t.backends.cuda.flash_sdp_enabled(),efficient_sdp=t.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=t.backends.cuda.cudnn_sdp_enabled())
    need(p==expected_precision(grad),'actual grad/MATH/noTF32/noautocast guard');return p


class Trace:
    def __init__(self,t,store,row,grad):self.t,self.store,self.row,self.grad=t,store,row,grad
    def call(self,kind,args,fn):
        self.store.guard();e=dict(index=len(self.row['events']),kind=kind,attempted=True,returned=False,
            inputs=[],layouts=[],output=None,output_layout=None,precision=precision(self.t,self.grad))
        self.row['events'].append(e)
        capture=time.monotonic()
        e['inputs']=[self.store.array(x) for x in args];e['layouts']=[layout(x) for x in args]
        e['CPUcapture_seconds']=time.monotonic()-capture;start=time.monotonic()
        y=fn();self.t.cuda.synchronize();e['callback_sync_wall_seconds']=time.monotonic()-start;self.store.guard()
        capture=time.monotonic()
        e.update(output=self.store.array(y),output_layout=layout(y),returned=True)
        e['CPUcapture_seconds']+=time.monotonic()-capture
        need(precision(self.t,self.grad)==e['precision'],'precision stable around callback');return y


def reference(t,root,z_hist,a_hist,future,trace):
    """Independent transcription of frozen OwnedLow; never calls seam kernels."""
    b,s,steps,width=future.shape
    z=z_hist.reshape(b*s,1,192);a=a_hist.reshape(b*s,1,10);pred=[];contexts=[]
    for step in range(steps):
        a=t.cat((a,future[:,:,step].reshape(b*s,1,width)),dim=1)
        def embed():
            n=root.action_encoder.patch_embed
            return root.action_encoder.embed(t.nn.functional.conv1d(a.transpose(1,2),n.weight,n.bias).transpose(1,2))
        emb=trace.call('action_encoder',[a],embed)
        ctx=min(3,z.shape[1],emb.shape[1]);zi,ai=z[:,-ctx:],emb[:,-ctx:]
        h=trace.call('low_predictor',[zi,ai],lambda:root.low_predictor(zi,ai))
        p=trace.call('low_pred_proj',[h.reshape(-1,192)],lambda:root.low_pred_proj(h.reshape(-1,192)))
        nxt=p.reshape(b*s,ctx,192)[:,-1:];z=t.cat((z,nxt),dim=1);pred.append(nxt[:,0])
        contexts.append((z,a))
    return t.stack(pred,dim=1).reshape(b,s,steps,192),contexts


def comparisons(a,b):
    need(a.dtype==b.dtype and a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'finite same ABI')
    return dict(exact=a.tobytes()==b.tobytes(),maxabs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))


def finite_difference_check(g,d,plus,minus):
    p=protocol()['finite_difference'];ad=math.fsum(float(x)*float(y) for x,y in zip(g.flat,d.flat,strict=True))
    fd=(float(plus)-float(minus))/(2*p['epsilon']);limit=p['atol']+p['rtol']*max(abs(ad),abs(fd))
    return dict(AD_dot=ad,FD=fd,error=abs(ad-fd),limit=limit,passed=abs(ad-fd)<=limit)


def resources(rows):
    es=[e for r in rows for e in r['events']];mods={}
    for kind in KINDS+('scalar_VJP',):
        xs=[e for e in es if e['kind']==kind]
        mods[kind]=dict(attempts=len(xs),returns=sum(e['returned'] for e in xs),
            input_rows=sum(e['layouts'][0]['shape'][0] for e in xs if e['layouts'] and e['layouts'][0]['shape']),
            input_tokens=sum(e['layouts'][0]['shape'][0]*e['layouts'][0]['shape'][1]
                for e in xs if e['layouts'] and len(e['layouts'][0]['shape'])==3))
    return dict(modules=mods,encoder=0,projector=0,high_predictor=0,physics=0,predict_API=0,
        mathematical_scalar_VJP=sum(e['kind']=='scalar_VJP' for e in es),backward_internal_calls=None,FLOPs=None,
        AD_primal_forward_calls=sum(e['kind']=='low_predictor' for r in rows[:3] for e in r['events']),
        inherited='parent receipt resources remain paid; not a new encoder/high run')


def dispatch(t,root,m,store,rows):
    f=fixed_arrays();device=next(root.parameters()).device
    need(device.type=='cuda','HOST CUDA only')
    w=t.from_numpy(f['weights']).to(device);z=t.from_numpy(np.asarray(m['z_init']['values'],np.float32)).to(device)
    ah=t.zeros((1,1,1,10),dtype=t.float32,device=device);zh=z.reshape(1,1,1,192)
    for index,row in enumerate(rows):
        route_start=time.monotonic();grad=index<3;row['status']='attempted';trace=Trace(t,store,row,grad)
        actions=f['actions'].copy()
        if index>=3:actions=actions+(1 if index==3 else -1)*protocol()['finite_difference']['epsilon']*f['direction']
        a=t.from_numpy(actions).to(device)
        if grad:a.requires_grad_(True)
        with t.set_grad_enabled(grad):
            row['precision_before']=precision(t,grad)
            row['z']=store.array(zh);row['a_hist']=store.array(ah);row['actions']=store.array(a)
            if row['route'] in ('reference','fd_plus','fd_minus'):
                pred,contexts=reference(t,root if grad else root._root,zh,ah,a,trace)
            else:
                k=seam.cuda_kernels(t,root,main_authorized=True,trace=trace)
                out=(seam.rollout(zh,ah,a,kernels=k) if row['route']=='whole' else
                     seam.segmented_rollout(zh,ah,a,segments=(1,1,1),kernels=k))
                pred=out.prediction;contexts=[(c.z,c.a) for c in out.contexts]
            row['prediction']=store.array(pred)
            row['contexts']=[dict(z=store.array(zz),a=store.array(aa)) for zz,aa in contexts]
            scalar=(pred[:,:,-1].to(t.float64)*w.to(t.float64)).sum()
            row['scalar']=store.array(scalar)
            if grad:
                gradient=trace.call('scalar_VJP',[scalar,a],lambda:t.autograd.grad(scalar,a,
                    create_graph=False,retain_graph=False,allow_unused=False)[0])
                row['gradient']=store.array(gradient)
            row['precision_after']=precision(t,grad);row['status']='returned'
            row['wall_seconds']=time.monotonic()-route_start


def run_cuda(manifest,mh,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN authorization before I/O/runtime')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','fixed deterministic CUBLAS')
    start=time.monotonic();store=Store(output,start);root=None;m=None
    r=dict(schema=TAG+'.report',manifest=str(Path(manifest).resolve()),manifest_sha256=mh,protocol=protocol(),
        arrays=store.arrays,rows=[dict(route=x,status='not_started',events=[]) for x in ROUTES],
        model_before=None,model_after=None,construction=None,sources_before=None,sources_after=None,
        errors=[],completed=False,timings={},scientific_admission=False)
    def timed(name,fn):
        begin=time.monotonic();e=dict(attempted=True,returned=False,seconds=None);r['timings'][name]=e
        try:
            value=fn();e['returned']=True;return value
        finally:e['seconds']=time.monotonic()-begin
    try:
        m=timed('initial_saved_verify',lambda:verify(manifest,mh));r['sources_before']=m['sources']
        import torch as t
        from torch.nn.attention import sdpa_kernel,SDPBackend
        need(t.cuda.is_available(),'HOST CUDA required, no CPU fallback')
        need(Path(t.__file__).resolve()==ROOT/'.venv/lib/python3.12/site-packages/torch/__init__.py','bound venv torch')
        t.use_deterministic_algorithms(True);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
        low=owner();e=m['export']
        root,r['construction']=timed('construct',lambda:low.base.loader.construct_cuda(e['export_report'],e['export_report_sha256'],
            e['export_receipt'],e['export_receipt_sha256'],main_authorized=True))
        r['model_before']=timed('model_hash_before',lambda:low.base.paid.diag.old.weights(root))
        need(r['model_before']==r['construction']['tensor_value_hashes']==m['expected_model_hashes'],'all original weights')
        root=diff.wrap(root,t,main_authorized=True)
        r['runtime']=dict(torch=str(t.__version__),cuda=t.version.cuda,device=str(next(root.parameters()).device))
        with t.enable_grad(),t.autocast('cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            timed('dispatch_all5',lambda:dispatch(t,root,m,store,r['rows']))
        r['runtime'].update(peak_allocated_bytes=t.cuda.max_memory_allocated(),peak_reserved_bytes=t.cuda.max_memory_reserved())
        r['model_after']=timed('model_hash_after',lambda:low.base.paid.diag.old.weights(root))
        r['sources_after']=timed('final_saved_verify',lambda:verify(manifest,mh))['sources']
        need(r['model_before']==r['model_after'] and r['sources_before']==r['sources_after'],'weights/authority unchanged')
        store.guard();r['resources']=resources(r['rows'])
        r['diagnostics']=validate_rows(m,r,lambda n:load_array(store.path,n,r['arrays'][n]))
        store.guard()
        r['completed']=True
    except BaseException as exc:r['errors'].append(dict(type=type(exc).__name__,message=str(exc)[:3000]))
    finally:
        if root is not None and r['model_after'] is None:
            try:r['model_after']=owner().base.paid.diag.old.weights(root)
            except BaseException as exc:r['errors'].append(dict(type=type(exc).__name__,message=str(exc)[:1000]))
        if m is not None and r['sources_after'] is None:
            try:r['sources_after']={p:sha(p) for p in m['sources']}
            except BaseException as exc:r['errors'].append(dict(type=type(exc).__name__,message=str(exc)[:1000]))
        r.update(resources=resources(r['rows']),elapsed_seconds=time.monotonic()-start)
        store.put('report.json',encode(r),final=True)
    need(r['completed'] and not r['errors'],'probe failed; partial report/arrays retained, no retry')
    return dict(report=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def load_array(folder,name,d):
    need(isinstance(name,str) and Path(name).name==name and name.startswith('array_') and name.endswith('.npy'),'flat NPY name')
    b=body(Path(folder)/name)
    need(len(b)==d['file_bytes'] and digest(b)==d['file_sha256'],'NPY file size/SHA')
    stream=io.BytesIO(b);v=np.lib.format.read_magic(stream);need(v in ((1,0),(2,0)),'NPY version')
    shape,fort,dt=(np.lib.format.read_array_header_1_0(stream) if v==(1,0) else np.lib.format.read_array_header_2_0(stream))
    need(not fort and dt.str in ('<f4','<f8') and len(shape)<=4 and all(type(x) is int and x>=0 for x in shape)
         and math.prod(shape)*dt.itemsize==len(b)-stream.tell(),'bounded original NPY ABI')
    stream.seek(0);a=np.load(stream,allow_pickle=False)
    need(stream.tell()==len(b) and list(a.shape)==d['shape'] and a.dtype.str==d['dtype'] and
         np.isfinite(a).all() and digest(a.tobytes())==d['data_sha256'],'array dtype/shape/raw SHA/finite/EOF');return a


def equal(a,b,message):need(comparisons(a,b)['exact'],message)


def check_layout(l,a,grad,device):
    need(set(l)=={'shape','stride','storage_offset','contiguous','dtype','device','requires_grad'} and
        l['shape']==list(a.shape) and l['dtype']=={'<f4':'torch.float32','<f8':'torch.float64'}[a.dtype.str] and
        l['device']==device and type(l['requires_grad']) is bool and
        len(l['stride'])==a.ndim and all(type(x) is int and x>=0 for x in l['stride']) and
        type(l['storage_offset']) is int and l['storage_offset']>=0 and type(l['contiguous']) is bool,
        'complete CUDA tensor layout')
    if not grad:need(l['requires_grad'] is False,'FD route has no graph')


def validate_rows(m,r,load):
    need([row['route'] for row in r['rows']]==list(ROUTES),'all five fixed route denominators')
    device=r['runtime']['device'];need(device.startswith('cuda:'),'CUDA runtime, no fallback')
    f=fixed_arrays();outputs=[];gradients=[];scalars=[]
    for i,row in enumerate(r['rows']):
        grad=i<3;need(row['status']=='returned' and row['precision_before']==row['precision_after']==expected_precision(grad),
            'complete route and fixed precision')
        need(type(row['wall_seconds']) in (int,float) and math.isfinite(row['wall_seconds']) and row['wall_seconds']>=0,
             'actual per-route wall scope')
        z=load(row['z']);a=load(row['a_hist']);acts=load(row['actions'])
        expected=f['actions'] if grad else f['actions']+(1 if i==3 else -1)*protocol()['finite_difference']['epsilon']*f['direction']
        equal(acts,expected,'fixed actual actions');equal(z,np.asarray(m['z_init']['values'],np.float32).reshape(1,1,1,192),'paid z_init')
        equal(a,np.zeros((1,1,1,10),np.float32),'explicit original zero history')
        z=z.reshape(1,1,192);a=a.reshape(1,1,10);pred=[];cursor=0
        def event(kind,inputs):
            nonlocal cursor
            need(cursor<len(row['events']),'missing native event');e=row['events'][cursor];cursor+=1
            need(e['index']==cursor-1 and e['kind']==kind and e['attempted'] is True and e['returned'] is True
                and e['precision']==expected_precision(grad) and len(e['inputs'])==len(e['layouts'])==len(inputs),
                'all native chronological attempts/returns')
            need(all(type(e[k]) in (int,float) and math.isfinite(e[k]) and e[k]>=0
                for k in ('CPUcapture_seconds','callback_sync_wall_seconds')),'actual bounded timing scopes')
            xs=[load(n) for n in e['inputs']];y=load(e['output'])
            for actual,wanted in zip(xs,inputs,strict=True):equal(actual,wanted,'native input delivery')
            for v,l in zip(xs+[y],e['layouts']+[e['output_layout']],strict=True):check_layout(l,v,grad,device)
            if kind=='scalar_VJP':
                need(all(l['requires_grad'] is True for l in e['layouts']) and
                     e['output_layout']['requires_grad'] is False,'explicit scalar VJP graph and non-higher-order result')
            return y
        need(len(row['contexts'])==3,'all three full contexts')
        for k in range(3):
            a=np.concatenate((a,acts[:,:,k].reshape(1,1,10)),axis=1)
            emb=event('action_encoder',[a]);need(emb.shape==(1,k+2,192) and emb.dtype==np.float32,'full AE ABI')
            ctx=min(3,z.shape[1],emb.shape[1]);h=event('low_predictor',[z[:,-ctx:],emb[:,-ctx:]])
            need(h.shape==(1,ctx,192) and h.dtype==np.float32,'predictor ABI')
            p=event('low_pred_proj',[h.reshape(-1,192)]);need(p.shape==(ctx,192) and p.dtype==np.float32,'projection ABI')
            nxt=p.reshape(1,ctx,192)[:,-1:];z=np.concatenate((z,nxt),axis=1);pred.append(nxt[:,0])
            equal(load(row['contexts'][k]['z']),z,'whole z history');equal(load(row['contexts'][k]['a']),a,'whole action history')
        out=load(row['prediction']);equal(out,np.stack(pred,axis=1).reshape(1,1,3,192),'full rollout')
        scalar=load(row['scalar']);need(scalar.dtype==np.float64 and scalar.shape==(),'float64 diagnostic scalar')
        products=[float(x)*float(y) for x,y in zip(out[:,:,-1].flat,f['weights'].flat,strict=True)]
        bound=192*np.finfo(np.float64).eps*math.fsum(abs(x) for x in products)+np.nextafter(0.,1.)*192
        need(abs(float(scalar)-math.fsum(products))<=bound,'independent fixed scalar contraction')
        if grad:
            g=event('scalar_VJP',[scalar,acts]);need(g.shape==(1,1,3,10) and g.dtype==np.float32,'complete30 action gradient')
            equal(g,load(row['gradient']),'gradient delivery');gradients.append(g)
            for e in row['events'][:9]:need(e['output_layout']['requires_grad'] is True,'graph retained in all neural outputs')
        need(cursor==len(row['events']),'no extra events');outputs.append(out);scalars.append(scalar)
    for index in (1,2):
        equal(outputs[0],outputs[index],'reference/seam full forward bitwise')
        equal(gradients[0],gradients[index],'reference/seam gradient bitwise')
        for re,ce in zip(r['rows'][0]['events'],r['rows'][index]['events'],strict=True):
            need(re['layouts']==ce['layouts'] and re['output_layout']==ce['output_layout'],'exact matching native input/output layouts')
            for rn,cn in zip(re['inputs']+[re['output']],ce['inputs']+[ce['output']],strict=True):equal(load(rn),load(cn),'exact module values')
    fd=finite_difference_check(gradients[0],f['direction'],scalars[3],scalars[4])
    need(fd['passed'],'fixed central directional derivative criterion failed')
    need(r['resources']==resources(r['rows']),'actual independent resource census')
    for kind in KINDS:need(r['resources']['modules'][kind]['attempts']==r['resources']['modules'][kind]['returns']==15,'all fifteen modules')
    need(r['resources']['mathematical_scalar_VJP']==3,'all three VJP attempts')
    return dict(forward_bitwise=True,layouts_exact=True,gradient_bitwise=True,directional_check=fd,
        official_parity=False,P2_pairing=False,scientific_admission=False)


def consume(output,rh):
    output=scoped(output);r=read(output/'report.json',rh);m=verify(r['manifest'],r['manifest_sha256'])
    need(r['schema']==TAG+'.report' and r['protocol']==protocol() and r['completed'] is True and r['errors']==[]
         and r['scientific_admission'] is False,'complete actual report')
    need(r['sources_before']==r['sources_after']==m['sources'] and
        r['model_before']==r['model_after']==m['expected_model_hashes']==r['construction']['tensor_value_hashes'],
        'all source/input/export receipts and weights before/after')
    need(set(r['runtime'])=={'torch','cuda','device','peak_allocated_bytes','peak_reserved_bytes'} and
         all(type(r['runtime'][k]) is int and r['runtime'][k]>=0 for k in ('peak_allocated_bytes','peak_reserved_bytes')),
         'actual runtime and GPU allocation scopes')
    c=r['construction'];need(c['export_report_sha256']==m['export']['export_report_sha256'] and
        c['export_receipt_sha256']==m['export']['export_receipt_sha256'] and c['all_parameters_frozen'] is True and c['forward_calls']==0,
        'original frozen loader construction receipt')
    need(all(c[k]==v for k,v in m['construction_expected'].items()) and
         c['tensor_copy_attempts']==c['tensor_copy_returns']==425 and c['device']==r['runtime']['device'] and
         c['torch']==r['runtime']['torch'] and all(m['sources'].get(p)==h for p,h in c['own_sha256'].items()),
         'full loader counts/source/runtime closure')
    need(set(r['timings'])=={'initial_saved_verify','construct','model_hash_before','dispatch_all5','model_hash_after','final_saved_verify'}
         and all(e['attempted'] is True and e['returned'] is True and type(e['seconds']) in (int,float)
         and math.isfinite(e['seconds']) and e['seconds']>=0 for e in r['timings'].values()),'complete actual timing scope ledger')
    arrays={n:load_array(output,n,d) for n,d in r['arrays'].items()}
    need({p.name for p in output.iterdir()}==set(arrays)|{'report.json'} and
        sum(p.stat().st_size for p in output.iterdir())<=CAP,'complete bounded directory census')
    refs=set()
    def load(n):need(n in arrays,'actual array reference');refs.add(n);return arrays[n]
    result=validate_rows(m,r,load);need(result==r['diagnostics'] and refs==set(arrays),'full diagnostics and array census')
    need(sha(output/'report.json')==rh and verify(r['manifest'],r['manifest_sha256'])==m,'report/authority stable')
    return dict(schema=TAG+'.consume',status='fixed_owned_low_probe_saved_verified_not_official_parity',
        report_sha256=rh,manifest_sha256=r['manifest_sha256'],diagnostics=result,resources=r['resources'],scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    p.add_argument('--manifest');p.add_argument('--manifest-sha256');p.add_argument('--output');p.add_argument('--report-sha256')
    p.add_argument('--receipt');p.add_argument('--main-authorized',action='store_true');a=p.parse_args()
    if a.command=='prepare':v=prepare(a.output)
    elif a.command=='verify':v=verify(a.manifest,a.manifest_sha256)
    elif a.command=='run':v=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else:
        v=consume(a.output,a.report_sha256)
        if a.receipt:write(scoped(a.receipt),encode(v))
    print(encode(v).decode(),end='')


if __name__=='__main__':main()

