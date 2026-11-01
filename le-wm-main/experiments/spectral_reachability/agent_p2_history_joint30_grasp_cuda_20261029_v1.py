"""History-aware joint30 bounded GRASP VARIANT, not official reproduction.

Only MAIN may run HOST CUDA. Import/prepare/verify/consume never import torch.
No physical release, outcomes, training, encoder, restart or best-iterate choice.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sysconfig
import time
import zlib
import numpy as np
from experiments.spectral_reachability import agent_p2_grasp_raw15_prepare_20261029_v1 as inputs
from experiments.spectral_reachability import agent_p2_three_program_20261028_v1 as upstream
from experiments.spectral_reachability import agent_p1_nonlinear_suffix_library_20261028_v2 as installed

NAME='agent_p2_history_joint30_grasp_cuda_20261029_v1'
AUTH='MAIN_HISTORY_JOINT30_GRASP_HOST_CUDA'
ROOT=inputs.ROOT
need=inputs.need
CAP,FINAL,FLOOR,WALL=512*1024**2,8*1024**2,200*1024**2,3600
BALL=float(.75*np.sqrt(15))
STEPS,SYNCEVERY,SYNCSTEPS=750,50,10
LRV,LRX,LRS,SIGMA=.1,.005,.05,.005


def raw(v):return inputs.encoded(v)
def digest(v):return inputs.sha(raw(v))
def filehash(p):
    p=Path(p);need(p.is_file() and not p.is_symlink(),'regular source/artifact '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def equal(a,b,msg):need(raw(a)==raw(b),msg)
def seed(s):need(type(s) is int and s in (0,1),'known slot');return 20261029+101*s
def phases():
    for k in range(STEPS):
        yield 'joint',k,None
        if (k+1)%SYNCEVERY==0 and k>0:
            for j in range(SYNCSTEPS):yield 'sync',k,j
def budget():
    n=3*(STEPS+(STEPS//SYNCEVERY)*SYNCSTEPS)+6
    return dict(predict_API=n,predict_samples=n,action_encoder=n,predictor=n,pred_proj=n,
        predictor_context_tokens=3*n,pred_proj_rows=3*n,scalar_backward=900,
        Adam_step=900,virtual_noise=750,encoder=0,projector=0,Jacobian=0,SVD=0,physics=0)
def contract():
    return dict(name=NAME,variant='history-aware joint30 bounded GRASP; NOT official H1 implementation',
        roots=[0,1],condition='off',horizon=3,raw_steps=15,packet_steps=5,history=3,
        steps=STEPS,sync_interval=SYNCEVERY,sync_steps=SYNCSTEPS,lr_virtual=LRV,lr_logits=LRX,sync_lr=LRS,
        noise_sigma=SIGMA,noise='torch.randn_like after joint Adam only; global CUDA RNG seeded per root',
        seed='20261029+101*root_slot',initialization='raw-zero logits; virtual linear thirds between original last latent and goal',
        logits_dtype='float64',virtual_dtype='float32',chart='NumPy tanh(float64).astype(float32); analytic first derivative once',
        domain='ONE joint30 L2 ball, projected after each Adam update',cap=BALL,
        loss='sum3 MSE(pred,virtual_next_or_goal) + sum3 MSE(pred,goal); goal weight1, no decay',
        stopgrad='ALL lifted latent inputs detached; next virtual targets and between-state actions live',
        sync='full rolling H3 serial3; terminal MSE; fresh action-only Adam10; reset joint optimizer, no virtual refresh',
        optimizer='installed torch Adam: betas(.9,.999),eps1e-8,weight_decay0,foreach=False,fused=False,amsgrad=False',
        final='LAST action only; whole15 plus independently reconstructed 3 segments, all3 contexts exact',
        per_root=budget(),total={k:2*v for k,v in budget().items()},
        old_common_predict_cap=1902,equal_budget_to_1902=False,equal_FLOPs=False,total_FLOPs=None,
        actual_partial_work_no_refund=True,failed_slots_retained=True,
        inherited_encoding='original paid tasks, full upstream rebuild in prepare; no new images',
        CUDA_only=True,SDP='MATH',TF32=False,autocast=False,new_training=False,
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,floor_bytes=FLOOR,wall_seconds=WALL,
        P1_admission=False,P2_admission=False,fresh=False,official_reproduction=False,physical_release=False)


def pack(a):
    a=np.asarray(a);need(a.dtype in (np.dtype('float32'),np.dtype('float64')) and np.isfinite(a).all(),'finite native f32/f64')
    shape=a.shape;a=np.ascontiguousarray(a,dtype='<f4' if a.dtype==np.float32 else '<f8').reshape(shape);b=a.tobytes()
    return dict(dtype=str(a.dtype),shape=list(a.shape),hex=b.hex(),sha256=inputs.sha(b))
def unpack(d,shape=None,dtype=None):
    need(set(d)=={'dtype','shape','hex','sha256'} and d['dtype'] in ('float32','float64') and
        len(d['shape'])<=3 and all(type(n) is int and 0<n<=192 for n in d['shape']),'bounded literal array descriptor')
    need(len(d['hex'])<=192*192*16,'bounded array hex')
    b=bytes.fromhex(d['hex']);need(inputs.sha(b)==d['sha256'],'array bytes SHA')
    a=np.frombuffer(b,dtype='<f4' if d['dtype']=='float32' else '<f8').reshape(d['shape'])
    need(np.isfinite(a).all() and (shape is None or a.shape==shape) and (dtype is None or a.dtype==dtype),'array ABI/finite')
    return a
def arr(v,shape,dtype=np.float32):
    a=np.asarray(v,dtype=dtype);need(a.shape==shape and np.isfinite(a).all(),'shape/finite');return a
def chart(x):return np.tanh(arr(x,(30,),np.float64)).astype(np.float32).reshape(3,5,2)
def project(x):
    x=arr(x,(30,),np.float64).copy()
    # Explicit algorithmic projection of a proposal, never repair an incumbent.
    if np.linalg.norm(x)>BALL:x*=np.nextafter(BALL,0.)/np.linalg.norm(x)
    for _ in range(8):
        if np.linalg.norm(x)<=BALL:return x
        x*=np.nextafter(1.,0.)
    raise ValueError('cannot project finite Adam proposal')
def contexts(z,h):
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.asarray(z,np.float32).reshape(3,192).tolist(),action_history=np.asarray(h,np.float32).reshape(2,10).tolist())
def lifted_context(root,v,k,tokens):
    z=arr(root['latent_history'],(3,192));h=arr(root['action_history'],(2,10))
    for j in range(k):z=np.concatenate((z[1:],v[j:j+1]));h=np.concatenate((h[1:],tokens[j:j+1]))
    return contexts(z,h)


def source_files():
    seen=set(upstream.files());todo=[Path(__file__).resolve(),Path(inputs.__file__),Path(installed.__file__)]
    while todo:
        p=todo.pop().resolve()
        if p in seen:continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(n,ast.Import):names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom) and n.module:names=[n.module]+[n.module+'.'+a.name for a in n.names]
            for name in names:
                f=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if name.startswith('experiments.') and f.is_file() and f not in seen:todo.append(f)
    site=Path(sysconfig.get_paths()['purelib'])
    seen.update(site/x for x in ('torch/optim/adam.py','torch/optim/optimizer.py','torch/optim/_functional.py','torch/random.py'))
    seen.update(Path(x['path']) for x in installed.installed_class_sources().values())
    return sorted(seen|{ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')})
def checkbindings(b):
    for p,h in b.items():need(filehash(p)==h,'source/model/input closure '+p)
def build():
    # MAIN preparation executes saved consumers, never a model or physical call.
    authority=upstream.build();original=inputs.parse(inputs.read_bound(inputs.UPSTREAM,inputs.UPSTREAM_SHA))
    equal(authority,original,'full original prepare/encoding authority rebuilt, no planner outcomes')
    exports=[inputs.parse(inputs.read_bound(s['task_file'],s['task_sha256'],compressed=True)) for s in original['slots']]
    slots=inputs.make_request(original,exports);bindings=dict(original['bindings_sha256'])
    bindings[str(inputs.UPSTREAM)]=inputs.UPSTREAM_SHA
    for s in slots:bindings[s['task_file']]=s['task_sha256']
    for p in source_files():bindings[str(p)]=filehash(p)
    checkbindings(bindings)
    return dict(schema=NAME+'.prepare',contract=contract(),slots=slots,policy=original['policy'],
        expected_model_hash=original['expected_model_hash'],bindings=bindings,
        upstream_sha256=inputs.UPSTREAM_SHA,source_report_sha256=inputs.SOURCE,
        full_saved_input_authority_rebuilt=True,model_visible_inputs_only=True)
def readjson(path,cap=2*1024**2):
    p=Path(path);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'bounded regular JSON')
    return inputs.parse(p.read_bytes())
def metadata(path,v):
    p=Path(path).absolute();b=raw(v)
    need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=2*1024**2 and
        shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive small own metadata')
    with p.open('xb') as f:need(f.write(b)==len(b),'short metadata');f.flush();os.fsync(f.fileno())
    upstream.fsync_dir(p.parent);return dict(path=str(p),sha256=filehash(p))
def verify(path,expected):
    need(filehash(path)==expected,'external prepare SHA');p=readjson(path)
    equal(p,build(),'entire own prepare rebuild');return p


class Store:
    def __init__(self,path,start):
        self.path=Path(path).absolute();self.start=start;self.seq=0;self.tail='0'*64;self.closed=False
        need(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink() and
            shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'exclusive512MiB output +200MiB floor')
        self.path.mkdir();upstream.fsync_dir(self.path.parent);self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.counts={}
        self.write('events.jsonl.gz',b'')
    def check(self):need(time.monotonic()-self.start<WALL,'whole-stage3600s boundary watchdog')
    def write(self,name,b,append=False,final=False):
        if not final:self.check()
        need(Path(name).name==name and name not in ('','.','..') and not (self.path/name).is_symlink(),'flat artifact')
        used=sum(p.stat().st_size for p in self.path.iterdir())
        need(used+len(b)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+len(b),'cap/floor including failed partials')
        with (self.path/name).open('ab' if append else 'xb') as f:need(f.write(b)==len(b),'short artifact');f.flush();os.fsync(f.fileno())
        upstream.fsync_dir(self.path)
    def put(self,name,v,final=False):self.write(name,raw(v),final=final)
    def event(self,v):
        need(not self.closed,'journal closed');e=dict(index=self.seq,previous=self.tail,payload=deepcopy(v));h=digest(e)
        b=self.comp.compress(raw(dict(e,sha256=h)))+self.comp.flush(zlib.Z_SYNC_FLUSH)
        self.write('events.jsonl.gz',b,append=True);self.tail=h;self.seq+=1;count_event(self.counts,v)
    def close(self):
        if not self.closed:self.write('events.jsonl.gz',self.comp.flush(),append=True,final=True);self.closed=True


class Native:
    def __init__(self,model,device,task,event,store):
        need(device.type=='cuda','HOST CUDA only')
        import torch
        self.t=torch;self.model=model;self.device=device;self.task=task;self.event=event;self.store=store;self.n=0;self.active=None
    def tensor(self,a,dtype=None):return self.t.tensor(a,device=self.device,dtype=dtype or self.t.float32)
    def snap(self,x):
        need(x.device==self.device and x.dtype in (self.t.float32,self.t.float64),'native precision array')
        return pack(x.detach().cpu().numpy())
    def check(self):
        self.store.check();self.t.cuda.synchronize(self.device)
        need(self.t.cuda.max_memory_allocated(self.device)<=4*1024**3 and
            self.t.cuda.max_memory_reserved(self.device)<=6*1024**3,'CUDA4GiB allocated/6GiB reserved cap')
    @contextmanager
    def observe(self):
        handles=[];pending=None
        def enter(name):
            def hook(module,args):
                nonlocal pending
                need(self.active is not None and pending is None and name==('action_encoder','predictor','pred_proj')[self.active], 'native exact module order')
                self.check();pending=(module,name)
                self.event(dict(kind='module_attempt',index=self.n,resource=name,inputs=[self.snap(x) for x in args]))
            return hook
        def returned(module,args,out):
            nonlocal pending
            need(pending is not None and pending[0] is module,'paired native module return')
            self.check();self.event(dict(kind='module_return',index=self.n,resource=pending[1],output=self.snap(out)))
            pending=None;self.active+=1
        def forbid(module,args):raise ValueError('new encoder/projector forbidden')
        try:
            for name in ('action_encoder','predictor','pred_proj'):
                m=getattr(self.model,name);handles.extend((m.register_forward_pre_hook(enter(name)),m.register_forward_hook(returned)))
            for name in ('encoder','projector'):handles.append(getattr(self.model,name).register_forward_pre_hook(forbid))
            yield
            need(pending is None and self.active is None,'all native operations returned')
        finally:
            for h in reversed(handles):h.remove()
    def step(self,z,h,a,label):
        t=self.t;need(self.n<budget()['predict_API'] and self.active is None,'predict budget/no recursion')
        self.check();self.event(dict(kind='forward_attempt',index=self.n,label=label,z=self.snap(z),h=self.snap(h),raw=self.snap(a)))
        mean,std=(self.tensor(self.task[k]).repeat(5) for k in ('action_mean','action_std'))
        token=(a.reshape(10)-mean)/std;outgoing=t.cat((h,token[None,None]),1);self.active=0
        emb=self.model.action_encoder(outgoing)
        self.event(dict(kind='predict_attempt',index=self.n))
        predicted=self.model.predict(z,emb)
        self.check();need(self.active==3 and tuple(predicted.shape)==(1,3,192),'complete native257-encoded full3 prediction')
        self.event(dict(kind='predict_return',index=self.n,output=self.snap(predicted)))
        after=t.cat((z[:,1:],predicted[:,-1:]),1);between=outgoing[:,1:]
        self.event(dict(kind='forward_return',index=self.n,z=self.snap(after),h=self.snap(between)))
        self.active=None;self.n+=1;return after,between
    def root(self):return self.tensor(self.task['root_context']['latent_history'])[None],self.tensor(self.task['root_context']['action_history'])[None]
    def exactraw(self,x):
        # Native f32 forward identical to the original NumPy float64 chart;
        # analytical surrogate for first derivative only, no quantization AD claim.
        point=x.detach().cpu().numpy();value=self.tensor(chart(point))
        chain=self.tensor(1-np.tanh(point)**2,self.t.float64)
        out=value+((x-x.detach())*chain).to(self.t.float32).reshape(3,5,2)
        if out.requires_grad:
            out.register_hook(lambda g:self._raw_gradient(g))
        return out
    def _raw_gradient(self,g):
        self.event(dict(kind='chart_backward',raw_gradient=self.snap(g)));return g
    def serial(self,x,label,segmented=False):
        z,h=self.root();a=self.exactraw(x);paths=[]
        for k in range(3):
            if segmented and paths:z,h=(self.tensor(paths[-1][key])[None] for key in ('latent_history','action_history'))
            z,h=self.step(z,h,a[k],label+'/'+str(k));paths.append(contexts(z.detach().cpu().numpy(),h.detach().cpu().numpy()))
        return z[:,-1],paths
    def soft(self,v,x):
        t=self.t;z,h=self.root();a=self.exactraw(x);goal=self.tensor(self.task['goal'])[None]
        dyn=goal.new_zeros(());dense=goal.new_zeros(())
        for k in range(3):
            z,h=self.step(z.detach(),h,a[k],'lifted/'+str(k));pred=z[:,-1]
            target=v[:,k] if k<2 else goal
            dyn=dyn+(pred-target).square().mean();dense=dense+(pred-goal).square().mean()
            if k<2:z=t.cat((z[:,:-1],v[:,k:k+1]),1)
        return dyn+dense,dyn,dense


def optimizer_snapshot(opt,p,snap):
    d=opt.state.get(p,{})
    return dict(step=int(d['step'].item()) if d else 0,m=snap(d['exp_avg'] if d else p.detach()*0),
        q=snap(d['exp_avg_sq'] if d else p.detach()*0))


def optimize(task,slot,b):
    t=b.t;z,_=b.root();g=b.tensor(task['goal'])[None];frac=t.linspace(0,1,4,device=b.device)[1:-1][None,:,None]
    v=(z[:,-1,None]+frac*(g[:,None]-z[:,-1,None])).detach().requires_grad_(True)
    x=t.zeros(30,device=b.device,dtype=t.float64,requires_grad=True)
    t.manual_seed(seed(slot));t.cuda.manual_seed_all(seed(slot))
    def adam(joint):
        return t.optim.Adam([{'params':[v],'lr':LRV},{'params':[x],'lr':LRX}] if joint else [x],
            lr=LRX if joint else LRS,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False)
    opt=adam(True);sopt=None
    b.event(dict(kind='initialized',seed=seed(slot),fraction=b.snap(frac),x=b.snap(x),v=b.snap(v)))
    for kind,k,j in phases():
        if kind=='sync' and j==0:sopt=adam(False)
        current=opt if kind=='joint' else sopt;current.zero_grad(set_to_none=True)
        b.event(dict(kind='optimization_attempt',phase=kind,iteration=k,sync_step=j,x=b.snap(x),v=b.snap(v)))
        if kind=='joint':loss,dyn,dense=b.soft(v,x)
        else:
            terminal,_=b.serial(x,'sync');loss=(terminal-g).square().mean();dyn=dense=None
        b.event(dict(kind='backward_attempt',phase=kind,iteration=k,sync_step=j,loss=b.snap(loss),
            dyn=None if dyn is None else b.snap(dyn),dense=None if dense is None else b.snap(dense)))
        need(bool(t.isfinite(loss)),'finite scalar objective');loss.backward();b.check()
        need(x.grad is not None and bool(t.isfinite(x.grad).all()) and
            (kind!='joint' or v.grad is not None and bool(t.isfinite(v.grad).all())),'finite native gradients')
        b.event(dict(kind='backward_return',gx=b.snap(x.grad),gv=b.snap(v.grad) if kind=='joint' else None))
        beforex=optimizer_snapshot(current,x,b.snap);beforev=optimizer_snapshot(current,v,b.snap) if kind=='joint' else None
        b.event(dict(kind='optimizer_attempt',phase=kind,iteration=k,sync_step=j))
        current.step();b.check();proposal=b.snap(x)
        with t.no_grad():x.copy_(b.tensor(project(x.detach().cpu().numpy()),t.float64))
        b.event(dict(kind='optimizer_return',x_before=beforex,v_before=beforev,x_after=optimizer_snapshot(current,x,b.snap),
            v_after=optimizer_snapshot(current,v,b.snap) if kind=='joint' else None,proposal=proposal,x=b.snap(x),v=b.snap(v)))
        if kind=='joint':
            with t.no_grad():noise=SIGMA*t.randn_like(v);v.add_(noise)
            b.event(dict(kind='noise',noise=b.snap(noise),v=b.snap(v)))
        if kind=='sync' and j==SYNCSTEPS-1:
            x=x.detach().clone().requires_grad_(True);opt=adam(True)
            b.event(dict(kind='joint_optimizer_reset',after=k+1,x=b.snap(x),v=b.snap(v)))
    commands=chart(x.detach().cpu().numpy()).reshape(15,2)
    b.event(dict(kind='actions_frozen',x=b.snap(x),commands=pack(commands)))
    with t.no_grad():
        _,whole=b.serial(x,'whole');_,segmented=b.serial(x,'segmented',True)
    equal(whole,segmented,'all3 fresh whole/segmented contexts exact')
    need(b.n==budget()['predict_API'],'complete2706 native calls/root')
    result=dict(root_slot=slot,task_id=inputs.core.task_id(task),logits=x.detach().cpu().tolist(),commands=commands.tolist(),
        virtual=v.detach().cpu().tolist(),whole=whole,segmented=segmented,
        terminal_Q=float(np.mean((np.asarray(whole[-1]['latent_history'][-1])-np.asarray(task['goal']))**2)),
        resources=budget(),physical_release=False,P1_admission=False,P2_admission=False,official_reproduction=False)
    b.event(dict(kind='root_complete',result_sha256=digest(result)));return result


def journal_lines(path):
    """Single bounded gzip, incremental parse; never load the whole trace to RAM."""
    d=zlib.decompressobj(31);pending=b'';total=0
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(64*1024),b''):
            need(not d.eof,'no gzip trailing bytes/second stream')
            out=d.decompress(chunk,16*1024**2)
            need(not d.unconsumed_tail and not d.unused_data,'gzip expansion/trailing cap')
            total+=len(out);need(total<=2*1024**3,'bounded expanded trace2GiB')
            pending+=out
            while b'\n' in pending:
                line,pending=pending.split(b'\n',1);need(line and len(line)<=1024**2,'bounded nonempty event')
                yield line
            need(len(pending)<=1024**2,'bounded partial event')
    need(d.eof and not pending and not d.unused_data and not d.unconsumed_tail,'complete single gzip LF journal')


def count_event(counts,payload):
    e=payload.get('event',payload);key=(str(payload['root_slot'])+':' if 'root_slot' in payload else '')+e['kind']
    if e['kind'] in ('module_attempt','module_return'):key+=':'+e['resource']
    counts[key]=counts.get(key,0)+1


def expected_census():
    counts=dict(load_intent=2,consumed_buffer_frozen=2)
    one=dict(initialized=1,optimization_attempt=900,backward_attempt=900,chart_backward=900,
        backward_return=900,optimizer_attempt=900,optimizer_return=900,noise=750,
        joint_optimizer_reset=15,actions_frozen=1,root_complete=1,
        forward_attempt=2706,forward_return=2706,predict_attempt=2706,predict_return=2706)
    for r in ('action_encoder','predictor','pred_proj'):
        one['module_attempt:'+r]=one['module_return:'+r]=2706
    for s in (0,1):counts.update({str(s)+':'+k:v for k,v in one.items()})
    return counts


class Cursor:
    def __init__(self,events):self.events=iter(events);self.n=0;self.tail='0'*64;self.root=None;self.counts={}
    def get(self,kind):
        try:line=next(self.events)
        except StopIteration:raise ValueError('missing event '+kind) from None
        e=inputs.parse(line);need(set(e)=={'index','previous','payload','sha256'} and
            type(e['index']) is int and e['index']==self.n and e['previous']==self.tail,'journal full ordinal/chain')
        need(raw(e)==line+b'\n' and digest({k:v for k,v in e.items() if k!='sha256'})==e['sha256'],'canonical event hash')
        self.tail=e['sha256'];self.n+=1;v=e['payload'];count_event(self.counts,v)
        if self.root is not None:
            need(set(v)=={'root_slot','event'} and v['root_slot']==self.root,'root exact scope/denominator')
            v=v['event']
        need(v['kind']==kind,'unknown/out-of-order event: '+str(v.get('kind'))+' expected '+kind)
        return v
    def expect(self,v):equal(self.get(v['kind']),v,'complete event/fields')
    def done(self):
        try:next(self.events)
        except StopIteration:return
        raise ValueError('extra event after terminal complete')


class Saved:
    def __init__(self,task,cursor):self.task=task;self.c=cursor;self.n=0
    def step(self,z,h,a,label):
        e=self.c.get('forward_attempt');equal(e,dict(kind='forward_attempt',index=self.n,label=label,z=pack(z),h=pack(h),raw=pack(a)),'actual full H3/raw5 request')
        mean,std=(np.tile(self.task[k],5).astype(np.float32) for k in ('action_mean','action_std'))
        tokens=(a.reshape(10)-mean)/std;outgoing=np.concatenate((h,tokens[None,None]),1)
        returned=[]
        for k,name in enumerate(('action_encoder','predictor','pred_proj')):
            if k==1:self.c.expect(dict(kind='predict_attempt',index=self.n))
            m=self.c.get('module_attempt');need(set(m)=={'kind','index','resource','inputs'} and m['index']==self.n and m['resource']==name,'module census')
            ins=[unpack(d) for d in m['inputs']]
            expected=[outgoing] if k==0 else [z,returned[0]] if k==1 else [returned[1].reshape(3,192)]
            need(len(ins)==len(expected) and all(np.array_equal(x,y) and x.dtype==y.dtype for x,y in zip(ins,expected,strict=True)),'all native inputs/normalizer/CLS projector wiring')
            ret=self.c.get('module_return');need(set(ret)=={'kind','index','resource','output'} and ret['index']==self.n and ret['resource']==name,'module exact paired return')
            returned.append(unpack(ret['output'],(3,192) if k==2 else (1,3,192),np.float32))
        pred=returned[-1].reshape(1,3,192)
        self.c.expect(dict(kind='predict_return',index=self.n,output=pack(pred)))
        z=np.concatenate((z[:,1:],pred[:,-1:]),1);h=outgoing[:,1:]
        self.c.expect(dict(kind='forward_return',index=self.n,z=pack(z),h=pack(h)));self.n+=1
        return z,h
    def serial(self,x,label,segmented=False):
        z,h=(arr(self.task['root_context'][k],s)[None] for k,s in (('latent_history',(3,192)),('action_history',(2,10))))
        paths=[]
        for k,a in enumerate(chart(x)):
            if segmented and paths:z,h=(arr(paths[-1][key],s)[None] for key,s in (('latent_history',(3,192)),('action_history',(2,10))))
            z,h=self.step(z,h,a,label+'/'+str(k));paths.append(contexts(z,h))
        return z[:,-1],paths
    def soft(self,v,x):
        root=self.task['root_context'];a=chart(x);tokens=np.concatenate([inputs.core.normalized_tokens(self.task,aa) for aa in a])
        preds=[]
        for k in range(3):
            c=lifted_context(root,v[0],k,tokens)
            z=arr(c['latent_history'],(3,192))[None];h=arr(c['action_history'],(2,10))[None]
            out,_=self.step(z,h,a[k],'lifted/'+str(k));preds.append(out[:,-1])
        return preds


def close_arithmetic(actual,expected,*,eps=np.finfo(np.float32).eps,ops=32):
    """Only saved elementwise/reduction/Adam rounding, NEVER neural parity."""
    actual=np.asarray(actual);expected=np.asarray(expected)
    need(actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all() and
        np.all(abs(actual-expected)<=ops*eps*(abs(expected)+1e-20)),'saved arithmetic rounding bound')
def mse(a,b):return np.mean((np.asarray(a,np.float32)-np.asarray(b,np.float32))**2,dtype=np.float32)
def loss_check(e,preds,v,goal,joint):
    value=float(unpack(e['loss'],(),np.float32))
    if joint:
        dyn=np.float32(0);dense=np.float32(0)
        for k,p in enumerate(preds):
            dyn=np.float32(dyn+mse(p,v[:,k] if k<2 else goal));dense=np.float32(dense+mse(p,goal))
        close_arithmetic(float(unpack(e['dyn'],(),np.float32)),dyn,ops=256)
        close_arithmetic(float(unpack(e['dense'],(),np.float32)),dense,ops=256)
        close_arithmetic(value,np.float32(dyn+dense),ops=512)
    else:
        need(e['dyn'] is e['dense'] is None,'sync terminal loss only');close_arithmetic(value,mse(preds[-1],goal),ops=256)
def blank(shape,dtype):return dict(step=0,m=pack(np.zeros(shape,dtype)),q=pack(np.zeros(shape,dtype)))
def adam_check(before,after,param,grad,proposal,lr):
    need(set(before)==set(after)=={'step','m','q'} and type(after['step']) is int and after['step']==before['step']+1,'installed Adam step chronology')
    dtype=param.dtype;eps=np.finfo(dtype).eps
    m=unpack(before['m'],param.shape,dtype);q=unpack(before['q'],param.shape,dtype)
    # Fused=False/foreach=False torch uses lerp for first moment; allow only
    # operation-count rounding, not a tolerance on model predictions or gradients.
    want_m=m+.1*(grad-m);want_q=.999*q+.001*grad*grad
    am=unpack(after['m'],param.shape,dtype);aq=unpack(after['q'],param.shape,dtype)
    close_arithmetic(am,want_m,eps=eps,ops=64);close_arithmetic(aq,want_q,eps=eps,ops=64)
    t=after['step'];update=lr/(1-.9**t)*am/(np.sqrt(aq)/np.sqrt(1-.999**t)+1e-8)
    # Bound absolute cancellation by the magnitudes before subtraction.
    want=param-update;error=128*eps*(abs(param)+abs(update)+1e-20)
    need(np.isfinite(proposal).all() and np.all(abs(proposal-want)<=error),'Adam proposal vs saved moments/gradient')


def validate_root(task,slot,c,result):
    c.root=slot;b=Saved(task,c);root=arr(task['root_context']['latent_history'],(3,192));goal=arr(task['goal'],(192,))[None]
    # Preserve the actual torch.linspace fractions; verify mathematical thirds
    # separately, without forcing NumPy's different endpoint rounding.
    init=c.get('initialized');need(set(init)=={'kind','seed','fraction','x','v'} and init['seed']==seed(slot),'fixed seed/init')
    frac=unpack(init['fraction'],(1,2,1),np.float32)
    close_arithmetic(frac,np.array([1/3,2/3]).reshape(1,2,1),ops=8)
    x=unpack(init['x'],(30,),np.float64);v=unpack(init['v'],(1,2,192),np.float32)
    need(not x.any() and np.array_equal(v,root[-1,None,None]+frac*(goal[:,None]-root[-1,None,None])),'zero logits/exact saved-fraction linear interpolation')
    mx=blank((30,),np.float64);mv=blank((1,2,192),np.float32)
    for kind,k,j in phases():
        if kind=='sync' and j==0:mx=blank((30,),np.float64)
        c.expect(dict(kind='optimization_attempt',phase=kind,iteration=k,sync_step=j,x=pack(x),v=pack(v)))
        preds=b.soft(v,x) if kind=='joint' else [b.serial(x,'sync')[0]]
        loss=c.get('backward_attempt');need(set(loss)=={'kind','phase','iteration','sync_step','loss','dyn','dense'} and
            (loss['phase'],loss['iteration'],loss['sync_step'])==(kind,k,j),'one scalar backward after3 forwards')
        loss_check(loss,preds,v,goal,kind=='joint')
        rb=c.get('chart_backward');need(set(rb)=={'kind','raw_gradient'},'one raw chart gradient')
        gr=unpack(rb['raw_gradient'],(3,5,2),np.float32)
        grads=c.get('backward_return');need(set(grads)=={'kind','gx','gv'},'gradient closure')
        gx=unpack(grads['gx'],(30,),np.float64)
        need(np.array_equal(gx,gr.reshape(30).astype(np.float64)*(1-np.tanh(x)**2)),'exact float64 tanh chain once')
        gv=None
        if kind=='joint':
            gv=unpack(grads['gv'],(1,2,192),np.float32)
            expected=np.stack([2/192*(v[:,kk]-preds[kk]) for kk in range(2)],1)
            close_arithmetic(gv,expected,ops=16)
        else:need(grads['gv'] is None,'no sync virtual optimization')
        c.expect(dict(kind='optimizer_attempt',phase=kind,iteration=k,sync_step=j))
        op=c.get('optimizer_return');need(set(op)=={'kind','x_before','v_before','x_after','v_after','proposal','x','v'},'full optimizer evidence')
        equal(op['x_before'],mx,'actual persisted action Adam memory');proposal=unpack(op['proposal'],(30,),np.float64)
        adam_check(mx,op['x_after'],x,gx,proposal,LRX if kind=='joint' else LRS);mx=op['x_after']
        xx=unpack(op['x'],(30,),np.float64);need(np.array_equal(xx,project(proposal)) and np.linalg.norm(xx)<=BALL,'one global30 projection')
        vv=unpack(op['v'],(1,2,192),np.float32)
        if kind=='joint':
            equal(op['v_before'],mv,'persisted virtual Adam memory');adam_check(mv,op['v_after'],v,gv,vv,LRV);mv=op['v_after']
        else:need(op['v_before'] is op['v_after'] is None and np.array_equal(vv,v),'sync does not refresh virtual')
        x,v=xx,vv
        if kind=='joint':
            noise=c.get('noise');need(set(noise)=={'kind','noise','v'},'only virtual noise')
            nn=unpack(noise['noise'],(1,2,192),np.float32);nv=unpack(noise['v'],(1,2,192),np.float32)
            need(np.array_equal(nv,v+nn),'literal saved virtual noise addition');v=nv
        if kind=='sync' and j==SYNCSTEPS-1:
            c.expect(dict(kind='joint_optimizer_reset',after=k+1,x=pack(x),v=pack(v)));mx=blank((30,),np.float64);mv=blank((1,2,192),np.float32)
    commands=chart(x).reshape(15,2);c.expect(dict(kind='actions_frozen',x=pack(x),commands=pack(commands)))
    _,whole=b.serial(x,'whole');_,segmented=b.serial(x,'segmented',True)
    equal(whole,segmented,'fresh whole and segmented all3 exact');need(b.n==2706,'all2706 forwards/root')
    rebuilt=dict(root_slot=slot,task_id=inputs.core.task_id(task),logits=x.tolist(),commands=commands.tolist(),
        virtual=v.tolist(),whole=whole,segmented=segmented,
        terminal_Q=float(np.mean((np.asarray(whole[-1]['latent_history'][-1])-np.asarray(task['goal']))**2)),
        resources=budget(),physical_release=False,P1_admission=False,P2_admission=False,official_reproduction=False)
    equal(result,rebuilt,'final last action, no best/outcome selection');c.expect(dict(kind='root_complete',result_sha256=digest(rebuilt)))
    c.root=None;return rebuilt


def precision(t):
    value=dict(matmul_TF32=t.backends.cuda.matmul.allow_tf32,cudnn_TF32=t.backends.cudnn.allow_tf32,
        autocast=t.is_autocast_enabled('cuda'),MATH=t.backends.cuda.math_sdp_enabled(),
        FLASH=t.backends.cuda.flash_sdp_enabled(),MEM_EFFICIENT=t.backends.cuda.mem_efficient_sdp_enabled(),
        CUDNN=t.backends.cuda.cudnn_sdp_enabled())
    equal(value,dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,MATH=True,FLASH=False,MEM_EFFICIENT=False,CUDNN=False),'original native precision guard')
    return value
def tasks(p):
    out=[]
    for s in p['slots']:
        e=inputs.parse(inputs.read_bound(s['task_file'],s['task_sha256'],compressed=True));t=e['task']
        equal(s['model_input']['root_context'],t['root_context'],'root input');equal(s['model_input']['goal'],t['goal'],'original goal')
        need(inputs.core.task_id(t)==s['task_id'],'original task identity');out.append(t)
    return out


def run(prepared,expected,output,authorization):
    need(authorization==AUTH,'explicit MAIN authorization before I/O or torch import')
    start=time.monotonic();p=verify(prepared,expected);store=Store(output,start)
    rows=[dict(root_slot=i,condition='off',task_id=s['task_id'],status='not_started',result_file=None) for i,s in enumerate(p['slots'])]
    errors=[];model=None;before=after=host=None;guard_before=guard_after=None;source_ok=False;peak=None
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        # Actual immutable buffers, strict compatibility remap and native tensor
        # hash use the already reviewed loader under its OWN receipt schema.
        lp=dict(p,bindings_sha256=p['bindings'])
        model,_=upstream.loader.loaded_model(lp,torch,device,store,expected)
        for module in model.modules():need(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'presealed installed native class source')
        before=installed.signal.model_hash(model);need(before['original_state_dict']==p['expected_model_hash'],'original state dict hash')
        with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            guard_before=precision(torch)
            for i,(task,row) in enumerate(zip(tasks(p),rows,strict=True)):
                row['status']='attempted'
                def emit(e):store.event(dict(root_slot=i,event=e))
                backend=Native(model,device,task,emit,store)
                with backend.observe():result=optimize(task,i,backend)
                name=f'root{i}.json';store.put(name,result);row.update(status='complete',result_file=name)
            guard_after=precision(torch)
        torch.cuda.synchronize(device);peak=dict(allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device))
        need(peak['allocated']<=4*1024**3 and peak['reserved']<=6*1024**3,'final CUDA peak caps')
        equal(store.counts,expected_census(),'actual complete forward/backward/optimizer/module census')
    except BaseException as exc:
        errors.append(dict(stage='execution',type=type(exc).__name__,message=str(exc)))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:
                upstream.loader.enc.frozen(model);after=installed.signal.model_hash(model)
                need(after==before,'all native model tensors/buffers unchanged')
            except BaseException as exc:errors.append(dict(stage='model_final',message=str(exc)))
        try:checkbindings(p['bindings']);store.check();source_ok=True
        except BaseException as exc:errors.append(dict(stage='source_wall_final',message=str(exc)))
        try:store.close()
        except BaseException as exc:errors.append(dict(stage='journal_final',message=str(exc)))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='complete' for r in rows)
        report=dict(schema=NAME+'.report',status='normalized_model_variant_complete_not_admitted' if complete else 'failed_partial',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=contract(),rows=rows,errors=errors,
            sources_verified=source_ok,model_before=before,model_after=after,host=host,CUDA_peak_bytes=peak,
            precision_before=guard_before,precision_after=guard_after,wall_seconds=time.monotonic()-start,
            event_counts=store.counts,journal_tail=store.tail,journal_events=store.seq,
            artifact_sha256={f.name:filehash(f) for f in store.path.iterdir()},
            actual_new_physics=0,actual_new_encoder=0,actual_new_projector=0,
            physical_release=False,P1_admission=False,P2_admission=False,official_reproduction=False,equal_budget_to_1902=False)
        store.put('report.json',report,final=True)
    need(complete,'failed partial preserved, no retry/refund: '+str(errors))
    return dict(output=str(store.path),report_sha256=filehash(store.path/'report.json'),status=report['status'])


def consume(output,expected):
    output=Path(output).absolute();need(filehash(output/'report.json')==expected,'external actual report SHA');r=readjson(output/'report.json')
    need(r['schema']==NAME+'.report' and r['status']=='normalized_model_variant_complete_not_admitted' and r['errors']==[] and
        r['sources_verified'] is True and r['model_before'] is not None and r['model_before']==r['model_after'], 'complete real variant not failed partial')
    p=verify(r['prepared'],r['prepare_sha256']);equal(r['contract'],contract(),'no protocol drift')
    need(r['model_before']['original_state_dict']==p['expected_model_hash'] and r['actual_new_physics']==r['actual_new_encoder']==r['actual_new_projector']==0 and
        all(r[k] is False for k in ('physical_release','P1_admission','P2_admission','official_reproduction','equal_budget_to_1902')),'original model and evidence boundaries')
    expected_guard=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,MATH=True,FLASH=False,MEM_EFFICIENT=False,CUDNN=False)
    equal(r['precision_before'],expected_guard,'precision at entry');equal(r['precision_after'],expected_guard,'precision at exit')
    need(0<=r['wall_seconds']<WALL and r['CUDA_peak_bytes']['allocated']<=4*1024**3 and r['CUDA_peak_bytes']['reserved']<=6*1024**3,'bounded native wall/peak')
    names={'binding.json','loader.json','events.jsonl.gz','root0.json','root1.json'}
    need(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'exact bounded artifact census')
    for n,h in r['artifact_sha256'].items():need(filehash(output/n)==h,'whole actual artifact SHA')
    equal(readjson(output/'binding.json'),dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'real execution binding')
    load=readjson(output/'loader.json');upstream.loader.validate_loader(load,dict(p,bindings_sha256=p['bindings']),r['prepare_sha256'])
    c=Cursor(journal_lines(output/'events.jsonl.gz'))
    for f in load['files']:
        c.expect(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']))
        c.expect(dict(kind='consumed_buffer_frozen',receipt=f))
    need(len(r['rows'])==2,'complete two-root denominator');plans=[]
    for i,(task,row,s) in enumerate(zip(tasks(p),r['rows'],p['slots'],strict=True)):
        equal(row,dict(root_slot=i,condition='off',task_id=s['task_id'],status='complete',result_file=f'root{i}.json'),'fixed source slots/order')
        result=validate_root(task,i,c,readjson(output/row['result_file']))
        plan=dict(schema=NAME+'.model_plan',root_slot=i,condition='off',task_id=s['task_id'],
            task_file=s['task_file'],task_sha256=s['task_sha256'],source_report_sha256=inputs.SOURCE,
            blind_pointer=s['blind_pointer'],original_context=deepcopy(task['root_context']),goal=deepcopy(task['goal']),
            Q=deepcopy(task['projection']),normalization=inputs.normalization(),model_sha256=p['expected_model_hash'],
            raw_steps=15,commands=result['commands'],raw_float32_sha256=inputs.sha(np.asarray(result['commands'],'<f4').tobytes()),
            logits=result['logits'],terminal=result['whole'][-1],model_terminal_Q=result['terminal_Q'],
            planner_report_sha256=expected,physical_release=False,official_reproduction=False,P1_admission=False,P2_admission=False)
        plans.append(dict(plan,selection_digest=digest(plan)))
    c.done();need(c.n==r['journal_events'] and c.tail==r['journal_tail'],'complete journal head/tail')
    equal(c.counts,r['event_counts'],'full physical-free actual census');equal(c.counts,expected_census(),'all resources/modules/no extra dispatch')
    checkbindings(p['bindings'])
    for n,h in r['artifact_sha256'].items():need(filehash(output/n)==h,'final actual file closure')
    need(filehash(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='history_joint30_variant_verified_not_admitted',plans=plans,
        planner_report_sha256=expected,prepare_sha256=r['prepare_sha256'],resources=contract()['total'],
        new_neural_calls=0,new_physics=0,physical_release=False,P1_admission=False,P2_admission=False,
        official_reproduction=False,equal_budget_to_1902=False,errors=[])


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);sub=q.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    for mode in ('verify','run_cuda'):
        a=sub.add_parser(mode);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if mode=='run_cuda':a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=sub.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha',required=True);a.add_argument('--receipt-output')
    a=q.parse_args(argv)
    if a.cmd=='prepare':v=metadata(a.output,build())
    elif a.cmd=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='prepared_not_dispatched',contract=p['contract'])
    elif a.cmd=='run_cuda':v=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:
        v=consume(a.output,a.report_sha)
        if a.receipt_output:v=metadata(a.receipt_output,v)
    print(raw(v).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())
