"""Installed LeWM CEM/get_cost cached-H3 known2 normalized slice; MAIN CUDA only."""
import argparse
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import numpy as np
from experiments.spectral_reachability import p1_moving_sampling_baselines_cuda_20261027_v1 as inputs

NAME='agent_lewm_native_cem_slice_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
SITE=ROOT/'.venv/lib/python3.12/site-packages'
CEM=SITE/'stable_worldmodel/solver/cem.py'
WM=SITE/'stable_worldmodel/wm/lewm/lewm.py'
PINS={Path(inputs.__file__):'f30a1c5ac5ef618cfe3add37326aae29513278fe7c4319b5a743d9f6422300e6',
      CEM:'a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b',
      CEM.parent/'utils.py':'d30c6225a9c4c83782132180a94c94188b07e0381d2c938c658fe5a49aed48e1',
      WM:'ccc2ac2880880f85ea71a5baa2152ba97e3f6dc5cb546d27570795aa6020d3d2'}
AUTH='MAIN_LEWM_NATIVE_CEM_NORMALIZED_HOSTCUDA_ONLY'
CAP,RESERVE,FLOOR,WALL=1024**3,32*1024**2,200*1024**2,900
SAMPLES,STEPS,ELITE,SEED=300,30,30,1234
KEYS=('candidates','costs','topk_vals','topk_inds','topk_candidates','mean','var','prev_mean','prev_var')
KINDS=('action_encoder','predictor','pred_proj')


def need(ok,msg):
    if not ok:raise ValueError(msg)


def raw(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p,h):
    p=Path(p);need(p.resolve()==p.absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size<=32*1024**2,'bounded regular JSON')
    need(sha(p)==h,'external metadata SHA')
    def unique(xs):
        d={}
        for k,v in xs:need(k not in d,'duplicate JSON field');d[k]=v
        return d
    return json.loads(p.read_bytes(),object_pairs_hook=unique,parse_constant=lambda _:need(False,'nonfinite JSON'))


def protocol():
    return dict(roots=[0,1],known_only=True,fresh=False,condition='off',history=[0,5,10],horizon=2,packet_steps=5,
        solver='installed CEMSolver unchanged',samples=SAMPLES,iterations=STEPS,topk=ELITE,seed=SEED,var_scale=1.,
        parameter_source='pinned CEMSolver.__init__ defaults; H2 inherited actual task',
        action_domain='unclipped normalized Gaussian width10, no raw ball/tanh/clamp',
        initialization='explicit full normalized zero mean, no actor/witness',
        cached='actual encoded history3 and goal; actual transformed RGB3 and goal still supplied',
        warmup='actual raw10 normalized once in original task; prepend incoming2 to future2 before official get_cost',
        criterion='official float32 terminal sum192; separate float64 sum/mean arithmetic diagnostics',
        return_action='official final elite mean, not evaluated winner; elite_average_cost is NOT returned_mean_cost',
        final_mean_evaluated=False,extra_replay=0,
        per_root=dict(cost_API=STEPS,rollout_API=STEPS,action_encoder=STEPS,predictor=2*STEPS,pred_proj=2*STEPS,
            predictor_samples=2*STEPS*SAMPLES,AE_samples=STEPS*SAMPLES,AE_tokens=4*STEPS*SAMPLES,
            predictor_context_tokens=6*STEPS*SAMPLES,pred_proj_rows=6*STEPS*SAMPLES),
        encoder=0,image_projector=0,AD=0,physics=0,training=0,Qfit=False,FLOPs=None,
        output_cap=CAP,reserve=RESERVE,floor=FLOOR,wall_seconds=WALL,cuda_allocated=3*1024**3,cuda_reserved=4*1024**3,
        source_scope='local installed official equations; not published-cell reproduction or original eval data contract',
        same_protocol_as_bounded_CEM=False,physical_release=False,P1_admission=False,scientific_admission=False)


def fixed_defaults():
    need((SAMPLES,STEPS,ELITE,SEED)==(300,30,30,1234),'production fixed installed defaults')


def transformed_rgb(rgb):
    # Bound 224 inputs: Resize224 is identity. ToDtype scales by multiplication,
    # then Normalize subtracts and divides separately (no fused expression).
    need(rgb.dtype==np.uint8 and rgb.shape==(224,224,3),'actual RGB224')
    a=rgb.transpose(2,0,1).astype(np.float32)*np.float32(1/255)
    a=a-np.asarray([.485,.456,.406],np.float32)[:,None,None]
    return a/np.asarray([.229,.224,.225],np.float32)[:,None,None]


def build():
    fixed_defaults()
    for p,h in PINS.items():need(sha(p)==h,'frozen primitive/official source')
    p=inputs.build(('CEM',))  # saved identity closure only, no owner consume/fullstrict
    enc=inputs.read(inputs.REPORT)
    packet=inputs.moving.build(enc['arguments']['approved_manifest'],enc['arguments']['manifest_sha'])
    need(len(p['slots'])==len(packet['slots'])==2,'fixed original known2, no subset')
    slots=[]
    for s,b in zip(p['slots'],packet['slots'],strict=True):
        need(s['source_slot']==b['source_slot']==len(slots) and s['blind_sha256']==b['blind_sha256'],'actual task/blind exact join')
        t=inputs.load_export(s['task_export'],s['task_sha256'])['gn_task']
        slots.append(dict(s,blind_npz=b['blind_npz'],pixel_sha256=b['pixel_sha256'],
            emb=t['root_context']['latent_history'],warm=t['root_context']['action_history'],goal_emb=t['goal']))
    sources=dict(p['bindings_sha256']);sources.update(packet['bindings_sha256'])
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),*PINS):sources[str(f.resolve())]=sha(f)
    for rel in ('stable_worldmodel/solver/callbacks/__init__.py','stable_worldmodel/solver/solver.py',
                'stable_worldmodel/protocols.py','torch/nn/modules/module.py','torch/nn/functional.py',
                'stable_pretraining/data/dataset_stats.py','torchvision/transforms/v2/functional/_misc.py',
                'torchvision/transforms/v2/functional/_geometry.py'):
        f=SITE/rel;sources[str(f)]=sha(f)
    for rel in ('experiments/spectral_reachability/preflight_pusht_d2p_p0.py','experiments/run_pusht_counterfactual.py',
                'experiments/checkpoint_compat.py','experiments/gpu_preflight.py'):
        f=ROOT/rel;sources[str(f)]=sha(f)
    need(all(sha(f)==h for f,h in sources.items()),'complete original source/model/input byte bindings')
    return dict(schema=NAME+'.prepare',protocol=protocol(),slots=slots,sources=sources,policy=p['policy'],
        expected_model_hash=p['expected_model_hash'],inherited=dict(encoder=8,projector=8,report=str(inputs.REPORT),
            report_sha256=inputs.REPORT_SHA,sourcepack_sha256=inputs.SOURCE_AUTHORITY_SHA,training_provenance_verified=False))


def own(p):
    p=Path(p).absolute();need(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and
        not any(x.is_symlink() for x in (p,*p.parents)),'own nonsymlink direct results child');return p


def durable(p,b):
    with Path(p).open('xb') as f:need(f.write(b)==len(b),'full exclusive write');f.flush();os.fsync(f.fileno())
    fd=os.open(Path(p).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def prepare(output):
    p=own(output);v=build();b=raw(v);need(len(b)<=1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'prepare1MiB/floor')
    durable(p,b);return dict(path=str(p),sha256=sha(p))


def verify(prepared,h):
    v=read(prepared,h);need(v==build(),'whole saved prepare closure; no nested fullstrict');return v


class Store:
    def __init__(self,path,started=None):
        self.path=own(path);self.started=time.monotonic() if started is None else started
        need(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'exclusive1GiB+floor')
        self.path.mkdir();self.inventory={};self.events=[];self.total=0;self.tail=None;self.extra_guard=lambda:None
    def guard(self,size=0,final=False):
        need(final or time.monotonic()-self.started<WALL,'cooperative900s including prepare/load')
        need(self.total+size<=CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free>=FLOOR+size,'all bytes/cap/reserve/floor')
        if not final:self.extra_guard()
    def put(self,name,b,final=False):
        need(Path(name).name==name and name not in ('','.','..') and name not in self.inventory,'flat unique artifact')
        self.guard(len(b),final);durable(self.path/name,b);self.total+=len(b)
        self.inventory[name]=dict(sha256=sha(self.path/name),file_bytes=len(b));return dict(file=name,**self.inventory[name])
    def array(self,value):
        a=value if type(value) is np.ndarray else value.detach().cpu().numpy()
        need(a.dtype in (np.float32,np.int64) and np.isfinite(a).all(),'original finite native dtype')
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False);rec=self.put(f'array_{len(self.inventory):06d}.npy',buf.getvalue())
        return dict(rec,shape=list(a.shape),dtype=a.dtype.str,nbytes=a.nbytes)
    def event(self,kind,final=False,**fields):
        e=dict(sequence=len(self.events),previous_sha256=self.tail,kind=kind,**fields)
        e['event_sha256']=hashlib.sha256(raw(e)).hexdigest();b=raw(e);self.guard(len(b),final)
        with (self.path/'journal.jsonl').open('ab') as f:need(f.write(b)==len(b),'one JSONL newline');f.flush();os.fsync(f.fileno())
        self.total+=len(b);self.tail=e['event_sha256'];self.events.append(e)


def load_array(folder,rec,inventory):
    need(Path(rec['file']).name==rec['file'] and inventory.get(rec['file'])=={k:rec[k] for k in ('sha256','file_bytes')},'array inventory role')
    p=Path(folder)/rec['file'];need(not p.is_symlink() and p.stat().st_size==rec['file_bytes']<=16*1024**2 and sha(p)==rec['sha256'],'bounded actual array bytes')
    b=io.BytesIO(p.read_bytes());version=np.lib.format.read_magic(b)
    need(version in ((1,0),(2,0)),'bounded NPY version')
    shape,fortran,dtype=(np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(b)
    need(not fortran and dtype in (np.dtype('float32'),np.dtype('int64')) and
        list(shape)==rec['shape'] and dtype.str==rec['dtype'] and
        int(np.prod(shape,dtype=object))*dtype.itemsize==rec['nbytes']==rec['file_bytes']-b.tell(),'bounded header before allocation')
    b.seek(0);a=np.load(b,allow_pickle=False)
    need(b.tell()==rec['file_bytes'] and list(a.shape)==rec['shape'] and a.dtype.str==rec['dtype'] and a.nbytes==rec['nbytes'] and
        a.dtype in (np.float32,np.int64) and np.isfinite(a).all(),'single finite original NPY')
    return a


def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


class NativeTrace:
    def __init__(self,model,store,row):self.model,self.store,self.row=model,store,row;self.handles=[];self.pending={}
    def __enter__(self):
        for kind in (*KINDS,'encoder','projector'):
            def pre(module,args,k=kind):
                need(k in KINDS,'no new encoder/projector allowed')
                import torch
                need(torch.is_inference_mode_enabled() and not torch.is_grad_enabled() and not torch.is_autocast_enabled('cuda') and
                    not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'observed inference/TF32/autocast guard')
                need(all(x.device.type=='cuda' and x.dtype==torch.float32 and not x.requires_grad for x in args),'native CUDA float32 inputs')
                index=len(self.row['modules'])%5
                shapes=[[(SAMPLES,4,10)],[(SAMPLES,3,192)]*2,[(SAMPLES*3,192)],[(SAMPLES,3,192)]*2,[(SAMPLES*3,192)]]
                need(k==('action_encoder','predictor','pred_proj','predictor','pred_proj')[index] and
                    [tuple(x.shape) for x in args]==shapes[index],'native module order/shapes')
                self.store.guard();need(k not in self.pending,'no nested same module')
                e=dict(index=len(self.row['modules']),kind=k,iteration=self.row['cost_attempted']-1,returned=False,inputs=[])
                self.row['modules'].append(e);self.store.event('module_intent',index=e['index'],kind_name=k,root=self.row['root'])
                e['inputs']=[self.store.array(x) for x in args];self.pending[k]=e
            def post(module,args,out,k=kind):
                import torch
                expected=(SAMPLES,4,192) if k=='action_encoder' else ((SAMPLES,3,192) if k=='predictor' else (SAMPLES*3,192))
                need(out.device.type=='cuda' and out.dtype==torch.float32 and not out.requires_grad and tuple(out.shape)==expected,'native output device/dtype/shape')
                e=self.pending.pop(k);e['output']=self.store.array(out);e['returned']=True
                self.store.event('module_return',index=e['index'],kind_name=k,root=self.row['root'])
            self.handles.extend([getattr(self.model,kind).register_forward_pre_hook(pre),getattr(self.model,kind).register_forward_hook(post)])
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        if args[0] is None:need(not self.pending,'no module left pending')


def run(prepared,h,output,authorization):
    need(authorization==AUTH and not os.environ.get('PYTHONPATH'),'explicit MAIN CUDA-only authority/no import overlay')
    start=time.monotonic();m=verify(prepared,h);store=Store(output,start)
    rows=[dict(root=i,status='not_started',cost_attempted=0,cost_returned=0,costs=[],callbacks=[],modules=[]) for i in (0,1)]
    r=dict(schema=NAME+'.report',protocol=protocol(),prepared=str(Path(prepared).absolute()),prepare_sha256=h,rows=rows,errors=[],
        inherited=m['inherited'],model_before=None,model_after=None,source_verified=False,physical_release=False,scientific_admission=False)
    model=None
    try:
        store.event('started',prepare_sha256=h)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        need(os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','fixed requested CUBLAS workspace')
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from gymnasium.spaces import Box
        from stable_worldmodel.solver.cem import CEMSolver
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        r['cuda_preflight']=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        need(Path(inspect.getfile(CEMSolver)).resolve()==CEM,'actual installed CEM class')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        model=load_compatible_pretrained(m['policy'],device=device).eval().requires_grad_(False)
        need(Path(inspect.getfile(type(model))).resolve()==WM,'actual official LeWM class')
        r['model_before']=inputs.moving.low.tensor_hash(model);need(r['model_before']==m['expected_model_hash'],'original model parameters/buffers')
        def gpu_guard():need(torch.cuda.memory_allocated()<=3*1024**3 and torch.cuda.memory_reserved()<=4*1024**3,'CUDA allocated/reserved caps')
        store.extra_guard=gpu_guard
        transform=p0.image_transform()
        tensor=lambda a:torch.as_tensor(np.asarray(a,np.float32),device=device)
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            for slot,row in zip(m['slots'],rows,strict=True):
                packet=inputs.moving.load_blind(slot['blind_npz'],slot['blind_sha256'])
                need([inputs.moving.digest(x.tobytes()) for x in [*packet['rgb_history'],packet['goal_rgb']]]==slot['pixel_sha256'],'actual past/goal pixels, no dummy')
                history=torch.stack([p0._image_tensor(x,transform=transform,device=device)[0,0] for x in packet['rgb_history']])[None]
                goal=p0._image_tensor(packet['goal_rgb'],transform=transform,device=device)
                need(exact(history.cpu().numpy(),np.stack([transformed_rgb(x) for x in packet['rgb_history']])[None]) and
                    exact(goal.cpu().numpy(),transformed_rgb(packet['goal_rgb'])[None,None]),'actual source preprocessing exact formula, no numeric tolerance')
                info=dict(pixels=history,goal=goal,emb=tensor(slot['emb'])[None],goal_emb=tensor(slot['goal_emb'])[None,None])
                warm=tensor(slot['warm'])[None,None]
                row.update(status='attempted',base_inputs={k:store.array(v) for k,v in info.items()},warm=store.array(warm))
                class Cost:
                    def parameters(self):return model.parameters()
                    def get_cost(self,expanded,candidates):
                        store.guard();j=row['cost_attempted'];need(j<STEPS and tuple(candidates.shape)==(1,SAMPLES,2,10),'fixed actual candidate denominator')
                        row['cost_attempted']+=1;store.event('cost_intent',root=row['root'],iteration=j)
                        for k,v in info.items():need(torch.equal(expanded[k],v[:,None].expand(1,SAMPLES,*v.shape[1:])),'CEM delivered real cached infos')
                        actions=torch.cat((warm.expand(1,SAMPLES,2,10),candidates),dim=2)
                        e=dict(iteration=j,candidates=store.array(candidates),actions=store.array(actions),returned=False)
                        row['costs'].append(e);actual=dict(expanded);cost=model.get_cost(actual,actions)
                        need(tuple(cost.shape)==(1,SAMPLES) and actual['emb'] is expanded['emb'] and actual['pixels'] is expanded['pixels'],'cached official API result identity')
                        e.update(predicted_emb=store.array(actual['predicted_emb']),action_prefix=store.array(actual['action']),cost=store.array(cost),returned=True)
                        row['cost_returned']+=1;store.event('cost_return',root=row['root'],iteration=j);return cost
                class Recorder:
                    output_key=NAME
                    def reset(self):self.history=[]
                    def start_batch(self):return None
                    def end_solve(self):return None
                    def __call__(self,**kw):
                        need(kw['step']==len(row['callbacks']),'all native CEM callbacks once/in order')
                        row['callbacks'].append({k:store.array(kw[k]) for k in KEYS});store.event('cem_update',root=row['root'],iteration=kw['step'])
                solver=CEMSolver(Cost(),device=device,callbacks=[Recorder()])  # exact installed defaults
                solver.configure(action_space=Box(-np.inf,np.inf,shape=(1,2),dtype=np.float32),n_envs=1,config=SimpleNamespace(horizon=2,action_block=5))
                need((solver.num_samples,solver.n_steps,solver.topk,solver.batch_size,solver.var_scale,solver.horizon,solver.action_dim)==
                    (SAMPLES,STEPS,ELITE,1,1,2,10),'actual installed solver defaults/config')
                with NativeTrace(model,store,row):out=solver.solve(info,init_action=torch.zeros((1,2,10),device=device))
                row.update(actions=store.array(out['actions']),mean=store.array(out['mean'][0]),std=store.array(out['var'][0]),
                    elite_average_cost=out['costs'],status='normalized_mean_sealed')
                store.event('root_sealed',root=row['root'])
        r['model_after']=inputs.moving.low.tensor_hash(model);need(r['model_after']==r['model_before'],'unchanged full model')
        need(all(x['status']=='normalized_mean_sealed' and x['cost_attempted']==x['cost_returned']==STEPS and
            len(x['modules'])==5*STEPS and len(x['callbacks'])==STEPS for x in rows),'whole native job denominators')
        r['cuda_peak_allocated']=torch.cuda.max_memory_allocated();r['cuda_peak_reserved']=torch.cuda.max_memory_reserved()
        r['numerics']=dict(requested_SDP='MATH',kernel_certificate=False,observed_inference=True,observed_TF32=False,observed_autocast=False)
        need(all(sha(p)==value for p,value in m['sources'].items()),'all source/input/model files unchanged');r['source_verified']=True
    except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)))
    finally:
        if model is not None and r['model_after'] is None:
            try:r['model_after']=inputs.moving.low.tensor_hash(model)
            except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)))
        r['elapsed_seconds']=time.monotonic()-start
        if r['elapsed_seconds']>=WALL:r['errors'].append(dict(type='wall_cap',message='whole job900s'))
        r['status']='all_known2_native_normalized_CEM_recorded' if not r['errors'] and r['source_verified'] else 'partial_or_failed'
        store.event('finished',final=True,status=r['status']);journal=store.path/'journal.jsonl'
        store.inventory['journal.jsonl']=dict(sha256=sha(journal),file_bytes=journal.stat().st_size)
        r.update(artifacts=dict(store.inventory),journal_tail=store.tail)
        store.put('report.json',raw(r),final=True)
    return dict(status=r['status'],path=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def bounded_reduction(actual,reference,n):
    # Only non-neural CUDA reduction arithmetic; never a model parity threshold.
    need(actual.shape==reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all() and
        np.all(np.abs(actual.astype(np.float64)-reference)<=8*n*np.finfo(np.float32).eps*np.maximum(np.abs(reference),1.)), 'saved float32 reduction roundoff bound')


def validate_row(slot,row,load):
    need(row['root']==slot['source_slot'] and row['status']=='normalized_mean_sealed' and
        row['cost_attempted']==row['cost_returned']==STEPS and len(row['costs'])==len(row['callbacks'])==STEPS,'all native iteration slots')
    base={k:load(v) for k,v in row['base_inputs'].items()};warm=load(row['warm'])
    need(set(base)=={'pixels','goal','emb','goal_emb'} and base['pixels'].shape==(1,3,3,224,224) and base['goal'].shape==(1,1,3,224,224),'complete actual RGB inputs')
    packet=inputs.moving.load_blind(slot['blind_npz'],slot['blind_sha256'])
    need(exact(base['pixels'],np.stack([transformed_rgb(x) for x in packet['rgb_history']])[None]) and
        exact(base['goal'],transformed_rgb(packet['goal_rgb'])[None,None]),'saved transformed pixels exact source RGB preprocessing')
    need(exact(base['emb'],np.asarray(slot['emb'],np.float32)[None]) and exact(base['goal_emb'],np.asarray(slot['goal_emb'],np.float32)[None,None]) and
        exact(warm,np.asarray(slot['warm'],np.float32)[None,None]),'same original latent/goal/action history')
    mean=np.zeros((1,2,10),np.float32);std=np.ones_like(mean);scores=[];pos=0
    for j,(e,c) in enumerate(zip(row['costs'],row['callbacks'],strict=True)):
        v={k:load(x) for k,x in c.items()};candidate=load(e['candidates']);actions=load(e['actions']);pred=load(e['predicted_emb']);cost=load(e['cost'])
        need(e['iteration']==j and e['returned'] is True and set(c)==set(KEYS) and
            candidate.shape==(1,SAMPLES,2,10) and pred.shape==(1,SAMPLES,5,192) and cost.shape==(1,SAMPLES), 'full native H2 candidate output shapes')
        need(exact(candidate,v['candidates']) and exact(candidate[:,0],mean) and exact(v['prev_mean'],mean) and exact(v['prev_var'],std) and
            exact(actions[:,:,:2],np.broadcast_to(warm,(1,SAMPLES,2,10))) and exact(actions[:,:,2:],candidate) and exact(load(e['action_prefix']),actions[:,:,:3]),'actual incoming2/outgoing2 and CEM distribution chain')
        need(exact(pred[:,:,:3],np.broadcast_to(base['emb'][:,None],(1,SAMPLES,3,192))),'cached observed0/5/10 exact')
        modules=row['modules'][pos:pos+5];pos+=5
        need([x['kind'] for x in modules]==['action_encoder','predictor','pred_proj','predictor','pred_proj'] and
            all(x['returned'] is True and x['iteration']==j and x['index']==pos-5+k for k,x in enumerate(modules)), 'actual once-AE/two-predict/projection sequence')
        ae=modules[0];need(exact(load(ae['inputs'][0]),actions.reshape(SAMPLES,4,10)),'native AE full sequence')
        embedding=load(ae['output']);need(embedding.shape==(SAMPLES,4,192),'native AE width')
        for step in (0,1):
            p,proj=modules[1+2*step:3+2*step];pi=[load(x) for x in p['inputs']];po=load(p['output']);final=load(proj['output'])
            need(len(pi)==2 and po.shape==(SAMPLES,3,192) and final.shape==(SAMPLES*3,192) and
                exact(pi[0],pred[:,:,step:step+3].reshape(SAMPLES,3,192)) and exact(pi[1],embedding[:,step:step+3]) and
                exact(load(proj['inputs'][0]),po.reshape(SAMPLES*3,192)) and exact(final.reshape(SAMPLES,3,192)[:,-1],pred[:,:,3+step].reshape(SAMPLES,192)), 'native rolling/predict/proj exact')
        residual=pred[:,:,-1].astype(np.float64)-base['goal_emb'][:,0,None].astype(np.float64)
        scalar=np.sum(residual**2,axis=-1,dtype=np.float64);bounded_reduction(cost,scalar,194)
        need(exact(v['costs'],cost),'official criterion cost delivered unchanged')
        inds=v['topk_inds'];need(inds.shape==(1,ELITE) and inds.dtype==np.int64 and len(set(inds[0].tolist()))==ELITE and
            (inds>=0).all() and (inds<SAMPLES).all(),'all original elite indices')
        need(exact(v['topk_vals'],np.take_along_axis(cost,inds,axis=1)) and exact(v['topk_candidates'],candidate[np.arange(1)[:,None],inds]) and
            exact(np.sort(v['topk_vals'],axis=1),np.sort(cost,axis=1)[:,:ELITE]),'native elite selection, no inferred tie order')
        elite=v['topk_candidates'].astype(np.float64);bounded_reduction(v['mean'],elite.mean(axis=1),ELITE)
        bounded_reduction(v['var'],elite.std(axis=1,ddof=1),ELITE)
        mean,std=v['mean'],v['var'];scores.append(dict(iteration=j,sum192_float64=scalar.tolist(),mean192_float64=(scalar/192).tolist()))
    need(pos==len(row['modules']) and exact(load(row['actions']),mean) and exact(load(row['mean']),mean) and exact(load(row['std']),std), 'returned actual mean/std, not winner')
    bounded_reduction(np.asarray(row['elite_average_cost']),v['topk_vals'].astype(np.float64).mean(axis=1),ELITE)
    return dict(root=row['root'],iterations=STEPS,candidate_count=STEPS*SAMPLES,scalars=scores,
        returned_mean_evaluated=False,raw_actions=None,physical_release=False)


def consume(output,h):
    d=own(output);r=read(d/'report.json',h);m=verify(r['prepared'],r['prepare_sha256'])
    need(r['schema']==NAME+'.report' and r['protocol']==protocol() and r['status']=='all_known2_native_normalized_CEM_recorded' and
        r['errors']==[] and r['source_verified'] is True and r['model_before']==r['model_after']==m['expected_model_hash'] and
        r['physical_release'] is r['scientific_admission'] is False and len(r['rows'])==2,'complete native known2 job, no partial release')
    need(r['inherited']==m['inherited'] and r['cuda_peak_allocated']<=3*1024**3 and r['cuda_peak_reserved']<=4*1024**3 and
        r['numerics']==dict(requested_SDP='MATH',kernel_certificate=False,observed_inference=True,observed_TF32=False,observed_autocast=False),'inherited costs/observed guards, not actual kernel certification')
    inv=r['artifacts'];need(set(inv)|{'report.json'}=={p.name for p in d.iterdir()},'whole artifact census')
    need(sum(v['file_bytes'] for v in inv.values())+(d/'report.json').stat().st_size<=CAP and r['elapsed_seconds']<WALL,'saved output/time cap')
    for f,v in inv.items():need(Path(f).name==f and not (d/f).is_symlink() and (d/f).stat().st_size==v['file_bytes'] and sha(d/f)==v['sha256'],'every exact file SHA/bytes')
    events=[];tail=None
    with (d/'journal.jsonl').open('rb') as f:
        for line in f:
            need(len(line)<=1024**2,'bounded journal event');e=json.loads(line);h0=e.pop('event_sha256')
            fields={'sequence','previous_sha256','kind'}
            fields.update({'started':{'prepare_sha256'},'finished':{'status'},'root_sealed':{'root'},
                'cost_intent':{'root','iteration'},'cost_return':{'root','iteration'},'cem_update':{'root','iteration'},
                'module_intent':{'root','index','kind_name'},'module_return':{'root','index','kind_name'}}.get(e['kind'],{'unknown_event'}))
            need(set(e)==fields,'exact native journal fields')
            need(e['sequence']==len(events) and e['previous_sha256']==tail and hashlib.sha256(raw(e)).hexdigest()==h0,'native durable chain')
            e['event_sha256']=h0;need(raw(e)==line,'canonical exactly one newline');tail=h0;events.append(e)
    expected=[('started',None,None,None)]
    for root in (0,1):
        for j in range(STEPS):
            expected.append(('cost_intent',root,j,None))
            for k,kind in enumerate(('action_encoder','predictor','pred_proj','predictor','pred_proj')):
                expected.extend([(event,root,None,(5*j+k,kind)) for event in ('module_intent','module_return')])
            expected.extend([('cost_return',root,j,None),('cem_update',root,j,None)])
        expected.append(('root_sealed',root,None,None))
    expected.append(('finished',None,None,None))
    actual=[(e['kind'],e.get('root'),e.get('iteration'),(e['index'],e['kind_name']) if 'index' in e else None) for e in events]
    need(actual==expected and tail==r['journal_tail'] and events[0]['prepare_sha256']==r['prepare_sha256'] and events[-1]['status']==r['status'],'entire native event order/denominator')
    used=set();load=lambda rec:(used.add(rec['file']),load_array(d,rec,inv))[1]
    rows=[validate_row(s,row,load) for s,row in zip(m['slots'],r['rows'],strict=True)]
    need(used==set(inv)-{'journal.jsonl'},'all actual arrays accounted')
    return dict(schema=NAME+'.consume',status='all_known2_native_normalized_CEM_saved_verified',report_sha256=h,rows=rows,
        per_root_resources=protocol()['per_root'],total_resources={k:2*v for k,v in protocol()['per_root'].items()},
        inherited=r['inherited'],new_encoder=0,new_AD=0,new_physics=0,
        published_reported_values=[],local_performance_claim=False,physical_release=False,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--output',required=True)
    for cmd in ('verify','run'):
        a=s.add_parser(cmd);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd=='run':a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=s.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha',required=True);a.add_argument('--receipt',required=True)
    q=p.parse_args()
    if q.command=='prepare':v=prepare(q.output)
    elif q.command=='verify':verify(q.prepared,q.prepare_sha);v=dict(status='native_CEM_prepare_verified')
    elif q.command=='run':v=run(q.prepared,q.prepare_sha,q.output,q.authorization)
    else:
        v=consume(q.output,q.report_sha);b=raw(v);need(len(b)<=RESERVE,'bounded receipt');p=own(q.receipt);need(shutil.disk_usage(p.parent).free>=FLOOR+len(b),'receipt floor');durable(p,b);v=dict(status=v['status'],path=str(p),sha256=sha(p))
    print(raw(v).decode(),end='')
    if v.get('status')=='partial_or_failed':raise SystemExit(2)


if __name__=='__main__':main()
