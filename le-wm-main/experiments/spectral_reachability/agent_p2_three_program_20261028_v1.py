"""Three imagined-context programs versus H3 joint; MAIN CUDA only, no admission."""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import agent_p2_contextual_composition_runtime_20261028_v1 as parent

NAME='agent_p2_three_program_20261028_v1'
AUTH='MAIN_THREE_PROGRAM_KNOWN2_HOST_CUDA_ONLY'
ROOT,assets,loader=parent.ROOT,parent.assets,parent.loader
compiler,kernel=assets.compiler,assets.kernel
require,sha=parent.require,parent.sha
ARMS=('joint_H3_no_composition','three_program_imagined_recompile')
ALPHAS=(1.,.5,.25,.125)
CAP_LOGIT=float(.75*np.sqrt(15))
CAP,FINAL,FLOOR,WALL,PREP_CAP=256*1024**2,16*1024**2,200*1024**2,1800,2*1024**2
FORWARDS,JACS=1068,85


def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def digest(x):return hashlib.sha256(raw(x)).hexdigest()
def same(a,b):return raw(a)==raw(b)
def need_equal(a,b,message):require(same(a,b),message)


def contract():
    return dict(schema=NAME+'.contract',roots=[0,1],condition='off',known_roots=2,independent_roots=False,
        arms=list(ARMS),candidate_denominator_per_arm=42,candidate_denominator=168,plan_denominator=4,
        horizon=3,raw_steps=15,steps_per_program=5,context='original root10 then own full imagined context at15/20',
        Q='original identity192 mean squared terminal cost',goal='original off task, unchanged',
        root_bank='zero,DLS; radius low/high then original modes0..9 then signs -/+; 42, including zero modes',
        extended_domain='joint30 logit L2 .75sqrt(15), NOT old joint20 .75sqrt(10) or independent packet balls',
        logit_cap=CAP_LOGIT,radii=[float(.25*np.sqrt(15)),CAP_LOGIT],damping=compiler.DAMPING,
        transport='NumPy tanh(float64 logits).astype(float32), never inverse/clip/repair raw',
        seed_rounding='new bank vectors inward-rounded at generation only, predeclared; retain exact legal incumbent thereafter',
        conditional='fixed bank prefix5; raw-J at original remaining H2 incumbent at own cp5; then H1 at own cp10',
        proposal='raw J times continuous float64 tanh chain; toward half remaining-ball negative gradient; zero direction if slope>=0',
        alphas=list(ALPHAS),local_candidates=5,selection='stable(model terminal Q,index), incumbent0; strict improvement only; all42 terminal scores',
        joint_per_root=dict(shared_predict=6,candidate_predict=126,winner_replay=6,standalone_predict=138),
        composed_per_root=dict(shared_predict=6,candidate_predict=924,winner_replay=6,standalone_predict=936),
        per_root=dict(predict=FORWARDS,action_encoder=FORWARDS,predictor=FORWARDS,pred_proj=FORWARDS,
            jacobian=JACS,mathematical_VJP=JACS*192,root_SVD=2,root_ridge=1,new_suffix_SVD=0),
        total=dict(predict=2*FORWARDS,jacobian=2*JACS,mathematical_VJP=2*JACS*192),
        local_predict=dict(H2=14,H1=7),local_AD=dict(H2=1,H1=1),
        replay='each winner fresh full15 plus separately initiated 3 raw5 segments; exact all3 intermediate contexts',
        root_J='zero native float32 logit chain 192x30',local_J='literal rawfloat32 incumbent 192x20/192x10, no quantization derivative claim',
        inherited_encoding=dict(encoder=16,projector=16,shared_original_four_tasks=True),inherited_other_costs=None,
        new_encoder=0,new_projector=0,new_physics=0,training=False,Qfit=False,retries=0,padding=0,
        equal_FLOPs=False,equal_total_compute=False,total_FLOPs=None,engine_internal_backward_calls=None,
        P1_admission=False,P2_mechanism=False,P2_admission=False,fresh=False,physical_release=False,
        CUDA_only=True,MATH=True,TF32=False,autocast=False,allocated_cap_bytes=4*1024**3,reserved_cap_bytes=6*1024**3,
        wall_seconds=WALL,wall_scope='includes initial saved verify and final rehash; boundary checks not preemption',
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR)


def files():
    seen=set(parent.source_files());todo=[Path(__file__).resolve()]
    while todo:
        path=todo.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]+[node.module+'.'+a.name for a in node.names]
            for name in names:
                child=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if name.startswith('experiments.') and child.is_file() and child not in seen:todo.append(child)
    return sorted(seen|{ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')})


def build():
    p=assets.build()  # Complete original overlay consumer, not old 2-edge outcomes.
    require([(s['root_slot'],s['condition']) for s in p['slots']]==assets.ORDER,'original fixed four task authority')
    slots=[deepcopy(p['slots'][i]) for i in (0,2)]
    for s in slots:
        parent.task_bindings(s)
        e=parent.overlay.read(s['task_file'],PREP_CAP)
        s['warm_raw10']=deepcopy(e['task']['root_raw_action_history'])
        s['blind_path']=e['task_contract']['blind_path']
    bindings=dict(p['bindings'])
    # Retain transitive model/blind/input closure verified by the original prepare.
    report=parent.overlay.read(assets.PARENT/'report.json',4*1024**2)
    upstream=parent.overlay.verify(report['prepared'],assets.PARENT_PREP_SHA)
    for k,h in upstream['bindings'].items():
        require(k not in bindings or bindings[k]==h,'shared immutable authority');bindings[k]=h
    for f in files():bindings[str(f.resolve())]=sha(f)
    out=dict(schema=NAME+'.prepare',contract=contract(),slots=slots,policy=p['policy'],
        expected_model_hash=p['expected_model_hash'],bindings_sha256=bindings,
        source_report_sha256=parent.overlay.SOURCE_SHA,inherited_encoding_report_sha256=assets.PARENT_SHA,
        inherited_full_consumer_rebuilt=True,P2_mechanism=False,physical_release=False)
    parent.check_files(out);require(len(raw(out))<=PREP_CAP,'fixed prepare2MiB');return out


def write_new(path,data):
    path=Path(path).absolute();require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink()
        and len(data)<=PREP_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(data),'exclusive own metadata')
    with path.open('xb') as f:require(f.write(data)==len(data),'short metadata');f.flush();os.fsync(f.fileno())
    fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path))


def prepare(output):return write_new(output,raw(build()))
def verify(prepared,expected):
    require(sha(prepared)==expected,'prepare external SHA');p=parent.overlay.read(prepared,PREP_CAP)
    need_equal(p,build(),'entire own prepare rebuilt');return p


def finite(x,shape):return compiler.finite(x,shape)
def chart(u):
    a=np.asarray(u,np.float64);require(a.ndim==1 and a.size in (10,20,30) and np.isfinite(a).all(),'logit vector')
    return np.tanh(a).astype(np.float32).reshape(-1,5,2)
def legal(prefix,tail):
    a=np.concatenate((np.asarray(prefix,np.float64),np.asarray(tail,np.float64)))
    return a.shape==(30,) and np.isfinite(a).all() and np.linalg.norm(a)<=CAP_LOGIT
def radius(prefix):
    return float(np.sqrt(max(0.,CAP_LOGIT**2-float(np.asarray(prefix,np.float64)@np.asarray(prefix,np.float64)))))
def inward(x,cap):
    a=np.asarray(x,np.float64).copy();require(a.ndim==1 and np.isfinite(a).all() and cap>=0,'new bank inward generation')
    if cap==0:return np.zeros_like(a)
    # New seeds only. No tolerance or incumbent inverse/rounding repair.
    for _ in range(16):
        n=float(np.linalg.norm(a))
        if n<=cap:return a
        a*=np.nextafter(cap,0.)/n
    raise ValueError('cannot generate legal seed')


def bank(j,n,goal):
    j=finite(j,(192,30));n=finite(n,(192,));g=finite(goal,(192,))
    _,s,right=np.linalg.svd(j,full_matrices=False)
    for v in right:
        if v[np.argmax(abs(v))]<0:v*=-1
    solve=compiler.bounded_ridge(j,g-n,cap=CAP_LOGIT)
    seeds=[np.zeros(30),np.asarray(solve['params'])];kinds=['zero','root_dls_progress']
    for r in contract()['radii']:
        for mode in range(10):
            for sign in (-1,1):seeds.append(sign*r*right[mode]);kinds.append([r,mode,sign])
    params=[inward(v,CAP_LOGIT).tolist() for v in seeds]
    return dict(params=params,kinds=kinds,unrounded_seeds=[v.tolist() for v in seeds],
        singular_values=s.tolist(),right_vectors=right.tolist(),solve=solve)


def objective(context,goal):
    z=finite(context['latent_history'][-1],(192,));g=finite(goal,(192,));return float(np.mean((z-g)**2))


def proposals(prefix,incumbent,z,j,goal):
    u=np.asarray(incumbent,np.float64);prefix=np.asarray(prefix,np.float64)
    require(u.shape in ((10,),(20,)) and legal(prefix,u),'legal exact incumbent, no repair')
    z=finite(z,(192,));j=finite(j,(192,u.size));g=finite(goal,(192,))
    grad=(2/192)*(j*(1-np.tanh(u)**2)[None,:]).T@(z-g)
    norm=float(np.linalg.norm(grad));r=radius(prefix)
    target=u.copy() if norm==0 else -.5*r*grad/norm
    d=target-u;slope=float(grad@d)
    if slope>=0:d=np.zeros_like(u)
    values=[u]+[u+alpha*d for alpha in ALPHAS]
    require(all(legal(prefix,v) for v in values),'all finite fixed proposals legal; no projection/repair')
    return dict(logits=[v.tolist() for v in values],gradient=grad.tolist(),direction=d.tolist(),slope=slope,remaining_radius=r)


def optimize(context,prefix,u,goal,backend,tag):
    paths=backend.rollout(context,chart(u),tag+'/incumbent')
    j=backend.jacobian(context,np.asarray(u,np.float64),tag+'/J',False)
    need_equal(backend.last_ad_contexts,paths,'raw Jacobian forward exact incumbent complete context')
    candidates=proposals(prefix,u,paths[-1]['latent_history'][-1],j,goal)
    evaluations=[dict(index=0,logits=np.asarray(u).tolist(),contexts=paths,cost=objective(paths[-1],goal))]
    for k,v in enumerate(candidates['logits'][1:],1):
        path=backend.rollout(context,chart(v),tag+'/candidate'+str(k))
        evaluations.append(dict(index=k,logits=v,contexts=path,cost=objective(path[-1],goal)))
    winner=min(evaluations,key=lambda e:(e['cost'],e['index']))
    backend.event(dict(kind='local_selection',tag=tag,selected=winner['index'],costs=[e['cost'] for e in evaluations],
        input_context_sha256=digest(context),goal_sha256=digest(goal),proposals_sha256=digest(candidates)))
    replay=backend.rollout(context,chart(winner['logits']),tag+'/selected_replay')
    need_equal(replay,winner['contexts'],'local incumbent-guarded selected literal replay')
    require(winner['cost']<=evaluations[0]['cost'],'modeled nonworsening, no physical claim')
    return dict(input_context=deepcopy(context),fixed_prefix_logits=np.asarray(prefix).tolist(),incumbent_logits=np.asarray(u).tolist(),
        jacobian=j.tolist(),proposals=candidates,evaluations=evaluations,selected_index=winner['index'],
        selected_logits=winner['logits'],selected_contexts=replay,incumbent_cost=evaluations[0]['cost'],selected_cost=winner['cost'])


def plan(slot,backend):
    """Real three context-dependent programs, no state/labels accepted."""
    packet=slot['packet'];context=deepcopy(packet['root_context']);goal=packet['goal']
    kernel.validate_context(context);require(np.array_equal(slot['q'],np.eye(192)),'original identity Q')
    zero=np.zeros(30,np.float64)
    root_paths=backend.rollout(context,chart(zero),'root_nominal');nominal=root_paths[-1]['latent_history'][-1]
    j=backend.jacobian(context,zero,'root_J',True)
    need_equal(backend.last_ad_contexts,root_paths,'root logit-zero AD forward exact nominal')
    b=bank(j,nominal,goal);backend.event(dict(kind='bank_frozen',bank_sha256=digest(b),candidates=42))
    candidates={a:[] for a in ARMS}
    for i,u in enumerate(b['params']):
        path=backend.rollout(context,chart(u),'joint/'+str(i))
        candidates[ARMS[0]].append(dict(index=i,logits=u,contexts=path,cost=objective(path[-1],goal)))
    for i,v in enumerate(b['params']):
        u=np.asarray(v,np.float64)
        first=backend.rollout(context,chart(u[:10]),'compose/'+str(i)+'/first')[0]
        second=optimize(first,u[:10],u[10:],goal,backend,'compose/'+str(i)+'/edge2')
        need_equal(second['evaluations'][0]['contexts'],candidates[ARMS[0]][i]['contexts'][1:],'joint full15 to same original H2 incumbent')
        tail=np.asarray(second['selected_logits'],np.float64)
        fixed=np.concatenate((u[:10],tail[:10]));mid=second['selected_contexts'][0]
        third=optimize(mid,fixed,tail[10:],goal,backend,'compose/'+str(i)+'/edge3')
        need_equal(third['evaluations'][0]['contexts'][0],second['selected_contexts'][-1],'H2 selected path to own H1 incumbent')
        logits=np.concatenate((fixed,third['selected_logits']))
        require(legal([],logits),'full15 domain')
        path=[first,mid,third['selected_contexts'][0]]
        require(objective(path[-1],goal)<=candidates[ARMS[0]][i]['cost'],'model-only per-candidate full15 safeguard')
        candidates[ARMS[1]].append(dict(index=i,logits=logits.tolist(),contexts=path,cost=objective(path[-1],goal),
            edge2=second,edge3=third,root_seed_sha256=digest(v)))
    winners={}
    for arm in ARMS:
        w=min(candidates[arm],key=lambda e:(e['cost'],e['index']))
        backend.event(dict(kind='selection_frozen',arm=arm,index=w['index'],commands_sha256=digest(chart(w['logits']).tolist()),
            costs=[e['cost'] for e in candidates[arm]],goal_sha256=digest(goal)))
        whole=backend.rollout(context,chart(w['logits']),arm+'/whole15')
        segmented=[];current=deepcopy(context)
        for k,actions in enumerate(chart(w['logits'])):
            current=backend.rollout(current,actions[None],arm+'/segment'+str(k))[0];segmented.append(current)
        need_equal(whole,w['contexts'],'whole15 all boundary replay');need_equal(segmented,whole,'three separately initiated segments replay')
        winners[arm]=dict(index=w['index'],logits=w['logits'],cost=w['cost'],contexts=whole,
            commands=chart(w['logits']).reshape(15,2).tolist(),segmented_contexts=segmented)
    return dict(root_nominal=nominal,root_jacobian=j.tolist(),bank=b,candidates=candidates,winners=winners)


def blob(a):
    a=np.asarray(a);require(a.dtype==np.float32 and np.isfinite(a).all(),'finite literal f32 blob')
    a=np.ascontiguousarray(a,dtype='<f4');data=a.tobytes()
    return dict(shape=list(a.shape),dtype='float32',raw_hex=data.hex(),sha256=hashlib.sha256(data).hexdigest())
def unblob(b):
    require(set(b)=={'shape','dtype','raw_hex','sha256'} and b['dtype']=='float32' and isinstance(b['shape'],list)
        and all(type(x) is int and 0<x<=192 for x in b['shape']) and len(b['shape'])<=3,'exact finite native blob ABI')
    data=bytes.fromhex(b['raw_hex']);require(len(data)<=192*30*4 and hashlib.sha256(data).hexdigest()==b['sha256'],'native blob SHA/size')
    a=np.frombuffer(data,dtype='<f4').reshape(b['shape']);require(np.isfinite(a).all(),'finite native array');return a


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class Store:
    def __init__(self,output,start):
        self.output=Path(output).absolute();self.started=start
        require(self.output.name.startswith(NAME+'_') and not self.output.exists() and not self.output.is_symlink()
            and shutil.disk_usage(self.output.parent).free>=CAP+FLOOR,'exclusive256MiB output plus floor')
        self.output.mkdir();self.previous='0'*64;self.index=0;self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=False
        self._write('events.jsonl.gz',b'')
    def check(self):require(time.monotonic()-self.started<=WALL,'whole1800s boundary wall, no retry')
    def _write(self,name,data,append=False,final=False):
        if not final:self.check()
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat artifact')
        used=sum(f.stat().st_size for f in self.output.iterdir())
        require(used+len(data)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(data),'total cap/final reserve/disk floor')
        with (self.output/name).open('ab' if append else 'xb') as f:require(f.write(data)==len(data),'full output');f.flush();os.fsync(f.fileno())
        fsync_dir(self.output)
    def put(self,name,value,final=False):
        data=raw(value)
        if name.endswith('.gz'):
            c=zlib.compressobj(6,zlib.DEFLATED,31);data=c.compress(data)+c.flush()
        self._write(name,data,final=final)
    def event(self,payload):
        require(not self.closed,'journal cannot reopen');self.check()
        unsigned=dict(index=self.index,previous_sha256=self.previous,payload=deepcopy(payload));h=digest(unsigned)
        data=raw(dict(unsigned,event_sha256=h));out=self.comp.compress(data)+self.comp.flush(zlib.Z_SYNC_FLUSH)
        self._write('events.jsonl.gz',out,append=True);self.previous=h;self.index+=1
    def close(self):
        if not self.closed:
            self._write('events.jsonl.gz',self.comp.flush(zlib.Z_FINISH),append=True,final=True);self.closed=True


def packet_request(index,context,actions,tag):
    require(type(index) is int and index<FORWARDS,'fixed forward cap')
    a=np.asarray(actions);require(a.dtype==np.float32 and a.shape==(5,2) and np.isfinite(a).all() and (abs(a)<1).all(),'literal unmodified raw5')
    kernel.validate_context(context)
    return dict(index=index,tag=tag,input_context_sha256=digest(context),raw_sha256=digest(blob(a)),completed=False)


def module_receipt(index,resource,inputs,output=None):
    e=dict(index=index,resource=resource,inputs=inputs,completed=output is not None)
    if output is not None:e['output']=output
    return e


def validate_packet(f):
    context=f['input'];a=unblob(f['raw']);need_equal(f['request'],packet_request(f['index'],context,a,f['tag']),'exact packet request')
    modules=f['modules'];require(len(modules)==3,'all native modules, no hidden extra')
    ins=[[unblob(b) for b in e['inputs']] for e in modules];outs=[unblob(e['output']) for e in modules]
    tokens=np.concatenate((np.asarray(context['action_history'],np.float32),kernel.normalized_history(a)))
    require([e['resource'] for e in modules]==['action_encoder','predictor','pred_proj'] and
        all(e['index']==3*f['index']+k and e['completed'] is True for k,e in enumerate(modules)),'actual module order/identity')
    require(len(ins[0])==1 and ins[0][0].shape==(1,3,10) and np.array_equal(ins[0][0][0],tokens)
        and len(ins[1])==2 and ins[1][0].shape==(1,3,192) and np.array_equal(ins[1][0][0],context['latent_history'])
        and np.array_equal(ins[1][1],outs[0]) and outs[0].shape==(1,3,192)
        and outs[1].shape==(1,3,192) and len(ins[2])==1 and ins[2][0].shape==(3,192)
        and np.array_equal(ins[2][0],outs[1].reshape(3,192)) and outs[2].shape==(3,192),'literal raw normalization/full native bridge')
    out=deepcopy(context);out.update(latent_history=np.concatenate((np.asarray(context['latent_history'],np.float32)[1:],outs[2][-1:])).tolist(),
        action_history=tokens[1:].tolist())
    need_equal(f['output'],out,'native last row/full latent and action history roll')
    require(f['completed'] is True,'no partial forward');return out


def module_events(f):
    """Exact lifecycle inside one original predict call; no unknown filtering."""
    events=[]
    for k,e in enumerate(f['modules']):
        if k==1:events.append(dict(kind='predict_API_intent',forward=f['index'],completed=False))
        events.extend([dict(kind='module_intent',index=e['index'],resource=e['resource'],completed=False),
            dict(kind='module_inputs',index=e['index'],inputs=e['inputs']),dict(kind='module_return',entry=e)])
        if k==2:events.append(dict(kind='predict_API_return',forward=f['index'],completed=True))
    return events


class SavedBackend:
    """Full algorithm replay using saved native arrays/J only, never model calls."""
    def __init__(self,row,event,load):self.row,self.event,self.load=row,event,load;self.n=self.ad=0
    def rollout(self,c,commands,tag):
        paths=[]
        for k,a in enumerate(commands):
            f=self.row['forwards'][self.n];request=packet_request(self.n,c,np.asarray(a,np.float32),tag+'/'+str(k))
            need_equal(f['request'],request,'every call exact tag/context/raw');need_equal(f['input'],c,'literal context, not digest only')
            require(unblob(f['raw']).tobytes()==np.asarray(a,np.float32).tobytes(),'literal raw bits')
            self.event(dict(kind='forward_intent',request=request))
            self.event(dict(kind='forward_inputs',index=self.n,context=c,raw=f['raw']))
            for e in module_events(f):self.event(e)
            c=validate_packet(f);self.event(dict(kind='forward_return',index=self.n,context=c,completed=True))
            self.n+=1;paths.append(deepcopy(c))
        return paths
    def jacobian(self,c,u,tag,logit):
        entry=self.row['AD'][self.ad];request=dict(index=self.ad,tag=tag,input_context_sha256=digest(c),
            at_logits=np.asarray(u).tolist(),raw=chart(u).tolist(),coordinates='logit_zero' if logit else 'raw_incumbent',
            shape=[192,len(u)],mathematical_VJP=192,vectorize=True,completed=False)
        require(entry['completed'] is True and entry['array']['file']==f'root{self.row["index"]}_J{self.ad:03}.npy','complete exact ordered AD array authority')
        need_equal(entry['request'],request,'actual full AD ABI');self.event(dict(kind='AD_intent',request=request))
        path=self.rollout(c,chart(u),tag+'/forward')
        need_equal(path,entry['contexts'],'AD forward contexts')
        self.last_ad_contexts=deepcopy(path)
        j=self.load(entry['array']);require(j.shape==(192,len(u)) and j.dtype==np.float32,'full original native J')
        self.event(dict(kind='AD_return',index=self.ad,array=entry['array'],contexts=path,completed=True));self.ad+=1
        return j.astype(np.float64)
    def complete(self):require(self.n==len(self.row['forwards'])==FORWARDS and self.ad==len(self.row['AD'])==JACS,'complete fixed forward/AD denominator')


class HostBackend:
    def __init__(self,model,device,t,store,row,event):
        require(device.type=='cuda','CUDA only')
        self.model,self.device,self.t,self.store,self.row,self.event=model,device,t,store,row,event
        self.transition=compiler.make_transition(model,device);self.current=None;self.pending=None;self.api=False
    def check(self):
        self.store.check();self.t.cuda.synchronize(self.device)
        require(self.t.cuda.memory_allocated(self.device)<=contract()['allocated_cap_bytes'] and
            self.t.cuda.memory_reserved(self.device)<=contract()['reserved_cap_bytes'],'actual device memory cap')
    def array(self,x):
        require(x.dtype==self.t.float32 and x.device==self.device,'original CUDA f32 tensor');return blob(x.detach().cpu().numpy())
    @contextmanager
    def observe(self):
        handles=[];original=self.model.predict;had='predict' in self.model.__dict__;local=self.model.__dict__.get('predict')
        def enter(resource):
            def hook(module,args):
                f=self.current;require(f is not None and self.pending is None and resource==('action_encoder','predictor','pred_proj')[len(f['modules'])],'native dispatch order')
                n=3*f['index']+len(f['modules']);self.check();self.event(dict(kind='module_intent',index=n,resource=resource,completed=False))
                inputs=[self.array(x) for x in args];self.event(dict(kind='module_inputs',index=n,inputs=inputs));self.pending=(module,n,resource,inputs)
            return hook
        def returned(module,args,output):
            require(self.pending is not None and self.pending[0] is module,'native module paired return')
            _,n,r,inputs=self.pending;e=module_receipt(n,r,inputs,self.array(output));self.current['modules'].append(e)
            self.event(dict(kind='module_return',entry=e));self.pending=None;self.check()
        def predict(*args,**kwargs):
            require(self.current is not None and not self.api,'no unbound/recursive predict');self.api=True
            self.event(dict(kind='predict_API_intent',forward=self.current['index'],completed=False))
            out=original(*args,**kwargs);self.check();self.event(dict(kind='predict_API_return',forward=self.current['index'],completed=True));self.api=False;return out
        def forbid(module,args):raise ValueError('new vision forbidden')
        try:
            for m,r in ((self.model.action_encoder,'action_encoder'),(self.model.predictor,'predictor'),(self.model.pred_proj,'pred_proj')):
                handles.append(m.register_forward_pre_hook(enter(r)));handles.append(m.register_forward_hook(returned))
            for m in (self.model.encoder,self.model.projector):handles.append(m.register_forward_pre_hook(forbid))
            self.model.predict=predict;yield
            require(self.pending is None and not self.api and self.current is None,'all callbacks returned')
        finally:
            for h in reversed(handles):h.remove()
            if had:self.model.predict=local
            else:del self.model.predict
    def _rollout(self,c,commands,tag,raw_tensor=None):
        t=self.t;state=t.tensor(c['latent_history'],device=self.device,dtype=t.float32)[None]
        history=t.tensor(c['action_history'],device=self.device,dtype=t.float32)[None];paths=[]
        actions=t.tensor(commands,device=self.device,dtype=t.float32) if raw_tensor is None else raw_tensor
        for k in range(len(commands)):
            self.check();a=actions[k];a_np=a.detach().cpu().numpy().copy();n=len(self.row['forwards'])
            request=packet_request(n,c,a_np,tag+'/'+str(k));self.event(dict(kind='forward_intent',request=request))
            f=dict(index=n,tag=tag+'/'+str(k),request=request,input=deepcopy(c),raw=blob(a_np),modules=[],completed=False)
            self.row['forwards'].append(f);self.current=f;self.event(dict(kind='forward_inputs',index=n,context=c,raw=f['raw']))
            state,history=self.transition(state,history,a);self.check()
            c=dict(c,latent_history=state[0].detach().cpu().tolist(),action_history=history[0].detach().cpu().tolist())
            f.update(completed=True,output=deepcopy(c));validate_packet(f)
            self.event(dict(kind='forward_return',index=n,context=c,completed=True));self.current=None;paths.append(deepcopy(c))
        return state,paths
    def rollout(self,c,commands,tag):
        with self.t.no_grad():return self._rollout(c,commands,tag)[1]
    def jacobian(self,c,u,tag,logit):
        self.check();n=len(self.row['AD']);require(n<JACS,'AD cap')
        request=dict(index=n,tag=tag,input_context_sha256=digest(c),at_logits=np.asarray(u).tolist(),raw=chart(u).tolist(),
            coordinates='logit_zero' if logit else 'raw_incumbent',shape=[192,len(u)],mathematical_VJP=192,vectorize=True,completed=False)
        self.event(dict(kind='AD_intent',request=request));entry=dict(request=request,completed=False);self.row['AD'].append(entry)
        t=self.t
        x=t.zeros(len(u),device=self.device,dtype=t.float32) if logit else t.tensor(chart(u).reshape(-1),device=self.device,dtype=t.float32)
        def fn(value):
            a=t.tanh(value).reshape(-1,5,2) if logit else value.reshape(-1,5,2)
            state,path=self._rollout(c,chart(u),tag+'/forward',a);entry['contexts']=path;return state[0,-1]
        with t.enable_grad():j=t.autograd.functional.jacobian(fn,x.requires_grad_(True),vectorize=True)
        self.check();a=j.detach().cpu().numpy().copy();stream=io.BytesIO();np.save(stream,a,allow_pickle=False)
        name=f'root{self.row["index"]}_J{n:03}.npy';self.store._write(name,stream.getvalue())
        desc=dict(file=name,sha256=sha(self.store.output/name),shape=list(a.shape),dtype='float32')
        entry.update(array=desc,completed=True);self.last_ad_contexts=deepcopy(entry['contexts'])
        self.event(dict(kind='AD_return',index=n,array=desc,contexts=entry['contexts'],completed=True));return a.astype(np.float64)


def model_plan(p,slot,result,arm):
    w=result['winners'][arm];context=deepcopy(slot['packet']['root_context']);edges=[]
    for k,c in enumerate(w['contexts']):
        commands=w['commands'][5*k:5*k+5]
        e=dict(index=k,input_context=context,output_context=c,commands=commands,input_context_sha256=digest(context),
            goal_sha256=digest(slot['packet']['goal']),task_id=slot['task_id'],source_report_sha256=p['source_report_sha256'])
        e['witness_sha256']=digest(e);edges.append(e);context=c
    plan=dict(schema=NAME+'.plan',root_slot=slot['root_slot'],condition='off',arm=arm,task_id=slot['task_id'],
        task_file=slot['task_file'],task_sha256=slot['task_sha256'],blind_sha256=slot['blind_sha256'],
        root_context=slot['packet']['root_context'],root_warm_raw10=slot['warm_raw10'],goal=slot['packet']['goal'],
        normalization=dict(mean=kernel.ACTION_MEAN.tolist(),std=kernel.ACTION_STD.tolist()),model_hash=p['expected_model_hash'],
        source_report_sha256=p['source_report_sha256'],blind_path=slot['blind_path'],raw_steps=15,commands=w['commands'],raw_float32_bits=np.asarray(w['commands'],'<f4').tobytes().hex(),
        logits=w['logits'],selected_index=w['index'],predicted_Q=w['cost'],predicted_terminal=w['contexts'][-1]['latent_history'][-1],
        edges=edges,program_count=3 if arm==ARMS[1] else 1,compiled_nodes=[0,5,10] if arm==ARMS[1] else [0],
        physical_runner_required='new full15 native replay/strict, cannot call full10 runner or reuse old outcomes',
        selection_uses_physics=False,P1_admission=False,P2_mechanism=False,physical_release=False)
    plan['selection_digest']=digest(plan);return plan


def run_cuda(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA authorization before any dispatch/I/O')
    start=time.monotonic();p=verify(prepared,expected);store=Store(output,start);rows=[];model=None;before=after=None;errors=[];host=None;source_ok=False
    summary=[dict(root_slot=s['root_slot'],condition='off',status='not_started',candidate_denominator=[42,42],plan_denominator=2) for s in p['slots']]
    try:
        store.check();store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,before=loader.loaded_model(p,torch,device,store,expected)
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
            for slot,ss in zip(p['slots'],summary,strict=True):
                ss['status']='attempted';row=dict(index=slot['root_slot'],forwards=[],AD=[],status='attempted');rows.append(row)
                store.event(dict(kind='root_started',root_slot=slot['root_slot'],task_id=slot['task_id']))
                emit=lambda e:store.event(dict(root_slot=slot['root_slot'],event=e))
                backend=HostBackend(model,device,torch,store,row,emit)
                with backend.observe():row['result']=plan(slot,backend)
                require(len(row['forwards'])==FORWARDS and len(row['AD'])==JACS,'all candidates/actual calls exact')
                row.update(status='sealed',plans=[model_plan(p,slot,row['result'],a) for a in ARMS])
                file=f'root{slot["root_slot"]}.json.gz';store.put(file,row)
                ss.update(status='sealed',file=file,predict=FORWARDS,jacobian=JACS)
                store.event(dict(kind='root_sealed',root_slot=slot['root_slot'],selections=[x['selection_digest'] for x in row['plans']]))
    except BaseException as exc:errors.append(loader.enc.error('execution',exc))
    finally:
        for ss in summary:
            if ss['status']=='attempted':ss['status']='failed_partial'
        if model is not None:
            try:loader.enc.frozen(model);after=loader.enc.tensor_hash(model)
            except BaseException as exc:errors.append(loader.enc.error('final_model',exc))
        try:parent.check_files(p);store.check();source_ok=True
        except BaseException as exc:errors.append(loader.enc.error('final_source_wall',exc))
        for r in rows:
            if r['status']!='sealed':
                try:store.put(f'partial_root{r["index"]}.json.gz',r,final=True)
                except BaseException as exc:errors.append(loader.enc.error('partial_persistence',exc))
        try:store.close()
        except BaseException as exc:errors.append(loader.enc.error('journal_close',exc))
        good=not errors and source_ok and before==after==p['expected_model_hash'] and all(s['status']=='sealed' for s in summary)
        report=dict(schema=NAME+'.report',status='all4_full15_plans_sealed_not_admitted' if good else 'failed_partial',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=contract(),rows=summary,errors=errors,
            first_error=errors[0] if errors else None,model_before=before,model_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,host=host,wall_seconds=time.monotonic()-start,P1_admission=False,P2_mechanism=False,physical_release=False,
            actual_resources=dict(predict_attempted=sum(len(r['forwards']) for r in rows),predict_completed=sum(f['completed'] for r in rows for f in r['forwards']),
                jacobian_attempted=sum(len(r['AD']) for r in rows),jacobian_completed=sum(e['completed'] for r in rows for e in r['AD'])),
            artifact_sha256={f.name:sha(f) for f in store.output.iterdir()})
        store.put('report.json',report,final=True)
    require(good,'failed partials retained; no refund/retry '+str(errors))
    return dict(report=str(store.output/'report.json'),sha256=sha(store.output/'report.json'),status=report['status'])


def journal(output):
    plain,eof=parent.overlay.blind.inflate((Path(output)/'events.jsonl.gz').read_bytes(),512*1024**2)
    require(eof and plain.endswith(b'\n'),'complete bounded gzip journal');previous='0'*64;out=[]
    for i,line in enumerate(plain.splitlines()):
        require(line,'exact single LF record ABI, no empty filtering');e=json.loads(line)
        require(set(e)=={'index','previous_sha256','payload','event_sha256'} and type(e['index']) is int and e['index']==i
            and e['previous_sha256']==previous,'entire journal ordinal/chain ABI')
        unsigned={k:v for k,v in e.items() if k!='event_sha256'};require(digest(unsigned)==e['event_sha256'] and raw(e)==line+b'\n','exact canonical event bytes/hash')
        previous=e['event_sha256'];out.append(e['payload'])
    return out


def consume(output,report_sha):
    output=Path(output).absolute();require(sha(output/'report.json')==report_sha,'external report SHA')
    r=parent.overlay.read(output/'report.json',PREP_CAP);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='all4_full15_plans_sealed_not_admitted' and r['errors']==[] and r['first_error'] is None
        and r['model_unchanged'] is r['source_verified'] is True and r['model_before']==r['model_after']==p['expected_model_hash']
        and r['P1_admission'] is r['P2_mechanism'] is r['physical_release'] is False,'complete engineering only')
    need_equal(r['contract'],contract(),'original fixed contract');need_equal(r['actual_resources'],
        dict(predict_attempted=2*FORWARDS,predict_completed=2*FORWARDS,jacobian_attempted=2*JACS,jacobian_completed=2*JACS),'actual full paid calls')
    names={'binding.json','loader.json','events.jsonl.gz','root0.json.gz','root1.json.gz'}|{f'root{i}_J{k:03}.npy' for i in (0,1) for k in range(JACS)}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'complete bounded all files, no unknown artifacts')
    for n,h in r['artifact_sha256'].items():require(not (output/n).is_symlink() and sha(output/n)==h,'artifact exact SHA')
    need_equal(parent.overlay.read(output/'binding.json'),dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'actual prepare binding')
    load=parent.overlay.read(output/'loader.json');loader.validate_loader(load,p,r['prepare_sha256']);events=journal(output);cursor=0
    def accept(e):
        nonlocal cursor
        require(cursor<len(events),'missing actual event');need_equal(events[cursor],e,'full lifecycle event chronology/unknown event');cursor+=1
    for f in load['files']:
        accept(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']));accept(dict(kind='consumed_buffer_frozen',receipt=f))
    def array(d):
        require(set(d)=={'file','sha256','shape','dtype'} and d['file'] in names and sha(output/d['file'])==d['sha256'],'saved AD descriptor')
        data=(output/d['file']).read_bytes();stream=io.BytesIO(data);a=np.load(stream,allow_pickle=False)
        require(stream.tell()==len(data) and d['dtype']=='float32' and a.dtype==np.float32 and list(a.shape)==d['shape'] and np.isfinite(a).all(),'literal full J array');return a
    require(len(r['rows'])==2,'all two roots/four plans retained');plans=[]
    for slot,ss in zip(p['slots'],r['rows'],strict=True):
        require(ss['root_slot']==slot['root_slot'] and ss['condition']=='off' and ss['status']=='sealed' and ss['predict']==FORWARDS and ss['jacobian']==JACS
            and ss['candidate_denominator']==[42,42] and ss['plan_denominator']==2 and ss['file']==f'root{slot["root_slot"]}.json.gz','full denominator/root identity')
        row=parent.overlay.read(output/ss['file'],CAP)
        require(row['index']==slot['root_slot'] and row['status']=='sealed' and all(e['completed'] is True for e in row['AD']),'all complete native records')
        accept(dict(kind='root_started',root_slot=slot['root_slot'],task_id=slot['task_id']))
        backend=SavedBackend(row,lambda e:accept(dict(root_slot=slot['root_slot'],event=e)),array)
        rebuilt=plan(slot,backend);backend.complete();need_equal(row['result'],rebuilt,'entire candidate bank/guards/full finite terminal scores/winner replay')
        own=[model_plan(p,slot,rebuilt,a) for a in ARMS];need_equal(row['plans'],own,'full15 action/goal/context/source/edge witnesses')
        accept(dict(kind='root_sealed',root_slot=slot['root_slot'],selections=[x['selection_digest'] for x in own]))
        plans.extend(dict(x,planner_report_sha256=report_sha,prepare_sha256=r['prepare_sha256']) for x in own)
    require(cursor==len(events),'no extra event after terminal seal');parent.check_files(p);require(sha(output/'report.json')==report_sha,'final report unchanged')
    return dict(schema=NAME+'.consume',status='full15_engineering_verified_science_blocked',planner_report_sha256=report_sha,
        plans=plans,errors=[],full_upstream_rebuilt_here=True,new_model_calls=0,new_physics_calls=0,P1_admission=False,P2_mechanism=False,physical_release=False)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True)
    for mode in ('verify','run_cuda'):
        q=sub.add_parser(mode);q.add_argument('--prepared',required=True);q.add_argument('--prepare-sha',required=True)
        if mode=='run_cuda':q.add_argument('--output',required=True);q.add_argument('--authorization',required=True)
    q=sub.add_parser('consume');q.add_argument('--output',required=True);q.add_argument('--report-sha',required=True);q.add_argument('--receipt-output')
    a=parser.parse_args()
    if a.cmd=='prepare':value=prepare(a.output)
    elif a.cmd=='verify':p=verify(a.prepared,a.prepare_sha);value=dict(status='verified_no_dispatch',contract=p['contract'])
    elif a.cmd=='run_cuda':value=run_cuda(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:
        value=consume(a.output,a.report_sha)
        if a.receipt_output:value=write_new(a.receipt_output,raw(value))
        else:value={k:v for k,v in value.items() if k!='plans'}|dict(plans_count=len(value['plans']))
    print(raw(value).decode(),end='')


if __name__=='__main__':main()
