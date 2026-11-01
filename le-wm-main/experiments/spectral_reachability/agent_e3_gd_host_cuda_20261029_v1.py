"""E3 fixed16 control5 HOSTCUDA numerical wave; no invented feedback/physics.

Independent native scalar-AD implementation; frozen preparation gate untouched.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from experiments.spectral_reachability import agent_e3_gd_numerical_preparation_20261029_v1 as numerical
from experiments.spectral_reachability import agent_p2_two_program_postcontext_20261029_v1 as source
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_20261029_v2 as primitives

NAME='agent_e3_gd_host_cuda_20261029_v1'
AUTH='MAIN_E3_KNOWN16_CONTROL5_GD_HOSTCUDA_ONLY'
ROOT=numerical.ROOT
CAP,RESERVE,FLOOR,META,WALL=512*1024**2,64*1024**2,200*1024**2,16*1024**2,1800
PREP_SHA='9e339db4be494428bccd5ba1e5562e31870d947a9d37be917068779f6736b261'
SHARED=dict(path=str(ROOT/'results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json'),
    sha256='7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a')
need,raw,digest=numerical.need,numerical.raw,numerical.digest
sha=numerical.resolver.sha
FLAGS=dict(scientific_admission=False,P1_admission=False,P2_admission=False,P3_admission=False,
    E3_demonstrated=False,freshness=False,Full=False,equal_FLOPs=False,physical_release=False)


def same(a,b,message):need(raw(a)==raw(b),message)


def exact_f32(a,b):
    x=np.asarray(a,dtype='<f4');y=np.asarray(b,dtype='<f4')
    return x.shape==y.shape and x.tobytes()==y.tobytes()


def bounded(path,limit):
    p=numerical.resolver.absolute(str(path))
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,'bounded regular file')
    with p.open('rb') as f:b=f.read(limit+1)
    need(len(b)<=limit,'bounded read raced');return b


def read(path,expected,limit=META):
    b=bounded(path,limit);need(hashlib.sha256(b).hexdigest()==expected,'saved SHA')
    return numerical.resolver.cp5.decode(b)


def budget(K):
    n=204*K;arrays=32*n+384;events=16*n+768
    return dict(packet_cap=n,array_count_cap=arrays,event_count_cap=events,
        array_each_bytes=7168,array_bytes=arrays*7168,event_each_bytes=16385,
        event_bytes=events*16385,metadata_bytes=META,reserve_bytes=RESERVE,
        total_bytes=arrays*7168+events*16385+META+RESERVE,cap_bytes=CAP,
        compression='none',flush_overhead_bytes=0,
        scope='3 numerical nodes per cell envelope; current control5 wave executes ONE node',
        encoder_new=0,physics_new=0)


def assemble_preparation(bodies,pp,bindings):
    need(type(bodies) is list and len(bodies)==16,'complete fixed16 body roster')
    for body,slot in zip(bodies,numerical.resolver.roster()):
        numerical.validate_body(body)
        need(body['root_sha256']==numerical.resolver.KNOWN_ROOTS[slot['root']]
            and body['horizon_packets']==slot['horizon_packets'] and body['domain']['name']==slot['domain'],
            'fixed body roster root/horizon/domain')
    need(pp['expected_model_hash']==numerical.resolver.MODEL_SHA,'original frozen model')
    return dict(schema=NAME+'.prepare',bodies=deepcopy(bodies),roster=numerical.resolver.roster(),policy=pp['policy'],
        expected_model_hash=pp['expected_model_hash'],bindings_sha256=deepcopy(bindings),
        actual_resolver=dict(path=str(numerical.ACTUAL_PREPARE),sha256=numerical.ACTUAL_PREPARE_SHA),
        shared_prepare=SHARED,budgets={str(K):budget(K) for K in (1,3)},fixed_cells=16,
        available_boundary=5,unavailable_feedback_boundaries=[10,15],
        current_wave_counts=dict(predict=2176,scalar_reverse=512),
        full_closed_loop_counts=dict(predict=6528,scalar_reverse=1536),
        full_upstream_rebuilt_here=False,closed_loop_complete=False,**FLAGS)


def build():
    need(sha(numerical.__file__)==PREP_SHA,'frozen numerical preparation')
    bodies=numerical.load_planner_inputs()
    pp=read(SHARED['path'],SHARED['sha256'])
    bindings={p:h for p,h in pp['sources'].items() if Path(p).suffix=='.py'}
    for f in set(source.files())|set(primitives.files()):source.merge(bindings,{str(f.resolve()):sha(f)})
    source.merge(bindings,source.installed_sources())
    for f in (Path(__file__).resolve(),Path(numerical.__file__).resolve(),Path(numerical.resolver.__file__).resolve(),
        Path(numerical.resources.__file__).resolve(),Path(numerical.resolver.cp5.__file__).resolve(),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):source.merge(bindings,{str(f):sha(f)})
    for name,expected in (('config.json','2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09'),
        ('weights.pt','48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607')):
        bindings[str(Path(pp['policy'])/name)]=expected
    source.rehash(bindings)
    return assemble_preparation(bodies,pp,bindings)


def write_json(path,value):
    p=numerical.resolver.absolute(str(path));b=raw(value);need(len(b)<=META,'metadata cap')
    with p.open('xb') as f:need(f.write(b)==len(b),'full write');f.flush();os.fsync(f.fileno())
    primitives.native.fsync_dir(p.parent);return sha(p)


def verify(path,expected,*,fixture_builder=None):
    # fixture_builder is a test seam, never exposed via the production CLI.
    p=read(path,expected);same(p,build() if fixture_builder is None else fixture_builder(),
        'whole own prepare reconstruction');return p


class Store:
    def __init__(self,path,started,K,clock=time.monotonic):
        self.output=numerical.resolver.absolute(str(path));self.started=started;self.clock=clock
        self.bounds=budget(K);self.used=0;self.meta=0;self.number=0;self.events=[];self.files={}
        need(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=CAP+FLOOR,'exclusive output/floor')
        self.output.mkdir();primitives.native.fsync_dir(self.output.parent)
    def check(self):
        need(0<=self.clock()-self.started<=WALL,'whole inclusive wall')
        need(self.used<=CAP-RESERVE,'working cap leaves reserve')
    def write(self,name,data,final=False,append=False):
        if not final:self.check()
        need(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat file')
        p=self.output/name;need(not p.is_symlink(),'no symlink')
        need(self.used+len(data)<=CAP-(0 if final else RESERVE)
            and shutil.disk_usage(self.output).free>=FLOOR+len(data),'write-before output/floor')
        with p.open('ab' if append else 'xb') as f:need(f.write(data)==len(data),'full write');f.flush();os.fsync(f.fileno())
        self.used+=len(data);primitives.native.fsync_dir(self.output)
    def put(self,name,value,final=False):
        b=raw(value);need(self.meta+len(b)<=META,'aggregate metadata including partial/report/completion')
        self.write(name,b,final=final);self.meta+=len(b);return sha(self.output/name)
    def array(self,value):
        a=np.asarray(value)
        need(a.dtype==np.float32 and a.size<=768 and np.isfinite(a).all(),'native f32 array ABI')
        s=io.BytesIO();np.save(s,a,allow_pickle=False);b=s.getvalue()
        need(len(b)<=4096+a.nbytes and self.number<self.bounds['array_count_cap'],'array/header/count cap')
        name=f'a{self.number:06}.npy';self.write(name,b);self.number+=1
        return dict(file=name,sha256=hashlib.sha256(b).hexdigest(),shape=list(a.shape),dtype='<f4')
    def event(self,e):
        unsigned=dict(index=len(self.events),previous=self.events[-1]['sha256'] if self.events else None,payload=deepcopy(e))
        value=dict(unsigned,sha256=digest(unsigned));b=raw(value)
        need(len(b)<=16384 and len(self.events)<self.bounds['event_count_cap'],'event payload/count cap')
        self.write('events.jsonl',b+b'\n',append=True);self.events.append(value)
    def inventory(self):return {f.name:sha(f) for f in self.output.iterdir()}


def load_array(output,d):
    need(type(d) is dict and set(d)=={'file','sha256','shape','dtype'} and d['dtype']=='<f4'
        and type(d['shape']) is list and all(type(n) is int and n>=0 for n in d['shape'])
        and Path(d['file']).name==d['file'],'exact array descriptor')
    b=bounded(Path(output)/d['file'],7168);need(hashlib.sha256(b).hexdigest()==d['sha256'],'array SHA')
    stream=io.BytesIO(b);a=np.load(stream,allow_pickle=False)
    need(stream.tell()==len(b),'no NPY trailing bytes')
    need(a.dtype==np.float32 and list(a.shape)==d['shape'] and a.size<=768 and np.isfinite(a).all(),'saved array ABI')
    return a


def journal(output,K):
    maximum=budget(K)['event_bytes'];data=bounded(Path(output)/'events.jsonl',maximum)
    need(data.endswith(b'\n') and b'\n\n' not in data,'exact single LF records')
    result=[];previous=None
    for i,line in enumerate(data.splitlines()):
        need(len(line)<=16384 and i<budget(K)['event_count_cap'],'bounded event record')
        e=numerical.resolver.cp5.decode(line)
        need(set(e)=={'index','previous','payload','sha256'} and type(e['index']) is int and e['index']==i
            and e['previous']==previous and digest({k:v for k,v in e.items() if k!='sha256'})==e['sha256'],'whole event chain')
        previous=e['sha256'];result.append(e['payload'])
    return result


def native_events(f):
    """Deterministic complete original bridge lifecycle, including predict nesting."""
    es=[dict(kind='packet_start',index=f['index'],tag=f['tag'],context_sha256=f['context_sha256'],arrays=f['inputs'])]
    for i,m in enumerate(f['modules']):
        if i==1:es.append(dict(kind='predict_start',index=f['index']))
        es.extend([dict(kind='module_start',packet=f['index'],resource=m['resource']),
            dict(kind='module_inputs',packet=f['index'],resource=m['resource'],inputs=m['inputs']),
            dict(kind='module_end',packet=f['index'],resource=m['resource'],output=m['output'])])
        if i==2:es.append(dict(kind='predict_end',index=f['index'],output=f['predicted']))
    es.append(dict(kind='packet_end',index=f['index'],arrays=f['boundary']))
    return es


def validate_packet(f,c,a,body,output):
    need(set(f)=={'index','tag','context_sha256','inputs','modules','predicted','boundary','completed'}
        and f['completed'] is True and f['context_sha256']==digest(c),'complete packet identity')
    get=lambda d:load_array(output,d)
    inputs=f['inputs'];need(set(inputs)=={'latent','history','raw','mean','std','token','outgoing'},'all packet inputs')
    z,h,ra,mean,std,token,outgoing=[get(inputs[k]) for k in ('latent','history','raw','mean','std','token','outgoing')]
    expected_mean=np.asarray(body['action_mean'],np.float32);expected_std=np.asarray(body['action_std'],np.float32)
    need(z.shape==(1,3,192) and h.shape==(1,2,10) and ra.shape==(5,2)
        and exact_f32(z[0],c['latent_history']) and exact_f32(h[0],c['action_history'])
        and ra.tobytes()==a.tobytes() and exact_f32(mean,expected_mean) and exact_f32(std,expected_std),
        'literal current context/raw/normalizer')
    need(token.shape==(10,) and exact_f32(token,((ra-mean)/std).reshape(10))
        and outgoing.shape==(1,3,10) and exact_f32(outgoing,np.concatenate([h,token[None,None]],axis=1)),
        'exact once-normalized history')
    mods=f['modules'];need(len(mods)==3 and [m['resource'] for m in mods]==['action_encoder','predictor','pred_proj'],'native module roster')
    for m in mods:need(set(m)=={'resource','inputs','output'},'exact module ABI')
    ins=[[get(d) for d in m['inputs']] for m in mods];outs=[get(m['output']) for m in mods]
    need([len(x) for x in ins]==[1,2,1] and exact_f32(ins[0][0],outgoing)
        and outs[0].shape==(1,3,192) and exact_f32(ins[1][0],z) and exact_f32(ins[1][1],outs[0])
        and outs[1].shape==(1,3,192) and ins[2][0].shape==(3,192) and exact_f32(ins[2][0],outs[1].reshape(3,192))
        and outs[2].shape==(3,192),'actual full AE/predictor/projection inputs and outputs')
    predicted=get(f['predicted']);bz,bh=[get(f['boundary'][k]) for k in ('latent','history')]
    need(set(f['boundary'])=={'latent','history'} and predicted.shape==(1,3,192)
        and exact_f32(predicted,outs[2].reshape(1,3,192)) and bz.shape==(1,3,192) and bh.shape==(1,2,10)
        and exact_f32(bz,np.concatenate([z[:,1:],predicted[:,-1:]],axis=1)) and exact_f32(bh,outgoing[:,1:]),
        'full output context native rolling')
    return dict(c,latent_history=bz[0].tolist(),action_history=bh[0].tolist())


def scalar_bound(loss,terminal,goal):
    """Independent f64 arithmetic check, not NumPy-vs-CUDA reduction bitparity."""
    need(type(loss) in (float,np.float32) and np.isfinite(loss) and loss>=0,'finite native scalar')
    d=np.asarray(terminal,np.float32)-np.asarray(goal,np.float32)
    squares=(d*d).astype(np.float32);exact=float(sum(float(x) for x in squares)/192)
    u=2.**-24;gamma=192*u/(1-192*u)
    need(abs(float(loss)-exact)<=gamma*exact+192*np.nextafter(np.float32(0),np.float32(1)).item(),'native meanQ arithmetic bound')


def solve(body,backend,event):
    """New host-capable numerical implementation; never changes frozen RAM gate."""
    body=numerical.validate_body(body);K=body['horizon_packets'];domain=body['domain']['name']
    goal=np.asarray(body['goal'],float);candidates=[];gradients=[]
    def evaluate(x,tag):
        a=numerical.resolver.mask_chart(x.tolist(),K,domain)
        replay=tag.startswith('replay')
        if replay:event(dict(kind='replay_start',tag=tag))
        c=backend.forward(body['root_context'],a,tag)
        event(dict(kind='metric_start',tag=tag))
        cost=float(np.mean((np.asarray(c['latent_history'][-1],float)-goal)**2))
        event(dict(kind='metric_end',tag=tag))
        value=dict(logits=x.tolist(),actions=a.tolist(),context=c,cost=cost)
        if replay:event(dict(kind='replay_end',tag=tag))
        event(dict(kind='candidate',tag=tag,value=value));return value
    current=evaluate(np.zeros((K,5,2)),'iterate0');candidates.append(current)
    for i in range(32):
        c,g,loss=backend.gradient(body['root_context'],np.asarray(current['actions'],np.float32),goal,'AD'+str(i))
        same(c,current['context'],'fresh full AD primal exact incumbent')
        need(isinstance(g,np.ndarray) and g.dtype==np.float32 and g.shape==(5*K,2) and np.isfinite(g).all(),'full raw gradient')
        scalar_bound(loss,c['latent_history'][-1],goal)
        chain=1-np.asarray(current['actions'],float)**2
        derivative=numerical.resolver.mask_gradient((g.astype(float)*chain).reshape(K,5,2).tolist(),K,domain)
        gradients.append(dict(raw_gradient=g.tolist(),loss=float(loss),objective_gradient=derivative.tolist()))
        x=numerical.project(np.asarray(current['logits'])-derivative,K,domain)
        current=evaluate(x,'iterate'+str(i+1));candidates.append(current)
    winner=min(range(33),key=lambda i:(candidates[i]['cost'],i))
    event(dict(kind='selection',winner=winner))
    for i in range(3):same(evaluate(np.asarray(candidates[winner]['logits']),'replay'+str(i)),candidates[winner],'independent replay exact')
    return dict(candidates=candidates,gradients=gradients,winner=winner,commands=candidates[winner]['actions'],
        context=candidates[winner]['context'],first5=candidates[winner]['actions'][:5])


class HostBackend:
    kind='HOST_CUDA'
    def __init__(self,model,device,torch,store,body,row):
        need(device.type=='cuda','HOSTCUDA only')
        self.model,self.device,self.t,self.store,self.body,self.row=model,device,torch,store,body,row
        self.current=None;self.pending=None
    def check(self):
        self.t.cuda.synchronize(self.device);self.store.check()
        need(self.t.cuda.max_memory_allocated(self.device)<=3*1024**3
            and self.t.cuda.max_memory_reserved(self.device)<=4*1024**3,'actual allocated/reserved peak guard')
    def capture(self,tensor):
        need(tensor.dtype==self.t.float32 and tensor.device==self.device,'native CUDA f32 only')
        return self.store.array(tensor.detach().cpu().numpy().copy())
    @contextmanager
    def observe(self):
        handles=[]
        def pre(resource):
            def hook(module,args):
                need(self.current is not None and self.pending is None and len(self.current['modules'])<3
                    and resource==('action_encoder','predictor','pred_proj')[len(self.current['modules'])],'native hook chronology')
                self.check();self.store.event(dict(kind='module_start',packet=self.current['index'],resource=resource))
                inputs=[self.capture(a) for a in args]
                self.store.event(dict(kind='module_inputs',packet=self.current['index'],resource=resource,inputs=inputs))
                self.pending=(module,resource,inputs)
            return hook
        def post(module,args,value):
            need(self.pending is not None and self.pending[0] is module,'native paired hook')
            _,resource,inputs=self.pending;d=self.capture(value)
            self.current['modules'].append(dict(resource=resource,inputs=inputs,output=d))
            self.store.event(dict(kind='module_end',packet=self.current['index'],resource=resource,output=d))
            self.pending=None;self.check()
        def forbid(module,args):raise ValueError('vision/encoding forbidden in numerical wave')
        try:
            for module,resource in ((self.model.action_encoder,'action_encoder'),(self.model.predictor,'predictor'),(self.model.pred_proj,'pred_proj')):
                handles.extend([module.register_forward_pre_hook(pre(resource)),module.register_forward_hook(post)])
            for module in (self.model.encoder,self.model.projector):handles.append(module.register_forward_pre_hook(forbid))
            yield
            need(self.current is None and self.pending is None,'all hooks returned')
        finally:
            for handle in reversed(handles):handle.remove()
    def tensor(self,a):return self.t.tensor(a,device=self.device,dtype=self.t.float32)
    def _forward(self,c,actions,tag,leaf=None):
        t=self.t;z=self.tensor(c['latent_history'])[None];h=self.tensor(c['action_history'])[None]
        a=self.tensor(actions) if leaf is None else leaf
        mean,std=[self.tensor(self.body[k]) for k in ('action_mean','action_std')]
        for k in range(len(actions)//5):
            self.check();n=len(self.row['forwards'])
            need(n<68*self.body['horizon_packets'],'node predict admission')
            before_context=deepcopy(c)
            raw5=a[5*k:5*k+5];token=((raw5-mean)/std).reshape(10);outgoing=t.cat([h,token[None,None]],dim=1)
            f=dict(index=n,tag=tag+'/'+str(k),context_sha256=digest(c),inputs={},modules=[],completed=False)
            self.row['forwards'].append(f);self.current=f
            f['inputs']={name:self.capture(value) for name,value in dict(latent=z,history=h,raw=raw5,mean=mean,std=std,token=token,outgoing=outgoing).items()}
            self.store.event(dict(kind='packet_start',index=n,tag=f['tag'],context_sha256=f['context_sha256'],arrays=f['inputs']))
            encoded=self.model.action_encoder(outgoing)
            self.store.event(dict(kind='predict_start',index=n))
            predicted=self.model.predict(z,encoded)
            f['predicted']=self.capture(predicted)
            self.store.event(dict(kind='predict_end',index=n,output=f['predicted']))
            need(tuple(predicted.shape)==(1,3,192),'full3 original predicted ABI')
            z,h=t.cat([z[:,1:],predicted[:,-1:]],dim=1),outgoing[:,1:]
            f['boundary']={name:self.capture(v) for name,v in dict(latent=z,history=h).items()}
            c=dict(c,latent_history=z[0].detach().cpu().tolist(),action_history=h[0].detach().cpu().tolist())
            f['completed']=True;self.store.event(dict(kind='packet_end',index=n,arrays=f['boundary']))
            same(validate_packet(f,before_context,np.asarray(actions[5*k:5*k+5],np.float32),self.body,self.store.output),
                c,'runtime full native IO/context before next dispatch')
            self.current=None;self.check()
        return z,c
    def forward(self,c,actions,tag):
        with self.t.no_grad():return self._forward(c,actions,tag)[1]
    def gradient(self,c,actions,goal,tag):
        t=self.t;self.check();i=len(self.row['AD']);need(i<32,'scalar reverse admission')
        e=dict(index=i,tag=tag,completed=False);self.row['AD'].append(e)
        self.store.event(dict(kind='AD_start',index=i,tag=tag))
        with t.enable_grad():
            # One fresh complete raw leaf; no detach/numpy copy in the LIVE path.
            leaf=self.tensor(actions).requires_grad_(True)
            z,fresh=self._forward(c,actions,tag,leaf)
            loss=((z[0,-1]-self.tensor(goal))**2).mean()
            need(loss.ndim==0 and leaf.is_leaf and leaf.grad is None and loss.requires_grad,'native scalar/connected fresh raw leaf')
            e['loss']=self.capture(loss);e['raw']=self.capture(leaf)
            self.store.event(dict(kind='grad_start',index=i,loss=e['loss'],raw=e['raw'],scalar_outputs=1,shape=list(leaf.shape)))
            callbacks=[]
            def raw_hook(value):
                callbacks.append(1);self.store.event(dict(kind='raw_backward',index=i));return value
            handle=leaf.register_hook(raw_hook)
            try:g=t.autograd.grad(loss,leaf,retain_graph=False,create_graph=False)[0]
            finally:handle.remove()
            need(len(callbacks)==1,'one actual raw backward callback')
            self.check();e['gradient']=self.capture(g);e['completed']=True
            self.store.event(dict(kind='grad_end',index=i,gradient=e['gradient']))
            value=np.float32(loss.detach().cpu().item());gradient=g.detach().cpu().numpy().copy()
        self.store.event(dict(kind='AD_end',index=i));return fresh,gradient,value


class SavedBackend:
    def __init__(self,body,row,output,event):self.body,self.row,self.output,self.event=body,row,Path(output),event;self.n=0;self.ad=0
    def forward(self,c,actions,tag):
        for k in range(len(actions)//5):
            need(self.n<len(self.row['forwards']),'all expected forwards present')
            f=self.row['forwards'][self.n]
            need(type(f['index']) is int and f['index']==self.n and f['tag']==tag+'/'+str(k),'exact forward index/tag')
            c=validate_packet(f,c,actions[5*k:5*k+5],self.body,self.output)
            for event in native_events(f):self.event(event)
            self.n+=1
        return c
    def gradient(self,c,actions,goal,tag):
        need(self.ad<len(self.row['AD']),'all expected AD entries present');e=self.row['AD'][self.ad]
        need(set(e)=={'index','tag','completed','loss','raw','gradient'} and type(e['index']) is int
            and e['index']==self.ad and e['tag']==tag and e['completed'] is True,'complete scalar AD entry')
        self.event(dict(kind='AD_start',index=self.ad,tag=tag));fresh=self.forward(c,actions,tag)
        leaf=load_array(self.output,e['raw']);loss=load_array(self.output,e['loss']);g=load_array(self.output,e['gradient'])
        need(leaf.shape==g.shape==actions.shape and leaf.tobytes()==actions.tobytes() and loss.shape==(),'native scalar transport/full raw gradient')
        scalar_bound(float(loss),fresh['latent_history'][-1],goal)
        self.event(dict(kind='grad_start',index=self.ad,loss=e['loss'],raw=e['raw'],scalar_outputs=1,shape=list(leaf.shape)))
        self.event(dict(kind='raw_backward',index=self.ad));self.event(dict(kind='grad_end',index=self.ad,gradient=e['gradient']))
        self.event(dict(kind='AD_end',index=self.ad));self.ad+=1;return fresh,g,np.float32(loss)
    def complete(self):need(self.n==len(self.row['forwards'])==68*self.body['horizon_packets'] and self.ad==len(self.row['AD'])==32,'complete node denominator')


def axes(events):
    attempted=dict.fromkeys(numerical.resources.P3_AXES,0);completed=dict(attempted)
    def charge(target,key,count=1):target[key]+=count
    for e in events:
        kind=e['kind']
        if kind in ('predict_start','predict_end'):
            out=attempted if kind.endswith('start') else completed
            for name,value in (('predict_API',1),('predict_samples',1),('predict_context_tokens',3)):charge(out,name,value)
        elif kind in ('module_start','module_end'):
            out=attempted if kind.endswith('start') else completed;r=e['resource']
            if r=='pred_proj':charge(out,'pred_proj_calls');charge(out,'pred_proj_rows',3)
            else:
                charge(out,r+'_calls');charge(out,r+'_samples');charge(out,r+'_context_tokens',3)
        elif kind=='grad_start':charge(attempted,'grad_API');charge(attempted,'VJP_directions')
        elif kind=='grad_end':charge(completed,'grad_API');charge(completed,'VJP_directions')
        elif kind=='raw_backward':charge(attempted,'raw_backward_callbacks');charge(completed,'raw_backward_callbacks')
        elif kind in ('metric_start','metric_end','replay_start','replay_end'):
            out=attempted if kind.endswith('start') else completed
            charge(out,'metric_calls' if kind.startswith('metric') else 'replay_calls')
    return dict(attempted=attempted,completed=completed,engine_backward=None,FLOPs=None)


def class_inventory(model,bindings):
    """Preserve each loaded module's name AND class, not only unique source files."""
    out=[]
    for name,module in model.named_modules():
        cls=type(module);path=str(Path(inspect.getfile(cls)).resolve())
        need(path in bindings and sha(path)==bindings[path],'every loaded class source presealed')
        out.append(dict(name=name,class_module=cls.__module__,class_qualname=cls.__qualname__,
            path=path,sha256=bindings[path]))
    need(out and len({x['name'] for x in out})==len(out),'nonempty unique named module inventory')
    return out


def run_cuda(prepared,expected,cell,output,authorization):
    need(authorization==AUTH and type(cell) is int and 0<=cell<16,'explicit MAIN CUDA cell authorization before I/O')
    started=time.monotonic();p=verify(prepared,expected);body=p['bodies'][cell]
    store=Store(output,started,body['horizon_packets']);model=None;before=after=None;errors=[]
    row=dict(forwards=[],AD=[],result=None);classes_before=classes_after=None;prec_before=prec_after=None;host=peak=None;source_ok=False
    try:
        store.check();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,before=primitives.native.loader.loaded_model(p,torch,device,store,expected)
        classes_before=class_inventory(model,p['bindings_sha256'])
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
            prec_before=primitives.precision(torch);same(prec_before,primitives.precision_expected(),'original precision')
            backend=HostBackend(model,device,torch,store,body,row)
            with backend.observe():row['result']=solve(body,backend,store.event)
            backend.check();prec_after=primitives.precision(torch)
        classes_after=class_inventory(model,p['bindings_sha256'])
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc),repr=repr(exc),traceback=traceback.format_exc()))
    finally:
        if model is not None:
            try:
                primitives.native.loader.enc.frozen(model);after=primitives.native.loader.enc.tensor_hash(model)
                peak=dict(allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device))
            except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        try:source.rehash(p['bindings_sha256']);store.check();source_ok=True
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        store.put('row.json',row,final=True)
        report=dict(schema=NAME+'.report',prepared=str(Path(prepared)),prepare_sha256=expected,cell=cell,
            body_sha256=digest(body),boundary=5,full_closed_loop_complete=False,model_before=before,model_after=after,
            classes_before=classes_before,classes_after=classes_after,precision_before=prec_before,precision_after=prec_after,
            host=host,GPU_peak=peak,sources_verified=source_ok,errors=errors,
            axes=axes([e['payload'] for e in store.events]),artifacts=store.inventory(),
            wall_before_report_seconds=time.monotonic()-started,**FLAGS)
        store.put('report.json',report,final=True);report_sha=sha(store.output/'report.json')
        elapsed=time.monotonic()-started
        counts=axes([e['payload'] for e in store.events])
        complete=(not errors and before==after==p['expected_model_hash'] and source_ok and elapsed<=WALL
            and row['result'] is not None and classes_before==classes_after and classes_before is not None
            and prec_before==prec_after==primitives.precision_expected() and peak is not None
            and peak['allocated']<=3*1024**3 and peak['reserved']<=4*1024**3
            and counts['attempted']==counts['completed']==numerical.axes27(body['horizon_packets']))
        completion=dict(schema=NAME+'.completion',report_sha256=report_sha,elapsed_seconds=elapsed,
            boundary='after_report_fsync_hash',complete=complete,cell=cell)
        store.put('completion.json',completion,final=True)
        # Completion fsync/hash also belongs to the inclusive wall boundary.
        sha(store.output/'completion.json')
        if time.monotonic()-started>WALL:
            store.put('publication_overrun.json',dict(reason='completion transport exceeded wall',complete=False),final=True)
            complete=False
    return dict(report_sha256=report_sha,completion_sha256=sha(store.output/'completion.json'),complete=complete)


def consume(output,report_sha,completion_sha):
    output=numerical.resolver.absolute(str(output));r=read(output/'report.json',report_sha);c=read(output/'completion.json',completion_sha)
    need(r['schema']==NAME+'.report' and c['schema']==NAME+'.completion' and c['complete'] is True
        and c['report_sha256']==report_sha and c['boundary']=='after_report_fsync_hash'
        and type(c['elapsed_seconds']) in (int,float) and np.isfinite(c['elapsed_seconds']) and 0<=c['elapsed_seconds']<=WALL
        and type(r['wall_before_report_seconds']) in (int,float) and 0<=r['wall_before_report_seconds']<=c['elapsed_seconds'],'complete bounded publication')
    p=verify(r['prepared'],r['prepare_sha256']);cell=r['cell']
    need(type(cell) is int and 0<=cell<16 and c['cell']==cell,'fixed16 cell')
    body=p['bodies'][cell];K=body['horizon_packets']
    need(r['body_sha256']==digest(body) and r['errors']==[] and r['sources_verified'] is True
        and r['model_before']==r['model_after']==p['expected_model_hash'],'actual model/source/exception guards')
    same(r['precision_before'],primitives.precision_expected(),'actual precision before');same(r['precision_after'],r['precision_before'],'precision invariant')
    need(r['classes_before']==r['classes_after'] and r['classes_before'],'actual loaded classes')
    need(type(r['classes_before']) is list and len({x['name'] for x in r['classes_before']})==len(r['classes_before']),
        'unique actual named module classes')
    for item in r['classes_before']:
        need(set(item)=={'name','class_module','class_qualname','path','sha256'}
            and all(type(item[k]) is str and item[k] for k in ('class_module','class_qualname','path','sha256'))
            and type(item['name']) is str and p['bindings_sha256'].get(item['path'])==item['sha256'],
            'actual named class source presealed')
    peak=r['GPU_peak'];need(set(peak)=={'allocated','reserved'} and all(type(x) is int and x>=0 for x in peak.values())
        and peak['allocated']<=3*1024**3 and peak['reserved']<=4*1024**3,'actual GPU peak caps')
    host=r['host'];need(type(host) is dict and host.get('device')=='cuda:0' and type(host.get('index')) is int
        and host['index']==0 and type(host.get('name')) is str and bool(host['name'])
        and type(host.get('allocation_sum')) in (float,int) and host['allocation_sum']==4
        and type(host.get('visible_nvidia_nodes')) is list and bool(host['visible_nvidia_nodes']), 'actual host CUDA allocation receipt')
    for k in FLAGS:need(r[k] is False,'science gates false')
    names={x.name for x in output.iterdir()};need(names==set(r['artifacts'])|{'report.json','completion.json'},'all-and-only artifact inventory')
    need(sum(x.stat().st_size for x in output.iterdir())<=CAP,'total disk cap')
    for name,h in r['artifacts'].items():need(Path(name).name==name and sha(output/name)==h,'every artifact SHA')
    row=read(output/'row.json',r['artifacts']['row.json']);events=journal(output,K);replayed=[]
    # Loader events are independently validated, never discarded as arbitrary kinds.
    need(len(events)>=4 and [e['kind'] for e in events[:4]]==['load_intent','consumed_buffer_frozen','load_intent','consumed_buffer_frozen'],'exact loader lifecycle')
    loader=read(output/'loader.json',r['artifacts']['loader.json'])
    primitives.native.loader.validate_loader(loader,p,r['prepare_sha256'])
    for i,f in enumerate(loader['files']):
        same(events[2*i],dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']),'loader intent exact')
        same(events[2*i+1],dict(kind='consumed_buffer_frozen',receipt=f),'loader consumed bytes exact')
    backend=SavedBackend(body,row,output,replayed.append)
    result=solve(body,backend,replayed.append);backend.complete()
    same(result,row['result'],'entire numerical result saved reconstruction')
    same(events[4:],replayed,'whole native/AD/selection chronology exact, no splicing')
    same(r['axes'],axes(events),'all observed axes rebuilt')
    need(r['axes']['attempted']==r['axes']['completed']==numerical.axes27(K),'complete actual 27axis census')
    referenced=set()
    def refs(v):
        if type(v) is dict:
            if set(v)=={'file','sha256','shape','dtype'}:load_array(output,v);referenced.add(v['file'])
            else:
                for x in v.values():refs(x)
        elif type(v) is list:
            for x in v:refs(x)
    refs(row);need(referenced=={name for name in names if name.endswith('.npy')},'all-and-only native array inventory')
    need(names==referenced|{'loader.json','row.json','events.jsonl','report.json','completion.json'},'no unexpected non-array artifact')
    source.rehash(p['bindings_sha256']);need(sha(output/'report.json')==report_sha,'report stable after consume')
    return dict(schema=NAME+'.consume',status='control5_cell_native_GD_verified_NOT_admitted',cell=cell,
        report_sha256=report_sha,completion_sha256=completion_sha,axes=r['axes'],plan=result,
        inherited_full_upstream_rebuilt_here=False,closed_loop_complete=False,new_neural_calls=0,new_physics_calls=0,**FLAGS)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume','budget'))
    for key in ('prepared','prepare-sha','output','authorization','report-sha','completion-sha','receipt'):parser.add_argument('--'+key)
    parser.add_argument('--cell',type=int);a=parser.parse_args()
    if a.command=='budget':print(raw({str(K):budget(K) for K in (1,3)}).decode())
    elif a.command=='prepare':print(write_json(a.output,build()))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);print('verified')
    elif a.command=='run':print(raw(run_cuda(a.prepared,a.prepare_sha,a.cell,a.output,a.authorization)).decode())
    else:
        value=consume(a.output,a.report_sha,a.completion_sha)
        if a.receipt:
            receipt=numerical.resolver.absolute(a.receipt)
            need(receipt.parent==ROOT/'results' and receipt.name.startswith(NAME+'_consume_')
                and receipt.parent!=numerical.resolver.absolute(a.output),'new own receipt outside artifact directory')
            print(raw(dict(path=str(receipt),sha256=write_json(receipt,value))).decode())
        else:print(raw(value).decode())


if __name__=='__main__':main()
