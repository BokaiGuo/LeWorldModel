"""Native Hi-LeWM high-target capture seam; production loader NOT certified.

No model loading, tensor imports, simulator, training or network at import/CLI.
The executable callback capture is explicitly RAM-only until a NEW certified
loader integration exists. Saved integrity is not learned-prior admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

TAG='agent_hilewm_high_target_adapter_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
TEXTS=ROOT/'results/p1_hilewm_text_contract_20261027_v1/texts'
SOURCES={
 'policies':('package__code__h_le_wm__planning__policies.py.source.txt','d6e7d476b45e47c5015494775b4d96727b8ebb8c801f719a100ba32878f288d1'),
 'jepa':('package__code__h_le_wm__models__jepa.py.source.txt','bb5338bbb4f9e76b0bd15b94767624168883c994ab4c2ad8209337a3d70022a2')}
MODULES=('macro_to_condition','high_predictor','high_pred_proj')
APIS=('high_solver','get_cost_high','rollout_high','predict_high','quantize_macro_actions_for_planning')+MODULES
AXES=tuple(k+'_'+u for k in APIS for u in ('calls','rows'))
CAP,FLOOR,WALL=96*1024**2,200*1024**2,600
UNKNOWN=('FLOPs','cuda_kernel_calls','peak_gpu_bytes','inherited_encoding_cost','backward_internal_calls')
BLOCKERS=(
 'no certified complete checkpoint loader / trusted deserialization authority in this version',
 'embedded-low tensor/config/preprocessing/normalizer pairing to E1 common coordinates unverified',
 'native solver dependency revision, configured bounds/macro-bank and input provenance need MAIN certification',
 'native single-last-frame high input is NOT the existing three-frame/nonzero-action-history contract')


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def decode(raw):
    def unique(items):
        v={}
        for k,x in items:require(k not in v,'duplicate JSON key');v[k]=x
        return v
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:require(False,'nonfinite JSON'))


def source_binding():
    b={}
    for filename,h in SOURCES.values():
        p=TEXTS/filename;require(sha(p)==h,'frozen native text source');b[str(p)]=h
    for p in (Path(__file__),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')):
        b[str(p)]=sha(p)
    return b


def run_cuda(*args,**kwargs):
    # Intentionally BEFORE any caller callback, I/O, torch/pickle import or GPU
    # allocation. A self-reported hash/"MAIN" bool cannot certify a loader.
    raise ValueError('production failclosed: '+ '; '.join(BLOCKERS))


def _array(value):
    if isinstance(value,np.ndarray):a=value
    else:a=value.detach().cpu().numpy()  # no dtype casting, including bfloat16
    require(a.dtype.kind=='f' and a.dtype.itemsize in (2,4,8)
        and a.ndim<=6 and np.isfinite(a).all(),'finite original float tensor ABI')
    return np.array(a,copy=True,order='C')


def _rows(value):return int(value.shape[0])


class Store:
    """Exclusive flat NPY/JSON durable receipt, bounded including failure footer."""
    def __init__(self,path):
        self.path=Path(path).absolute();self.started=time.monotonic();self.seq=0
        self.previous=None;self.events=[];self.files={};self.arrays=0
        require(not self.path.exists() and not self.path.is_symlink()
            and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'exclusive output/floor')
        self.path.mkdir()

    def used(self):return sum(p.stat().st_size for p in self.path.iterdir())

    def guard(self,n=4096,final=False):
        require(final or time.monotonic()-self.started<WALL,'wall before dispatch/write')
        require(self.used()+n<=CAP-(0 if final else 1024**2)
            and shutil.disk_usage(self.path).free>=FLOOR+n,'output cap/floor')

    def sync(self):
        fd=os.open(self.path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

    def put(self,name,data,final=False):
        require(Path(name).name==name and not (self.path/name).is_symlink(),'flat file')
        self.guard(len(data),final)
        with (self.path/name).open('xb') as f:
            require(f.write(data)==len(data),'short write');f.flush();os.fsync(f.fileno())
        self.sync();self.files[name]=sha(self.path/name)

    def pack(self,value):
        if isinstance(value,(list,tuple)):return [self.pack(x) for x in value]
        if isinstance(value,dict):
            require(all(type(k) is str for k in value),'string tensor-tree keys')
            return {k:self.pack(x) for k,x in value.items()}
        if value is None or type(value) in (str,bool,int,float):canonical(value);return value
        a=_array(value);name=f'array_{self.arrays:05d}.npy';self.arrays+=1
        import io
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False);data=buf.getvalue();self.put(name,data)
        return {'array':name,'sha256':self.files[name],'bytes':len(data),'shape':list(a.shape),'dtype':a.dtype.str,
            'raw_sha256':hashlib.sha256(a.tobytes()).hexdigest()}

    def emit(self,kind,**fields):
        row=dict(index=self.seq,previous=self.previous,kind=kind,**fields)
        row['sha256']=digest(row);data=canonical(row)+b'\n';self.guard(len(data))
        with (self.path/'journal.jsonl').open('ab') as f:
            require(f.write(data)==len(data),'short journal');f.flush();os.fsync(f.fileno())
        self.sync();self.seq+=1;self.previous=row['sha256'];self.events.append(row)


class Meter:
    """Real callback entries/returns in scope; nested axes overlap, never add."""
    def __init__(self,store,limits):
        require(set(limits)==set(AXES) and all(type(v) is int and v>=0 for v in limits.values()),'full predeclared integer caps')
        self.store,self.limits=store,dict(limits);self.stack=[];self.records=[];self.poisoned=False
        self.attempted=dict.fromkeys(AXES,0);self.completed=dict.fromkeys(AXES,0)

    def begin(self,kind,rows,inputs):
        try:
            require(not self.poisoned and kind in APIS and type(rows) is int and rows>0,'known actual call/rows')
            self.store.guard();units={kind+'_calls':1,kind+'_rows':rows}
            require(all(self.attempted[k]+n<=self.limits[k] for k,n in units.items()),'predispatch cap')
            rec=dict(index=len(self.records),parent=self.stack[-1] if self.stack else None,kind=kind,
                units=units,inputs=self.store.pack(inputs),status='pending',output=None)
            self.store.emit('entry',record=rec)
        except BaseException:self.poisoned=True;raise
        for k,n in units.items():self.attempted[k]+=n
        self.records.append(rec);self.stack.append(rec['index']);return rec['index']

    def end(self,i,output):
        require(not self.poisoned and self.stack and self.stack[-1]==i,'paired nested return')
        rec=self.records[i]
        for k,n in rec['units'].items():self.completed[k]+=n
        rec['status']='returned_capture_pending'
        try:
            rec['output']=self.store.pack(output);rec['status']='returned'
            self.store.emit('return',record=rec)
        except BaseException:self.poisoned=True;raise
        self.stack.pop()

    def call(self,kind,rows,inputs,fn):
        i=self.begin(kind,rows,inputs)
        try:out=fn();self.end(i,out);return out
        except BaseException:
            self.poisoned=True  # no retry/refund, pending nested entries retained
            raise


class ModelTap:
    """Rebind native high methods to a facade; never patch model or globals.

    Internal self.rollout_high/predict_high calls pass this facade, while real
    module objects and all other helpers are delegated unchanged. No new math.
    """
    METHODS=('get_cost','get_cost_high','rollout_high','predict_high','quantize_macro_actions_for_planning')
    def __init__(self,model,meter):self.model,self.meter=model,meter

    def __getattr__(self,name):return getattr(self.model,name)

    def native(self,name,*args):
        method=getattr(type(self.model),name)
        return method(self,*args)

    def get_cost(self,info,candidates):
        require({'z_init','z_goal'}<=set(info)<= {'planner_level','z_init','z_goal'},'no low/flat fallback')
        return self.native('get_cost',info,candidates)

    def get_cost_high(self,info,candidates):
        require(len(candidates.shape)==4,'B,S,H,grouped macro')
        return self.meter.call('get_cost_high',int(candidates.shape[0]*candidates.shape[1]),
            dict(info=info,candidates=candidates),lambda:self.native('get_cost_high',info,candidates))

    def rollout_high(self,z,macro):
        require(len(macro.shape) in (3,4),'raw macro sequence rank')
        n=int(macro.shape[0]*(macro.shape[1] if len(macro.shape)==4 else 1))
        return self.meter.call('rollout_high',n,dict(z_init=z,macro=macro),lambda:self.native('rollout_high',z,macro))

    def predict_high(self,z,macro):
        return self.meter.call('predict_high',_rows(z),dict(emb=z,macro=macro),lambda:self.native('predict_high',z,macro))

    def quantize_macro_actions_for_planning(self,macro):
        return self.meter.call('quantize_macro_actions_for_planning',_rows(macro),dict(macro=macro),
            lambda:self.native('quantize_macro_actions_for_planning',macro))


class SolverTap:
    def __init__(self,solver,meter):self.solver,self.meter=solver,meter
    def __call__(self,info,init_action=None):
        return self.meter.call('high_solver',_rows(info['z_init']),dict(info=info,init_action=init_action),
            lambda:self.solver(info,init_action=init_action))


def validate_config(cfg):
    require(set(cfg)=={'horizon','receding_horizon','action_block','warm_start'} and
        all(type(cfg[k]) is int and cfg[k]>0 for k in ('horizon','receding_horizon','action_block'))
        and cfg['receding_horizon']<=cfg['horizon'] and type(cfg['warm_start']) is bool,'native high config')


def _capture(native_plan,model,solver_factory,cfg,z_init,z_goal,next_high_init,store,meter):
    """Concrete native policy/high-model seam. Caller must supply real methods.

    This private tensor-neutral kernel has no loader/auth authority. Its only
    public caller in v1 is capture_fixture; production run_cuda is CLOSED.
    """
    validate_config(cfg)
    require(len(z_init.shape)==len(z_goal.shape)==2 and z_init.shape==z_goal.shape
        and z_init.shape[0]==1 and z_init.shape[1]>0,'single-query native 1,D init/goal')
    tap=ModelTap(model,meter);solver=solver_factory(tap)
    require(solver.model is tap,'solver must use instrumented SAME high model')
    policy=SimpleNamespace(model=tap,high_solver=SolverTap(solver,meter),high_cfg=SimpleNamespace(**cfg),
        _next_high_init=next_high_init,_latent_action_dim=model._infer_latent_action_dim(),
        _z_subgoal=None,_steps_since_high=None)
    handles=[];pending={k:[] for k in MODULES};errors=[]
    try:
        for key in MODULES:
            module=getattr(model,key)
            def pre(mod,args,key=key):
                pending[key].append(meter.begin(key,_rows(args[0]),dict(args=list(args))))
            def post(mod,args,out,key=key):
                require(pending[key],'paired actual module hook');meter.end(pending[key].pop(),out)
            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(post))
        native_plan(policy,z_init=z_init,z_goal=z_goal)
        require(not meter.stack and not meter.poisoned and not any(pending.values()),'all native entries returned')
        return policy
    finally:
        for h in reversed(handles):
            try:h.remove()
            except Exception as exc:errors.append(repr(exc))
        require(not errors,'hook cleanup failures: '+str(errors))


def capture_fixture(output,*,native_plan,model,solver_factory,cfg,z_init,z_goal,next_high_init,limits):
    """NON-neural RAM callbacks only. Never call with a torch/model/env object."""
    require(isinstance(z_init,np.ndarray) or getattr(z_init,'RAM_ONLY',False),'RAM-only tensor fixture')
    require(getattr(model,'RAM_ONLY',False) and getattr(native_plan,'RAM_ONLY',False)
        and getattr(solver_factory,'RAM_ONLY',False),'explicit non-neural callbacks only')
    bindings=source_binding();store=Store(output);meter=Meter(store,limits)
    r=dict(schema=TAG+'.report',status='partial_or_failed',fixture_only=True,production_executed=False,
        scientific_admission=False,learned_prior_established=False,lowlevel_target_delivered=False,
        bindings=bindings,config=dict(cfg),limits=dict(limits),errors=[],output=None,
        unknown_costs=dict.fromkeys(UNKNOWN),new_physics=0,new_encoder=0,new_low_predict=0,new_training=0)
    try:
        inputs=store.pack(dict(z_init=z_init,z_goal=z_goal,next_high_init=next_high_init))
        r['inputs']=inputs;store.emit('started',inputs=inputs,config=cfg,limits=limits,bindings=bindings)
        p=_capture(native_plan,model,solver_factory,cfg,z_init,z_goal,next_high_init,store,meter)
        r['output']=store.pack(dict(target=p._z_subgoal,next_high_init=p._next_high_init,steps_since_high=p._steps_since_high))
        r['status']='RAM_native_high_capture_complete_NOT_neural_evidence'
    except BaseException as exc:r['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        r.update(attempted=meter.attempted,completed=meter.completed,records=meter.records,
            journal_tail=store.previous,wall_seconds=time.monotonic()-store.started)
        try:
            require(all(sha(p)==h for p,h in bindings.items()),'source closure unchanged')
            store.put('report.json',canonical(r),final=True)
        except BaseException:
            # Failed footer is not a successful receipt; existing files survive.
            raise
    return r


def consume(output,report_sha256):
    """Pure saved verifier; no neural recomputation or mock->real upgrade."""
    out=Path(output).absolute();require(not out.is_symlink(),'regular output directory')
    files=list(out.iterdir());require(all(p.is_file() and not p.is_symlink() for p in files)
        and sum(p.stat().st_size for p in files)<=CAP,'bounded flat saved layout')
    require(sha(out/'report.json')==report_sha256,'external report SHA')
    r=decode((out/'report.json').read_bytes())
    require(r['schema']==TAG+'.report' and r['fixture_only'] is True and r['production_executed'] is False
        and r['scientific_admission'] is r['learned_prior_established'] is r['lowlevel_target_delivered'] is False,
        'v1 fixture only; production loader absent')
    require(r['status']=='RAM_native_high_capture_complete_NOT_neural_evidence' and r['errors']==[], 'complete capture required, retain failed files')
    require(r['bindings']==source_binding() and r['unknown_costs']==dict.fromkeys(UNKNOWN)
        and r['new_physics']==r['new_encoder']==r['new_low_predict']==r['new_training']==0,'scope/source integrity')
    require(0<=r['wall_seconds']<WALL and set(r['limits'])==set(AXES),'wall/caps')
    raw=(out/'journal.jsonl').read_bytes();require(raw.endswith(b'\n'),'complete journal')
    events=[decode(x) for x in raw.splitlines()];previous=None
    for i,e in enumerate(events):
        require(e['index']==i and e['previous']==previous and e['sha256']==digest({k:v for k,v in e.items() if k!='sha256'}),'journal chain')
        previous=e['sha256']
    require(previous==r['journal_tail'] and events[0]['kind']=='started'
        and events[0]['inputs']==r['inputs'] and events[0]['config']==r['config']
        and events[0]['limits']==r['limits'] and events[0]['bindings']==r['bindings'],'start/end binding')
    seen=set()
    def unpack(v):
        if isinstance(v,list):return [unpack(x) for x in v]
        if isinstance(v,dict) and 'array' in v:
            require(set(v)=={'array','sha256','bytes','shape','dtype','raw_sha256'},'exact array descriptor')
            name=v['array'];require(Path(name).name==name and name.startswith('array_') and name.endswith('.npy'),'flat NPY')
            p=out/name;require(p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'array SHA/bytes')
            with p.open('rb') as f:a=np.load(f,allow_pickle=False);require(f.read()==b'','NPY exact EOF')
            require(list(a.shape)==v['shape'] and a.dtype.str==v['dtype'] and hashlib.sha256(a.tobytes()).hexdigest()==v['raw_sha256'],'original tensor bytes/shape/dtype')
            _array(a);seen.add(name);return a
        if isinstance(v,dict):return {k:unpack(x) for k,x in v.items()}
        return v
    started=unpack(r['inputs']);result=unpack(r['output']);stack=[];entries={};records=[]
    attempted=dict.fromkeys(AXES,0);completed=dict.fromkeys(AXES,0)
    for e in events[1:]:
        require(e['kind'] in ('entry','return'),'known event kind');rec=e['record'];i=rec['index'];kind=rec['kind']
        require(kind in APIS and set(rec['units'])=={kind+'_calls',kind+'_rows'} and rec['units'][kind+'_calls']==1
            and type(rec['units'][kind+'_rows']) is int and rec['units'][kind+'_rows']>0,'real separated call/row axes')
        unpack(rec['inputs']);unpack(rec['output'])
        if e['kind']=='entry':
            require(i==len(entries) and rec['parent']==(stack[-1] if stack else None) and rec['status']=='pending' and rec['output'] is None,'ordered native entry')
            entries[i]=rec;stack.append(i)
            for k,n in rec['units'].items():attempted[k]+=n
        else:
            require(stack and stack.pop()==i and rec['status']=='returned' and
                dict(rec,status='pending',output=None)==entries[i],'paired native return')
            records.append(rec)
            for k,n in rec['units'].items():completed[k]+=n
    records.sort(key=lambda v:v['index'])
    require(not stack and len(records)==len(entries) and records==r['records'] and attempted==r['attempted']
        and completed==r['completed'] and attempted==completed and all(type(r['limits'][k]) is int
            and 0<=attempted[k]<=r['limits'][k] for k in AXES),'all actual entry-return/caps, no refunded failure')
    calls=[dict(v,inputs=unpack(v['inputs']),output=unpack(v['output'])) for v in records]
    diagnostics=_validate_calls(calls,started,result,r['config'])
    require({p.name for p in files}==seen|{'journal.jsonl','report.json'},'complete exact file inventory')
    require(sha(out/'report.json')==report_sha256,'final report drift')
    return dict(schema=TAG+'.consume',status='RAM_saved_high_target_verified_NOT_neural_evidence',fixture_only=True,
        report_sha256=report_sha256,attempted=attempted,completed=completed,diagnostics=diagnostics,
        unknown_costs=r['unknown_costs'],scientific_admission=False,learned_prior_established=False,
        production_executed=False,lowlevel_target_delivered=False,production_blockers=list(BLOCKERS))


def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def _validate_calls(calls,started,result,cfg):
    """Independent saved native reshape/target/cost lineage, not a model replay."""
    validate_config(cfg)
    def children(v,kind):return [x for x in calls if x['parent']==v['index'] and x['kind']==kind]
    def one(v,kind):
        xs=children(v,kind);require(len(xs)==1,'exact single native '+kind);return xs[0]
    top=[v for v in calls if v['parent'] is None]
    require([v['kind'] for v in top]==['high_solver','rollout_high'],'one solver then one selected receding rollout, no extra replay')
    parents={v['index']:v['kind'] for v in calls}
    allowed={'high_solver':{None},'get_cost_high':{'high_solver'},'rollout_high':{None,'get_cost_high'},
        'predict_high':{'rollout_high'},'quantize_macro_actions_for_planning':{'rollout_high'},
        **{key:{'predict_high'} for key in MODULES}}
    for v in calls:
        require(parents.get(v['parent']) in allowed[v['kind']],'native call parent semantic')
        inp=v['inputs'];kind=v['kind']
        if kind=='high_solver':n=inp['info']['z_init'].shape[0]
        elif kind=='get_cost_high':n=int(np.prod(inp['candidates'].shape[:2]))
        elif kind=='rollout_high':n=int(inp['macro'].shape[0]*(inp['macro'].shape[1] if inp['macro'].ndim==4 else 1))
        elif kind=='predict_high':n=inp['emb'].shape[0]
        elif kind=='quantize_macro_actions_for_planning':n=inp['macro'].shape[0]
        else:n=inp['args'][0].shape[0]
        require(v['units'][kind+'_rows']==n,'independently recomputed native rows')
    solver,selected=top;info=solver['inputs']['info']
    require(info['planner_level']=='high' and exact(info['z_init'],started['z_init']) and exact(info['z_goal'],started['z_goal']),'source/goal identity')
    initial=solver['inputs']['init_action'];old=started['next_high_init']
    require(initial is old is None or isinstance(initial,np.ndarray) and isinstance(old,np.ndarray) and exact(initial,old),'original warm start')
    actions=solver['output']['actions'];b,d=started['z_init'].shape;keep=cfg['receding_horizon'];block=cfg['action_block']
    require(b==1 and d>0,'single-query denominator')
    macro=selected['inputs']['macro'];require(actions.ndim==3 and actions.shape[:2]==(b,cfg['horizon'])
        and actions.shape[2]==block*macro.shape[-1] and exact(macro,actions[:,:keep].reshape(b,keep*block,-1))
        and exact(selected['inputs']['z_init'],started['z_init']),'original full solver output / receding macro reshape')
    pred=selected['output'];require(pred.shape==(b,1,keep*block,d) and exact(result['target'],pred[:,0,0,:])
        and result['steps_since_high']==0,'native FIRST waypoint, not last/goal relabeling')
    warm=result['next_high_init']
    require(exact(warm,actions[:,keep:]) if cfg['warm_start'] else warm is None,'original next-high CPU warm-start slice')
    cost_rows=[]
    for cost in children(solver,'get_cost_high'):
        inp=cost['inputs'];candidates=inp['candidates'];cr=one(cost,'rollout_high');cb,s,h,group=candidates.shape
        cm=cr['inputs']['macro'];require(cm.shape==(cb,s,h*block,macro.shape[-1])
            and exact(cm,candidates.reshape(cm.shape)),'ALL original cost candidates/grouped macro binding')
        for key in ('z_init','z_goal'):
            a=inp['info'][key];expected=np.broadcast_to(started[key][:,None,:],(cb,s,d))
            if a.ndim==2:a=a[:,None,:]
            require(exact(np.ascontiguousarray(np.broadcast_to(a,expected.shape)),np.ascontiguousarray(expected)),'candidate source/goal expansion')
        require(cr['output'].shape==(cb,s,h*block,d) and cost['output'].shape==(cb,s),'full cost rollout/result shape')
        # Native GPU sum is retained separately. No invented CPU/GPU ULP gate.
        goal=np.broadcast_to(started['z_goal'][:,None,:],(cb,s,d))
        cpu=np.sum((cr['output'][:,:,-1,:].astype(np.float64)-goal.astype(np.float64))**2,axis=-1)
        cost_rows.append(dict(call_index=cost['index'],native_L2sum=cost['output'].tolist(),saved_float64_L2sum=cpu.tolist()))
    require(cost_rows,'solver must make real cost callbacks, not opaque guessed counts')
    for roll in (v for v in calls if v['kind']=='rollout_high'):
        raw_macro=roll['inputs']['macro'];m=raw_macro if raw_macro.ndim==4 else raw_macro[:,None,:,:]
        rb,rs,t,md=m.shape;rp=roll['output'];z=roll['inputs']['z_init']
        z=z[:,None,:] if z.ndim==2 else z
        z=np.broadcast_to(z,(rb,rs,d)).reshape(rb*rs,1,d)
        qs=children(roll,'quantize_macro_actions_for_planning');ps=children(roll,'predict_high')
        require(len(qs)==len(ps)==t and rp.shape==(rb,rs,t,d),'full high temporal calls')
        for k,(q,p) in enumerate(zip(qs,ps)):
            history=m[:,:,:k+1,:].reshape(rb*rs,k+1,md)
            require(exact(q['inputs']['macro'],history),'raw vs actual quantized macro distinction')
            ctx=p['inputs']['emb'].shape[1]
            require(0<ctx<=k+1 and exact(p['inputs']['emb'],z[:,-ctx:])
                and exact(p['inputs']['macro'],q['output'][:,-ctx:]),'actual quantized macro/high context')
            require(exact(p['output'][:,-1,:].reshape(rb,rs,d),rp[:,:,k,:]),'each high prediction to full rollout')
            z=np.concatenate((z,p['output'][:,-1:,:]),axis=1)
            for mod in MODULES:
                rec=one(p,mod)
                require(rec['units'][mod+'_rows']==rec['inputs']['args'][0].shape[0],'actual module rows')
            cond=one(p,'macro_to_condition');net=one(p,'high_predictor');proj=one(p,'high_pred_proj')
            require(exact(cond['inputs']['args'][0],p['inputs']['macro']) and len(net['inputs']['args'])==2
                and exact(net['inputs']['args'][0],p['inputs']['emb']) and exact(net['inputs']['args'][1],cond['output'])
                and exact(proj['inputs']['args'][0],net['output'].reshape(-1,net['output'].shape[-1]))
                and exact(p['output'],proj['output'].reshape(p['output'].shape)),'native module conditioning/projection chain')
    return dict(costs=cost_rows,first_target_sha256=hashlib.sha256(result['target'].tobytes()).hexdigest(),
        target_shape=list(result['target'].shape),selection='native solver actions output, NOT assumed sampled argmin',
        native_cost_vs_float64_bit_equality_claimed=False,lowlevel_execution=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('consume','cuda'))
    p.add_argument('--output',type=Path);p.add_argument('--report-sha256');a=p.parse_args(argv)
    if a.command=='cuda':run_cuda()
    require(a.output is not None and a.report_sha256,'actual saved output/report SHA required')
    print(canonical(consume(a.output,a.report_sha256)).decode())


if __name__=='__main__':main()
