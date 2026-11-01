"""Fixed12 saved factual H1; MAIN-only CUDA. All predictions sealed BEFORE label decoding."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import threading
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_live_feedback_analysis_20261027_v2 as saved
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving

NAME='p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
LIVE=ROOT/'results/p1_moving_live_feedback_physical_20261027_v1_main01/report.json'
LIVE_SHA='990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3'
LIVE_PREP=ROOT/'results/p1_moving_live_feedback_physical_20261027_v1_prepare_main01.json'
LIVE_PREP_SHA='bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5'
ANALYSIS=ROOT/'results/p1_moving_live_feedback_analysis_20261027_v2_main01.json'
ANALYSIS_SHA='5df7dd90b3da010d1bafbb66198334272ec6470b7b0f7bc62991e4b0dae0b5d8'
PREREG=ROOT/'innerdocs/p1_moving_live_feedback_teacher_fidelity_protocol_20261027_v1.md'
PREREG_SHA='43c903c821c161b4c79cc1211e5a3d3d7d6766695511f6294f4b69e1ac037103'
SAVED_SHA='535ea82c8d1d4d0ac8179716a53e04b95cff96f6a48eb2ff3e99d45ee75d8bc2'
MOVING_SHA='246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63'
CAP,FLOOR,FINAL,PREP_CAP=8*1024**2,200*1024**2,256*1024,256*1024
RESOURCES=('action_encoder','predict_API','predictor','pred_proj')
require,canonical,digest=saved.require,saved.canonical,saved.digest
_LOCK=threading.Lock()


def sha(path):
    import hashlib
    p=Path(path);require(p.is_file() and not p.is_symlink(),'regular sealed file '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def fields(value,names):require(type(value) is dict and set(value)==set(names.split()),'exact fields '+names)


def literal(value,shape):
    a=saved.array(value,shape)
    require(np.array_equal(a,a.astype(np.float32)),'original float32 literal, no rounding');return a.astype(np.float32)


def packet_digest(value):
    return saved.sha_bytes((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())


def contract():
    return dict(episodes=12,independent_tasks=2,order=[list(k) for k in saved.KEYS],horizon=1,packet_steps=5,
        resources={k:12 for k in RESOURCES},encoder=0,image_projector=0,AD=0,J=0,SVD=0,physics=0,
        batch=1,latent_tokens=3,action_tokens=3,latent_dim=192,no_retry=True,no_dedup=True,
        label_access='decode only after ALL12 predictions durable-sealed',training=False,selection=False,Qfit=False,
        P1_admission=False,P4_admission=False,independent_confirmation=False,CPU_neural_fallback=False,
        output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,prepare_cap_bytes=PREP_CAP,
        wall_seconds=300,wall_scope='run entry including verify/model load/forward/score/final checks',
        cuda_allocated_cap=3*1024**3,cuda_reserved_cap=4*1024**3,MATH=True,TF32=False,autocast=False,
        normalizer='original task float32 subtract/divide once, training provenance unknown',
        science='known-root teacher factual calibration; no new control, generalization or scientific pass')


def receipt_array(receipt):
    raw=saved.read_bytes(receipt['path'],receipt['file_sha256'],1024**2);stream=BytesIO(raw)
    a=np.load(stream,allow_pickle=False)
    require(stream.tell()==len(raw) and str(a.dtype)==receipt['dtype'] and list(a.shape)==receipt['shape'] and
        saved.sha_bytes(np.ascontiguousarray(a).tobytes())==receipt['array_sha256'],'single original NPY receipt')
    return a


def normalized(raw,mean,std):
    return ((literal(raw,(len(raw),2))-literal(mean,(2,)))/literal(std,(2,))).reshape(-1,10)


def request(identity,old_latents,prefix,warm,past,suffix,mean,std):
    z=literal(old_latents,(3,192));p=literal(prefix,(192,))
    history=np.concatenate((literal(warm,(10,2))[5:],literal(past,(5,2))))
    r=dict(schema=NAME+'.forward_input',identity=deepcopy(identity),
        latents=np.concatenate((z[1:],p[None])).tolist(),history=normalized(history,mean,std).tolist(),
        raw5=literal(suffix,(5,2)).tolist(),mean=literal(mean,(2,)).tolist(),std=literal(std,(2,)).tolist())
    r['input_digest']=digest(r);validate_request(r);return r


def validate_request(r):
    fields(r,'schema identity latents history raw5 mean std input_digest')
    require(r['schema']==NAME+'.forward_input' and r['input_digest']==digest({k:v for k,v in r.items() if k!='input_digest'}),
        'own whitelisted forward input seal; no target/goal/state/path')
    fields(r['identity'],'index episode_id source_slot prefix_arm mode parent_task_id parent_selection_sha256')
    i=r['identity'];require(type(i['index']) is int and 0<=i['index']<12 and saved.key(i)==saved.KEYS[i['index']] and
        i['episode_id']==f'e{i["index"]//3:02}_{i["mode"]}','fixed literal episode identity')
    literal(r['latents'],(3,192));literal(r['history'],(2,10));a=literal(r['raw5'],(5,2))
    require((abs(a)<=1).all() and np.array_equal(literal(r['mean'],(2,)),np.asarray(moving.normalization()['mean'],np.float32)) and
        np.array_equal(literal(r['std'],(2,)),np.asarray(moving.normalization()['std'],np.float32)), 'original raw domain/public normalizer')
    return np.concatenate((literal(r['history'],(2,10)),normalized(a,r['mean'],r['std'])))


def check_trace(entries,inputs,load):
    require(len(entries)==48,'all12 API/module entries')
    for i,r in enumerate(inputs):
        es=entries[4*i:4*i+4]
        require(all(e['index']==4*i+j and e['episode_index']==i and e['resource']==RESOURCES[j] and e['completed'] is True
            for j,e in enumerate(es)),'fixed saved operation index/order/completion')
        ae,api,pred,proj=es
        require(len(ae['inputs'])==len(proj['inputs'])==1 and len(pred['inputs'])==2,'saved module input arity')
        a_in,a_out=load(ae['inputs'][0]),load(ae['output'])
        p_in,pa_in,p_out=load(pred['inputs'][0]),load(pred['inputs'][1]),load(pred['output'])
        q_in,q_out=load(proj['inputs'][0]),load(proj['output'])
        require(a_in.shape==(1,3,10) and np.array_equal(a_in[0],validate_request(r)) and
            a_out.shape[:2]==(1,3) and p_in.shape==(1,3,192) and np.array_equal(p_in[0],r['latents']) and
            np.array_equal(pa_in,a_out) and p_out.shape==(1,3,192) and q_in.shape==q_out.shape==(3,192) and
            np.array_equal(q_in,p_out.reshape(3,192)),'full3 observed context/raw history/native projection saved joins')


def raw_rows(events,kind,steps,field):
    rows=[e for e in events if e['kind']==kind];require([e['step'] for e in rows]==list(steps),'complete factual action chronology')
    values=[]
    for e in rows:
        bits=bytes.fromhex(e['raw_bits']);a=np.frombuffer(bits,dtype=np.float32).copy()
        require(len(bits)==8 and a.shape==(2,) and np.isfinite(a).all() and (abs(a)<=1).all() and
            a.astype(float).tolist()==e[field],'literal actual returned raw float32 bits');values.append(a)
    return np.asarray(values,np.float32)


def build(report_path=str(LIVE),report_sha=LIVE_SHA,live_prepared=str(LIVE_PREP),live_prepare_sha=LIVE_PREP_SHA,
          analysis_path=str(ANALYSIS),analysis_sha=ANALYSIS_SHA):
    require((report_sha,live_prepare_sha,analysis_sha)==(LIVE_SHA,LIVE_PREP_SHA,ANALYSIS_SHA),'fixed approved actual authorities')
    bindings={}
    def bind(path,h=None):
        path=str(Path(path).resolve());h=sha(path) if h is None else h;saved.hash_value(h)
        require(sha(path)==h and (path not in bindings or bindings[path]==h),'nonconflicting saved/source/model seal '+path)
        bindings[path]=h;return Path(path)
    def read(path,h=None):
        f=bind(path,h);require(f.stat().st_size<=64*1024**2,'bounded saved JSON');return saved.parse(f.read_bytes())
    bind(saved.__file__,SAVED_SHA);bind(moving.__file__,MOVING_SHA);bind(PREREG,PREREG_SHA)
    rp=bind(report_path,report_sha);base=rp.parent;r=read(rp,report_sha);p=read(live_prepared,live_prepare_sha)
    saved.validate_contract(p,r)
    require(r['status']=='live_feedback_calibration_recorded' and r['errors']==[] and r['full_horizons']==12 and
        r['model_final_check']['model_unchanged'] is True and r['prefix_lineage']['matched'] is True,'terminal full12 completed before teacher')
    b=read(base/'execution_binding.json');require(Path(b['prepared']).resolve()==Path(live_prepared).resolve() and
        b['prepare_sha256']==live_prepare_sha,'actual driver execution binding')
    analysis=read(analysis_path,analysis_sha)
    require(analysis['schema']==saved.NAME+'.report' and analysis['observed_endpoint_count']==24 and
        analysis['inputs']['report_sha256']==report_sha and analysis['inputs']['prepare_sha256']==live_prepare_sha and
        all(x['prefix_pairing_exact'] is True for x in analysis['contrasts']),'MAIN complete paired saved analysis authority')
    jp=bind(base/'journal.jsonl',r['journal_sha256']);events=saved.journal(jp.read_bytes(),r['journal_chain_tail'])
    saved.census(events,r,p)
    require(events[-1]['kind']=='run_finished' and events[-1]['status']==r['status'] and
        events[-1]['counts']==r['counts'] and events[-1]['errors']==[],'actual terminal journal')
    require([saved.key(x) for x in r['rows']]==list(saved.KEYS),'all12 rows in fixed order')
    tasks={};inputs=[];evaluators=[]
    for layer in p['layers']:
        slot=layer['source_slot'];path=bind(layer['task_export'],layer['task_sha256'])
        t=moving.load_task(str(path),layer['task_sha256']);require(t['task_id']==layer['task_id'],'original same moving task')
        tasks[slot]=t['gn_task']
    for index,row in enumerate(r['rows']):
        slot,arm,mode=saved.key(row);ep=row['episode_id'];t=tasks[slot]
        layer=next(x for x in p['layers'] if x['source_slot']==slot and x['arm']==arm)
        require(row['status']=='completed10' and row['full_horizon'] is True and row['initial_exact'] is True and
            row['first_error'] is None and not row['secondary_errors'] and
            (row['warmup_returned'],row['prefix_returned'],row['suffix_returned'])==(10,5,5),'own completed factual10')
        scoped=[e for e in events if e.get('episode_id')==ep]
        closed=[e for e in events if e['kind']=='episode_closed' and e['receipt']['episode_id']==ep]
        require(len(closed)==1 and closed[0]['receipt']==row,'own durable row closure')
        warm=raw_rows(scoped,'warmup_return',range(1,11),'raw')
        controls=raw_rows(scoped,'control_return',range(1,11),'command');prefix,suffix=controls[:5],controls[5:]
        require(np.array_equal(warm,moving.fixed_raw()) and np.array_equal(prefix,literal(layer['commands'],(10,2))[:5]),'own original warmup/prefix')
        seal=read(base/(ep+'_dispatch_seal.json'))
        require(seal==row['dispatch_seal'] and seal['seal_sha256']==digest({k:v for k,v in seal.items() if k!='seal_sha256'}) and
            seal['episode_id']==ep and seal['source_slot']==slot and seal['prefix_arm']==arm and seal['mode']==mode and
            seal['parent_task_id']==layer['task_id'] and seal['parent_selection_sha256']==layer['selection_sha256'] and
            seal['commands_sha256']==saved.sha_bytes(suffix.tobytes()) and np.array_equal(seal['commands'],suffix),'own executed suffix seal')
        authorized=[e for e in scoped if e['kind']=='authorized_dispatch']
        require([e['step'] for e in authorized]==list(range(6,11)) and [e['command'] for e in authorized]==suffix.astype(float).tolist() and
            all(e['seal_sha256']==seal['seal_sha256'] for e in authorized),'literal authorized vs returned suffix')
        if mode=='O':require(np.array_equal(suffix,literal(layer['commands'],(10,2))[5:]),'O own original suffix')
        else:
            req=read(base/(ep+'_suffix_request.json'));response=read(base/(ep+'_suffix_response.json'))
            require(req['mode']==mode and packet_digest(req)==row['suffix_request_sha256']==seal['request_sha256'] and
                response['packet_sha256']==seal['request_sha256'] and packet_digest(response)==seal['response_sha256'] and
                response['model_unchanged'] is True and response['source_verified'] is True and
                np.array_equal(response['dispatch_raw5'],suffix),'I/R own actual compiler suffix, no recompute or reselection')
            for name,h in response['artifact_sha256'].items():bind(saved.local(base/Path(row['suffix_namespace']).name,name),h)
        encodings={}
        for step in (5,10):
            cp=saved.checkpoint(row,step);require(cp is not None,'own required checkpoint')
            saved.array(cp['state'],(7,));saved.array(cp['goal'],(7,));saved.array(row['final_state'],(7,))
            png=bind(saved.local(base,cp['path']),cp['file_sha256'])
            from PIL import Image
            with Image.open(png) as im:a=np.asarray(im).copy()
            require(a.dtype==np.uint8 and a.shape==(224,224,3) and saved.sha_bytes(a.tobytes())==cp['pixel_sha256'],'original fresh PNG pixels, no encoding')
            fresh=[e for e in scoped if e['kind']=='fresh_observation' and e['receipt']==cp]
            guards=[e for e in scoped if e['kind']=='guard_passed' and e['phase']=='plan' and e['step']==step]
            require(len(fresh)==len(guards)==1 and fresh[0]['index']<guards[0]['index'],'fresh saved observation/guard')
            q=read(base/(ep+f'_encode{step}_request.json'));v=read(base/(ep+f'_encode{step}_response.json'))
            require(q['schema']==saved.DRIVER+'.encode_request' and q['episode_id']==ep and q['observed_step']==step and
                q['request_sha256']==digest({k:x for k,x in q.items() if k!='request_sha256'}) and
                q['parent_task_id']==layer['task_id'] and q['task_export_sha256']==layer['task_sha256'] and
                q['parent_selection_sha256']==layer['selection_sha256'] and q['png']=={k:cp[k] for k in ('path','file_sha256','pixel_sha256')} and
                v==row[f'encoding{step}'] and v['schema']==saved.DRIVER+'.encode_response' and v['episode_id']==ep and
                v['observed_step']==step and v['request_sha256']==q['request_sha256'] and v['device_type']=='cuda' and
                v['source_verified'] is True and v['model_unchanged'] is True and v['model_hash_before']==v['model_hash_after']==moving.low.MODEL_TENSOR_SHA and
                v['resources']=={k:dict(attempted=1,completed=1) for k in ('encoder','projector')},'own original CUDA visual receipt')
            receipt=v['latent'];require(Path(receipt['path'])==base/(ep+f'_encode{step}')/'native192.npy' and
                receipt['dtype']=='float32' and receipt['shape']==[192],'own native encoding namespace')
            bind(receipt['path'],receipt['file_sha256'])  # Hash bytes, NEVER decode terminal label here.
            encodings[step]=receipt
        prefix_latent=receipt_array(encodings[5]);saved.array(prefix_latent,(192,),np.dtype('float32'))
        identity=dict(index=index,episode_id=ep,source_slot=slot,prefix_arm=arm,mode=mode,
            parent_task_id=layer['task_id'],parent_selection_sha256=layer['selection_sha256'])
        inputs.append(request(identity,t['root_context']['latent_history'],prefix_latent,warm,prefix,suffix,t['action_mean'],t['action_std']))
        evaluators.append(dict(identity=identity,label=encodings[10],goal=deepcopy(t['goal']),prefix=encodings[5]))
    require(p['modelbinding']['tensor_sha256']==moving.low.MODEL_TENSOR_SHA and p['modelbinding']['model_files']==moving.MODELS,'original frozen model binding')
    for f,h in {**p['modelbinding']['source_closure'],**moving.MODELS}.items():bind(f,h)
    for f in moving.source_files()+[Path(saved.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bind(f)
    for f in (ROOT/'experiments/checkpoint_compat.py',ROOT/'experiments/gpu_preflight.py'):bind(f)
    arguments=dict(report_path=str(rp),report_sha=report_sha,live_prepared=str(Path(live_prepared).resolve()),
        live_prepare_sha=live_prepare_sha,analysis_path=str(Path(analysis_path).resolve()),analysis_sha=analysis_sha)
    for f,h in bindings.items():require(sha(f)==h,'final prepare seal')
    return dict(schema=NAME+'.prepare',arguments=arguments,inputs=inputs,evaluators=evaluators,bindings=bindings,
        contract=contract(),policy=str(moving.POLICY),expected_model_hash=moving.low.MODEL_TENSOR_SHA)


def prepare(output,**arguments):
    p=build(**arguments);raw=canonical(p);path=Path(output)
    require(not path.exists() and len(raw)<=PREP_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive small prepare/200MiB floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path.resolve()),prepare_sha256=sha(path),bytes=len(raw))


def verify(prepared,expected):
    p=saved.parse(saved.read_bytes(prepared,expected,PREP_CAP));require(p==build(**p['arguments']),'complete own saved reconstruction');return p


class Store:
    def __init__(self,output):
        self.output=Path(output).absolute();require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW8MiB plus200MiB floor')
        self.output.mkdir();self.previous=None;self.index=0
    def write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..'),'flat exclusive output')
        path=self.output/name;require(not path.is_symlink(),'nonsymlink output')
        used=sum(f.stat().st_size for f in self.output.iterdir() if f.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'whole8MiB/floor including final')
        with path.open('ab' if append else 'xb') as f:require(f.write(raw)==len(raw),'short artifact');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return sha(path)
    def put(self,name,value,final=False):return self.write(name,canonical(value),final=final)
    def event(self,e):
        v=dict(index=self.index,previous_sha256=self.previous,**deepcopy(e));v['event_sha256']=digest(v)
        self.write('events.jsonl',canonical(v),append=True);self.previous=v['event_sha256'];self.index+=1
    def array(self,name,a):
        a=np.asarray(a);require(a.dtype==np.float32 and np.isfinite(a).all() and a.nbytes<=1024**2,'native float32 bounded artifact')
        b=BytesIO();np.save(b,a,allow_pickle=False);h=self.write(name,b.getvalue())
        return dict(file=name,sha256=h,dtype='float32',shape=list(a.shape))


class Ledger:
    def __init__(self,store,check=lambda:None):self.store,self.check=store,check;self.entries=[];self.current=None;self.blocked=[]
    def begin(self,kind):
        self.check();n=len(self.entries);require(n<48 and self.current==n//4 and kind==RESOURCES[n%4], 'fixed12 actual API/module order; no retry')
        require(not self.entries or self.entries[-1]['completed'] or kind=='predictor','previous actual operation must return')
        e=dict(index=n,episode_index=self.current,resource=kind,completed=False);self.entries.append(e)
        self.store.event(dict(kind='operation_attempt',entry=e));return e
    def end(self,e):
        self.check();require(not e['completed'],'single matching actual return')
        if e['resource']=='predict_API':require(all(v['completed'] for v in self.entries[-2:]),'all native predict modules before API return')
        e['completed']=True;self.store.event(dict(kind='operation_return',entry=e))
    def deny(self,name):
        def blocked(*a,**kw):
            self.blocked.append(name);self.store.event(dict(kind='forbidden',name=name));raise ValueError('forbidden '+name)
        return blocked
    def counts(self):return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in RESOURCES}


@contextmanager
def no_ad(torch,ledger,errors):
    require(_LOCK.acquire(blocking=False),'exclusive process AD instrumentation');undo=[]
    try:
        groups=[(torch.autograd,('grad','backward','_engine_run_backward')),(torch.Tensor,('backward',)),
            (torch.autograd.functional,('jacobian','hessian','jvp','vjp','hvp','vhp')),
            (torch.func,('jacrev','jacfwd','hessian','jvp','vjp','grad','grad_and_value')),
            (torch.linalg,('svd','solve','lstsq','pinv'))]
        for obj,names in groups:
            for name in names:
                if hasattr(obj,name):undo.append((obj,name,getattr(obj,name)));setattr(obj,name,ledger.deny('AD/math:'+name))
        yield
    finally:
        for obj,name,old in reversed(undo):
            try:setattr(obj,name,old)
            except BaseException as e:errors.append(dict(stage='restore_AD',message=str(e)))
        _LOCK.release()


class Native:
    """Explicit original model.predict H1; actual module arrays and full3 chronology."""
    def __init__(self,model,torch,device,ledger,store):
        require(getattr(device,'type',None)=='cuda','HOST CUDA only; no CPU neural fallback')
        self.model,self.torch,self.device,self.ledger,self.store=model,torch,device,ledger,store
        self.hooks=[];self.pending={};self.request=None;self.embedding=self.prediction=self.projected=None
    def array(self,x):
        require(x.device==self.device and x.dtype==self.torch.float32,'native CUDA float32 module tensor')
        a=x.detach().cpu().numpy().copy();require(np.isfinite(a).all(),'finite native module');return a
    def pre(self,kind):
        def enter(module,args):
            require(not self.pending and self.request is not None and not self.torch.is_grad_enabled(),'inference only single native module')
            e=self.ledger.begin(kind);self.pending[module]=e;e['inputs']=[]
            for n,x in enumerate(args):
                a=self.array(x)
                if kind=='action_encoder':require(n==0 and a.shape==(1,3,10) and np.array_equal(a[0],validate_request(self.request)),'full3 real normalized actions')
                elif kind=='predictor':
                    require(n<2 and (a.shape==(1,3,192) and np.array_equal(a[0],self.request['latents']) if n==0 else
                        self.embedding is not None and np.array_equal(a,self.embedding)),'full3 real latent/actual action embedding')
                else:require(n==0 and a.shape==(3,192) and self.prediction is not None and np.array_equal(a,self.prediction.reshape(3,192)),'unmodified full3 pred_proj input')
                e['inputs'].append(self.store.array(f'op_{e["index"]:02d}_input{n}.npy',a))
            require(len(args)==(2 if kind=='predictor' else 1),'exact module arity')
            self.store.event(dict(kind='module_inputs_saved',entry=e))
        return enter
    def post(self,module,args,out):
        e=self.pending.get(module)
        if e is None or out is None:return
        self.torch.cuda.synchronize(self.device);a=self.array(out);kind=e['resource']
        if kind=='action_encoder':require(a.ndim==3 and a.shape[:2]==(1,3),'full3 action embeddings');self.embedding=a
        elif kind=='predictor':require(a.shape==(1,3,192),'full3 predictor');self.prediction=a
        else:require(a.shape==(3,192),'full3 projected outputs');self.projected=a
        e['output']=self.store.array(f'op_{e["index"]:02d}_output.npy',a);self.pending.pop(module);self.ledger.end(e)
    def __enter__(self):
        self.had='predict' in self.model.__dict__;self.old_local=self.model.__dict__.get('predict');self.old_predict=self.model.predict
        def counted(*a,**kw):
            e=self.ledger.begin('predict_API');out=self.old_predict(*a,**kw);self.ledger.end(e);return out
        try:
            self.model.predict=counted
            for mod,kind in ((self.model.action_encoder,'action_encoder'),(self.model.predictor,'predictor'),(self.model.pred_proj,'pred_proj')):
                self.hooks.append(mod.register_forward_pre_hook(self.pre(kind)));self.hooks.append(mod.register_forward_hook(self.post,always_call=True))
            for mod in (self.model.encoder,self.model.projector):self.hooks.append(mod.register_forward_pre_hook(self.ledger.deny('image_encoder/projector')))
            return self
        except BaseException as e:self.__exit__(type(e),e,e.__traceback__);raise
    def __exit__(self,typ,exc,tb):
        errors=[]
        for h in reversed(self.hooks):
            try:h.remove()
            except BaseException as e:errors.append(e)
        try:
            if self.had:self.model.predict=self.old_local
            else:del self.model.predict
        except BaseException as e:errors.append(e)
        if errors:
            if exc is not None:
                for e in errors:exc.add_note('hook cleanup '+str(e))
            else:raise errors[0]
    def __call__(self,r):
        outgoing=validate_request(r);self.request=deepcopy(r);self.embedding=self.prediction=self.projected=None
        tensor=lambda a:self.torch.tensor(a,dtype=self.torch.float32,device=self.device)
        z=tensor(r['latents'])[None];a=tensor(outgoing)[None]
        # Python evaluates AE first; counted predict wrapper records actual API entry only afterward.
        out=self.model.predict(z,self.model.action_encoder(a))
        self.torch.cuda.synchronize(self.device);value=self.array(out)
        require(value.shape==(1,3,192) and self.projected is not None and np.array_equal(value.reshape(3,192),self.projected) and
            not self.pending,'actual full3 native predict output');self.request=None
        return value[0,-1].copy()


def initial_rows(p):return [dict(identity=deepcopy(r['identity']),status='not_started',prediction=None,label=None,metrics=None) for r in p['inputs']]


def metrics(pred,label,goal,prefix):
    for z in (pred,label,prefix):saved.array(z,(192,),np.dtype('float32'))
    goal=literal(goal,(192,)).astype(float);q=lambda a,b:float(np.mean((a.astype(float)-b.astype(float))**2))
    return dict(factual_MSE=q(pred,label),predicted_Q=q(pred,goal),actual_Q=q(label,goal),
        predicted_Q_minus_actual_Q=q(pred,goal)-q(label,goal),actual_prefix_terminal_mse=q(prefix,label))


def execute(p,store,ledger,forward,rows,check=lambda:None,label_loader=receipt_array):
    require(len(p['inputs'])==len(p['evaluators'])==len(rows)==12,'fixed12 no subset')
    store.put('initial_rows.json',rows);store.event(dict(kind='run_started',contract=p['contract']))
    predictions=[]
    for i,r in enumerate(p['inputs']):
        check();validate_request(r);require(r['identity']['index']==i,'fixed prediction order')
        row=rows[i];row['status']='attempted';ledger.current=i;store.put(f'input_{i:02d}.json',r)
        store.event(dict(kind='prediction_attempt',identity=r['identity'],input_digest=r['input_digest']))
        z=forward(deepcopy(r));saved.array(z,(192,),np.dtype('float32'))
        require(len(ledger.entries)==4*(i+1) and all(e['completed'] for e in ledger.entries),'actual single forward receipt')
        row.update(status='predicted',prediction=store.array(f'prediction_{i:02d}.npy',z));predictions.append(z.copy())
        store.event(dict(kind='prediction_saved',receipt=row));ledger.current=None
    require(not ledger.blocked and ledger.counts()=={k:dict(attempted=12,completed=12) for k in RESOURCES},'exact12 forward/resources before labels')
    seal=dict(schema=NAME+'.predictions_seal',rows=deepcopy(rows),resources=ledger.counts(),ledger=deepcopy(ledger.entries))
    h=store.put('predictions_seal.json',seal);store.event(dict(kind='all_predictions_sealed',sha256=h))
    for i,e in enumerate(p['evaluators']):
        check();require(e['identity']==rows[i]['identity'],'own episode label join after seal')
        label=label_loader(e['label']);prefix=label_loader(e['prefix'])
        rows[i]['label']=store.array(f'label_{i:02d}.npy',saved.array(label,(192,),np.dtype('float32')))
        rows[i].update(status='scored',metrics=metrics(predictions[i],label,e['goal'],prefix))
        store.event(dict(kind='episode_scored',receipt=rows[i]))
    return h


def error(stage,e):return dict(stage=stage,type=type(e).__name__,message=str(e),notes=list(getattr(e,'__notes__',[])))


def run(prepared,expected,output):
    started=time.monotonic();p=verify(prepared,expected);store=Store(output);rows=initial_rows(p);errors=[]
    model=None;before=after=host=None;source_ok=False;seal=None;ledger=Ledger(store)
    def check():
        require(time.monotonic()-started<300,'whole job300s including load/verify')
        if model is not None:
            require(torch.cuda.memory_allocated(device)<=3*1024**3 and torch.cuda.memory_reserved(device)<=4*1024**3,'3GiB allocated/4GiB reserved')
        used=sum(f.stat().st_size for f in store.output.iterdir() if f.is_file())
        require(used<CAP-FINAL and shutil.disk_usage(store.output).free>=FLOOR,'runtime byte/floor gate')
    ledger.check=check
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));check()
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');check()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        moving.low.frozen(model);before=moving.low.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen tensor SHA');check()
        with no_ad(torch,ledger,errors),torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH),Native(model,torch,device,ledger,store) as forward:
            seal=execute(p,store,ledger,forward,rows,check)
    except BaseException as e:
        errors.append(error('execution',e))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary:errors.append(error('error_journal',secondary))
    finally:
        ledger.current=None
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:moving.low.frozen(model);after=moving.low.tensor_hash(model)
            except BaseException as e:errors.append(error('final_model',e))
        try:verify(prepared,expected);source_ok=True;check()
        except BaseException as e:errors.append(error('final_source_or_limits',e))
        complete=(not errors and not ledger.blocked and source_ok and before is not None and before==after and seal is not None and
            all(r['status']=='scored' for r in rows) and ledger.counts()=={k:dict(attempted=12,completed=12) for k in RESOURCES})
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:files=[];errors.append(error('final_listing',e));complete=False
        for f in files:
            try:hashes[f.name]=sha(f)
            except BaseException as e:errors.append(error('final_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='teacher_factual_recorded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,rows=rows,errors=errors,first_error=errors[0] if errors else None,
            resources=ledger.counts(),ledger=ledger.entries,blocked=ledger.blocked,predictions_seal_sha256=seal,artifact_sha256=hashes,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,source_verified=source_ok,
            contract=p['contract'],fixed_denominator=12,independent_tasks=2,independent_confirmation=False,P1_admission=False,P4_admission=False,
            training=False,selection=False,new_AD=0,new_encoder=0,new_physics=0,host=host,wall_seconds=time.monotonic()-started)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(error('final_report',e));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial evidence retained, no retry: '+str(errors));return report


def consume(output,report_sha):
    root=Path(output).resolve();r=saved.parse(saved.read_bytes(root/'report.json',report_sha));p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='teacher_factual_recorded_not_admitted' and not r['errors'] and not r['blocked'] and
        r['source_verified'] is True and r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['contract']==contract() and r['fixed_denominator']==12 and len(r['rows'])==12 and
        r['resources']=={k:dict(attempted=12,completed=12) for k in RESOURCES},'complete actual saved teacher report')
    require({f.name for f in root.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'exact actual artifact set')
    for name,h in r['artifact_sha256'].items():require(sha(saved.local(root,name))==h,'actual artifact SHA')
    seal=saved.parse(saved.read_bytes(root/'predictions_seal.json',r['predictions_seal_sha256']))
    require(seal['schema']==NAME+'.predictions_seal' and seal['resources']==r['resources'] and seal['ledger']==r['ledger'],'actual12 prediction seal')
    events=saved.journal((root/'events.jsonl').read_bytes(),saved.parse((root/'events.jsonl').read_bytes().splitlines()[-1])['event_sha256'])
    seals=[e for e in events if e['kind']=='all_predictions_sealed'];scored=[e for e in events if e['kind']=='episode_scored']
    require(len(seals)==1 and seals[0]['sha256']==r['predictions_seal_sha256'] and len(scored)==12 and
        all(e['index']>seals[0]['index'] for e in scored),'all12 sealed before any scoring')
    def load(entry):
        raw=saved.read_bytes(saved.local(root,entry['file']),entry['sha256'],1024**2);stream=BytesIO(raw);z=np.load(stream,allow_pickle=False)
        require(stream.tell()==len(raw) and str(z.dtype)==entry['dtype'] and list(z.shape)==entry['shape'],'saved output NPY');return z
    for i,(row,ev) in enumerate(zip(r['rows'],p['evaluators'],strict=True)):
        require(row['identity']==ev['identity']==p['inputs'][i]['identity'] and row['status']=='scored' and
            scored[i]['receipt']==row and seal['rows'][i]['status']=='predicted' and seal['rows'][i]['prediction']==row['prediction'],'exact own episode sealed prediction/score')
        pred,label=load(row['prediction']),load(row['label']);original=receipt_array(ev['label']);prefix=receipt_array(ev['prefix'])
        require(np.array_equal(label,original) and row['metrics']==metrics(pred,label,ev['goal'],prefix),'own factual label and mean192 metrics')
        projected=load(r['ledger'][4*i+3]['output'])
        require(np.array_equal(pred,projected[-1]),'prediction exact final native projected token')
    check_trace(r['ledger'],p['inputs'],load)
    return dict(schema=NAME+'.consume',report_sha256=report_sha,rows=deepcopy(r['rows']),resources=r['resources'],
        fixed_denominator=12,independent_tasks=2,training=False,selection=False,P1_admission=False,P4_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='verify':r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command=='consume':r=consume(a.output,a.report_sha)
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
