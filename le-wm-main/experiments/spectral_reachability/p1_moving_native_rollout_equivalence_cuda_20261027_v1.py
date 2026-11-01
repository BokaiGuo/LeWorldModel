"""Official cached LeWM.rollout vs sealed packet traces; MAIN HOST CUDA only."""
from __future__ import annotations
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import sysconfig
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_teacher_horizon_localization_cuda_20261027_v1 as horizon

core=horizon.core
NAME='p1_moving_native_rollout_equivalence_cuda_20261027_v1'
ROOT=core.ROOT
OFFICIAL=Path(sysconfig.get_paths()['purelib'])/'stable_worldmodel/wm/lewm/lewm.py'
OFFICIAL_SHA='ccc2ac2880880f85ea71a5baa2152ba97e3f6dc5cb546d27570795aa6020d3d2'
HORIZON_SHA='8b8e4bdb708b914b3fc50bba9e0b6c3b624cc724396c05d660cc79432879e1ed'
REFERENCE=ROOT/'results/p1_moving_teacher_horizon_localization_cuda_20261027_v1_cuda_MAIN01/report.json'
REFERENCE_SHA='e319035d9e13a6ae2dc4cbfa5cefb9eb2e6ccd49dbcf35408fda1704f5f34253'
REFERENCE_PREP_SHA='c87b35732ebaa3a3f6c74ec2e65ec817d9b15f319e0b687cc3ffa453f03935ec'
KINDS=('root_H2','suffix_H1')
RESOURCES=('rollout_API','action_encoder','predict_API','predictor','pred_proj')
require,canonical,digest,sha=core.require,core.canonical,core.digest,core.sha
CAP,FLOOR,FINAL,PREP_CAP,SEMANTIC_CAP=core.CAP,core.FLOOR,core.FINAL,core.PREP_CAP,core.SEMANTIC_CAP
Store=core.Store


def contract():
    return dict(episodes=12,independent_tasks=2,queries=24,kinds=list(KINDS),packet_steps=5,history_size=3,
        resources=dict(rollout_API=24,action_encoder=24,predict_API=36,predictor=36,pred_proj=36),
        encoder=0,image_projector=0,AD=0,J=0,SVD=0,physics=0,training=False,selection=False,Qfit=False,
        API_scope='direct official class rollout with cached emb; no get_cost/criterion/encode calls',
        official_source_sha256=OFFICIAL_SHA,normalizer='same original public float32 subtract/divide once',
        output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,prepare_cap_bytes=PREP_CAP,
        prepare_semantic_cap_bytes=SEMANTIC_CAP,wall_seconds=300,cuda_allocated_cap=3*1024**3,cuda_reserved_cap=4*1024**3,
        MATH=True,TF32=False,autocast=False,CPU_neural_fallback=False,no_retry=True,no_dedup=True,
        comparison='bitwise/maxabs/ULP only; no tolerance fitting or declaration of scientific equivalence',
        criterion='official sum192; saved arithmetic float64 sum192 and mean192 separately, no criterion invocation',
        P1_admission=False,P4_admission=False,independent_confirmation=False)


def validate_query(q):
    core.fields(q,'schema identity kind emb action_sequence rgb_history')
    require(q['schema']==NAME+'.rollout_input' and q['kind'] in KINDS,'own official rollout request')
    i=q['identity'];core.fields(i,'index episode_id source_slot prefix_arm mode parent_task_id parent_selection_sha256')
    require(type(i['index']) is int and 0<=i['index']<12 and core.saved.key(i)==core.saved.KEYS[i['index']] and
        i['episode_id']==f'e{i["index"]//3:02}_{i["mode"]}','same own episode identity')
    core.literal(q['emb'],(1,1,3,192));t=4 if q['kind']=='root_H2' else 3
    core.literal(q['action_sequence'],(1,1,t,10))
    expected=[('warmup',s) for s in (0,5,10)] if t==4 else [('warmup',5),('plan',0),('plan',5)]
    require(type(q['rgb_history']) is list and len(q['rgb_history'])==3,'three REAL past RGB receipts')
    for r,(phase,step) in zip(q['rgb_history'],expected,strict=True):
        core.fields(r,'phase step path file_sha256 pixel_sha256')
        require((r['phase'],r['step'])==(phase,step) and Path(r['path']).name==f'{i["episode_id"]}_{phase}_{step:02}.png',
            'past frames own episode exact; no future/dummy')
        core.saved.hash_value(r['file_sha256']);core.saved.hash_value(r['pixel_sha256'])
    return t


def make_queries(p,events):
    queries=[]
    for i,(root,suffix) in enumerate(zip(p['root_inputs'],p['inputs'],strict=True)):
        require(root['identity']==suffix['identity'] and root['identity']['index']==i,'own fixed12 requests')
        ep=root['identity']['episode_id'];obs=[e['receipt'] for e in events if e['kind']=='fresh_observation' and e.get('episode_id')==ep]
        def rgb(phase,step):
            found=[r for r in obs if r['phase']==phase and r['step']==step];require(len(found)==1,'unique actual causal RGB')
            return {k:found[0][k] for k in ('phase','step','path','file_sha256','pixel_sha256')}
        require(rgb('warmup',10)['pixel_sha256']==rgb('plan',0)['pixel_sha256'],'goal attachment did not change root RGB')
        tokens=core.validate_request(root)
        require(np.array_equal(tokens[1:],suffix['history']),'actual between-action chronology')
        four=np.concatenate((tokens,core.normalized(core.literal(suffix['raw5'],(5,2)),suffix['mean'],suffix['std'])))
        for kind in KINDS:
            emb=root['latents'] if kind=='root_H2' else suffix['latents']
            actions=four if kind=='root_H2' else core.validate_request(suffix)
            history=[rgb('warmup',s) for s in (0,5,10)] if kind=='root_H2' else [rgb('warmup',5),rgb('plan',0),rgb('plan',5)]
            q=dict(schema=NAME+'.rollout_input',identity=deepcopy(root['identity']),kind=kind,
                emb=np.asarray(emb,np.float32)[None,None].tolist(),action_sequence=actions[None,None].tolist(),rgb_history=history)
            validate_query(q);queries.append(q)
    require(len(queries)==24,'fixed24 not selected subset');return queries


def build():
    require(sha(OFFICIAL)==OFFICIAL_SHA and sha(horizon.__file__)==HORIZON_SHA,'official and sealed horizon source')
    ref=core.saved.parse(core.saved.read_bytes(REFERENCE,REFERENCE_SHA))
    require(ref['schema']==horizon.NAME+'.report' and ref['status']=='horizon_localization_recorded_not_admitted' and
        ref['errors']==[] and ref['model_unchanged'] is True and ref['prepare_sha256']==REFERENCE_PREP_SHA,'actual reference complete')
    p=horizon.verify(ref['prepared'],REFERENCE_PREP_SHA)  # No terminal-label decoding; source reconstruction only.
    require(ref['source_verified'] is True and ref['blocked']==[] and
        ref['model_hash_before']==ref['model_hash_after']==p['expected_model_hash'] and
        ref['contract']==horizon.contract() and ref['resources']=={k:dict(attempted=36,completed=36) for k in core.RESOURCES} and
        len(ref['rows'])==12 and len(ref['ledger'])==144,'reference fixed12/36 native resources')
    bindings=deepcopy(p['bindings'])
    def bind(path,h=None):
        f=Path(path).resolve();value=sha(f) if h is None else h
        require(sha(f)==value and (str(f) not in bindings or bindings[str(f)]==value),'full authority bindings')
        bindings[str(f)]=value;return f
    bind(REFERENCE,REFERENCE_SHA);bind(ref['prepared'],REFERENCE_PREP_SHA);bind(OFFICIAL,OFFICIAL_SHA)
    for name,h in ref['artifact_sha256'].items():bind(core.saved.local(REFERENCE.parent,name),h)
    a=p['arguments'];live=core.saved.parse(core.saved.read_bytes(a['report_path'],a['report_sha']))
    events=core.saved.journal((Path(a['report_path']).parent/'journal.jsonl').read_bytes(),live['journal_chain_tail'])
    queries=make_queries(p,events);references=[]
    for i in range(12):
        row=ref['rows'][i];require(row['identity']==p['inputs'][i]['identity'] and row['status']=='scored','reference own identity')
        for kind,steps in (('root_H2',(0,1)),('suffix_H1',(2,))):
            references.append(dict(identity=deepcopy(row['identity']),kind=kind,
                finals=[deepcopy(row['predictions'][horizon.STAGES[s]]) for s in steps],
                module_entries=[deepcopy(ref['ledger'][12*i+4*s:12*i+4*s+4]) for s in steps]))
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bind(f)
    for f,h in bindings.items():require(sha(f)==h,'final source closure')
    return dict(schema=NAME+'.prepare',queries=queries,references=references,
        goals=[dict(identity=deepcopy(e['identity']),goal=deepcopy(e['goal'])) for e in p['evaluators']],
        bindings=bindings,policy=p['policy'],expected_model_hash=p['expected_model_hash'],contract=contract(),
        reference_report=dict(path=str(REFERENCE),sha256=REFERENCE_SHA),official_source=dict(path=str(OFFICIAL),sha256=OFFICIAL_SHA),
        source_census=p['source_census'])


def encode_prepare(p):
    import base64,gzip,json
    semantic=canonical(p);require(len(semantic)<=SEMANTIC_CAP,'semantic1MiB')
    raw=json.dumps(dict(schema=NAME+'.prepare_envelope',codec='single-gzip+base64',semantic_bytes=len(semantic),
        semantic_sha256=core.saved.sha_bytes(semantic),payload=base64.b64encode(gzip.compress(semantic,mtime=0)).decode()),
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    require(len(raw)<=PREP_CAP,'disk256KiB');return raw


def decode_prepare(raw):
    import base64,zlib
    require(type(raw) is bytes and len(raw)<=PREP_CAP,'disk256KiB');e=core.saved.parse(raw)
    core.fields(e,'schema codec semantic_bytes semantic_sha256 payload')
    require(e['schema']==NAME+'.prepare_envelope' and e['codec']=='single-gzip+base64' and
        type(e['semantic_bytes']) is int and 0<e['semantic_bytes']<=SEMANTIC_CAP,'own bounded prepare')
    packed=base64.b64decode(e['payload'],validate=True);require(base64.b64encode(packed).decode()==e['payload'],'canonical base64')
    d=zlib.decompressobj(31);semantic=d.decompress(packed,SEMANTIC_CAP+1)
    require(len(semantic)<=SEMANTIC_CAP and d.eof and not d.unused_data and not d.unconsumed_tail and
        len(semantic)==e['semantic_bytes'] and core.saved.sha_bytes(semantic)==e['semantic_sha256'],'single gzip/SHA/bound')
    p=core.saved.parse(semantic);require(canonical(p)==semantic and p['schema']==NAME+'.prepare','canonical semantic');return p


def prepare(output):
    raw=encode_prepare(build());f=Path(output)
    require(not f.exists() and shutil.disk_usage(f.parent).free>=FLOOR+len(raw),'NEW prepare/200MiB floor')
    with f.open('xb') as out:require(out.write(raw)==len(raw),'short prepare');out.flush();os.fsync(out.fileno())
    return dict(prepared=str(f.resolve()),prepare_sha256=sha(f),bytes=len(raw))


def verify(prepared,expected):
    p=decode_prepare(core.saved.read_bytes(prepared,expected,PREP_CAP));require(p==build(),'complete own source/request reconstruction');return p


def operation_plan():
    plan=[]
    for call in range(24):
        plan.extend([(call,'rollout_API',None),(call,'action_encoder',None)])
        for step in range(2 if call%2==0 else 1):plan.extend((call,k,step) for k in ('predict_API','predictor','pred_proj'))
    return plan


class Ledger(core.Ledger):
    def begin(self,kind):
        self.check();n=len(self.entries);plan=operation_plan()
        require(n<len(plan) and (self.current,kind)==plan[n][:2],'fixed24 official rollout/module order no retry')
        call,kind,step=plan[n];e=dict(index=n,call_index=call,resource=kind,step=step,completed=False)
        self.entries.append(e);self.store.event(dict(kind='operation_attempt',entry=e));return e
    def end(self,e):
        self.check();require(not e['completed'],'single actual completion')
        if e['resource']=='predict_API':require(all(x['completed'] for x in self.entries[-2:]),'predictor/proj before API return')
        if e['resource']=='rollout_API':require(all(x['completed'] for x in self.entries if x['call_index']==e['call_index'] and x is not e),'all official operations before rollout return')
        e['completed']=True;self.store.event(dict(kind='operation_return',entry=e))
    def counts(self):return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in RESOURCES}


def verify_class(model):
    cls=type(model)
    require(cls.__module__=='stable_worldmodel.wm.lewm.lewm' and cls.__name__=='LeWM' and
        Path(inspect.getsourcefile(cls)).resolve()==OFFICIAL.resolve() and sha(OFFICIAL)==OFFICIAL_SHA,'actual installed official LeWM class')
    for name in ('rollout','predict','encode'):
        require(name not in model.__dict__ and getattr(model,name).__func__ is getattr(cls,name) and
            Path(getattr(cls,name).__code__.co_filename).resolve()==OFFICIAL.resolve(),'unmodified official class API '+name)


class NativeRollout:
    """Instrumentation only. The rollout loop is the installed method, never reimplemented here."""
    def __init__(self,model,torch,device,ledger,store,pixels):
        require(getattr(device,'type',None)=='cuda','HOSTCUDA only no CPU neural fallback')
        self.model,self.torch,self.device,self.ledger,self.store,self.pixels=model,torch,device,ledger,store,pixels
        self.hooks=[];self.pending={};self.query=None;self.embedding=None;self.projected=[]
    def array(self,x):
        require(x.device==self.device and x.dtype==self.torch.float32,'actual CUDA float32 tensor')
        a=x.detach().cpu().numpy().copy();require(np.isfinite(a).all(),'finite native tensor');return a
    def pre(self,kind):
        def enter(module,args):
            require(self.query is not None and not self.pending and not self.torch.is_grad_enabled(),'cached official inference module only')
            e=self.ledger.begin(kind);self.pending[module]=e;e['inputs']=[]
            for j,x in enumerate(args):
                a=self.array(x)
                if kind=='action_encoder':expected=np.asarray(self.query['action_sequence'],np.float32).reshape(1,-1,10)
                elif kind=='predictor' and j==0:
                    z=np.asarray(self.query['emb'],np.float32).reshape(1,3,192)
                    expected=z if not self.projected else np.concatenate((z[:,1:],self.projected[0][-1][None,None]),axis=1)
                elif kind=='predictor':
                    require(self.embedding is not None,'actual full sequence AE before predict')
                    step=len(self.projected);expected=self.embedding[:,step:step+3]
                else:expected=self.predictor_output.reshape(3,192)
                require(a.shape==expected.shape and a.tobytes()==expected.tobytes(),'actual official module input '+kind)
                e['inputs'].append(self.store.array(f'op_{e["index"]:03d}_input{j}.npy',a))
            require(len(args)==(2 if kind=='predictor' else 1),'exact module arity')
            self.store.event(dict(kind='module_inputs_saved',entry=e))
        return enter
    def post(self,module,args,out):
        e=self.pending.get(module)
        if e is None or out is None:return
        self.torch.cuda.synchronize(self.device);a=self.array(out);kind=e['resource']
        if kind=='action_encoder':
            require(a.ndim==3 and a.shape[:2]==(1,validate_query(self.query)),'one full-sequence AE output');self.embedding=a
        elif kind=='predictor':require(a.shape==(1,3,192),'full3 official predictor');self.predictor_output=a
        else:require(a.shape==(3,192),'full3 official pred_proj');self.projected.append(a)
        e['output']=self.store.array(f'op_{e["index"]:03d}_output.npy',a);self.pending.pop(module);self.ledger.end(e)
    def __enter__(self):
        verify_class(self.model);self.old_predict=self.model.predict;self.old_encode=self.model.encode
        def counted(*args,**kwargs):
            require(len(args)==2 and not kwargs,'official predict exact call')
            e=self.ledger.begin('predict_API');out=self.old_predict(*args)
            self.torch.cuda.synchronize(self.device);a=self.array(out)
            require(a.shape==(1,3,192) and self.projected and a.reshape(3,192).tobytes()==self.projected[-1].tobytes(),'API output actual pred_proj')
            e['output']=self.store.array(f'op_{e["index"]:03d}_output.npy',a);self.ledger.end(e);return out
        try:
            self.model.predict=counted;self.model.encode=self.ledger.deny('encode_API')
            for module,kind in ((self.model.action_encoder,'action_encoder'),(self.model.predictor,'predictor'),(self.model.pred_proj,'pred_proj')):
                self.hooks.append(module.register_forward_pre_hook(self.pre(kind)));self.hooks.append(module.register_forward_hook(self.post,always_call=True))
            for module in (self.model.encoder,self.model.projector):self.hooks.append(module.register_forward_pre_hook(self.ledger.deny('image_encoder/projector')))
            return self
        except BaseException as e:self.__exit__(type(e),e,e.__traceback__);raise
    def __exit__(self,typ,exc,tb):
        errors=[]
        for hook in reversed(self.hooks):
            try:hook.remove()
            except BaseException as e:errors.append(e)
        for name in ('predict','encode'):
            try:delattr(self.model,name)
            except BaseException as e:errors.append(e)
        if errors:
            if exc is not None:
                for e in errors:exc.add_note('native hook restore '+str(e))
            else:raise errors[0]
    def __call__(self,q):
        t=validate_query(q);self.query=deepcopy(q);self.embedding=None;self.projected=[]
        tensor=lambda a:self.torch.tensor(a,dtype=self.torch.float32,device=self.device)
        pixels=self.pixels(q['rgb_history']);pa=self.array(pixels)
        require(pa.shape==(1,1,3,3,224,224),'REAL original-transform RGB3, legal official pixels shape')
        emb=tensor(q['emb']);actions=tensor(q['action_sequence']);call=self.ledger.current
        self.store.put(f'call_{call:02d}_input.json',dict(schema=NAME+'.actual_rollout_call',identity=q['identity'],kind=q['kind'],
            pixels_shape=list(pa.shape),pixels_dtype=str(pa.dtype),pixels_array_sha256=core.saved.sha_bytes(pa.tobytes()),
            rgb_history=q['rgb_history'],emb=self.store.array(f'call_{call:02d}_emb.npy',self.array(emb)),
            action_sequence=self.store.array(f'call_{call:02d}_actions.npy',self.array(actions)),history_size=3))
        info=dict(pixels=pixels,emb=emb);e=self.ledger.begin('rollout_API')
        # DIRECT installed API. It alone performs cat/split/AE once and autoregressive predictor calls.
        result=self.model.rollout(info,actions,history_size=3)
        self.torch.cuda.synchronize(self.device)
        require(result is info and result['emb'] is emb and result['pixels'] is pixels and not self.pending and len(self.projected)==t-2,'official cached emb returned; exact prediction count')
        action_prefix=self.array(result['action'])
        require(action_prefix.shape==(1,1,3,10) and action_prefix.tobytes()==np.asarray(q['action_sequence'],np.float32)[:,:,:3].tobytes(),'official H=3 action prefix')
        e['action_prefix']=self.store.array(f'call_{call:02d}_action_prefix.npy',action_prefix)
        pred=self.array(result['predicted_emb']);require(pred.shape==(1,1,t+1,192),'official full predicted_emb shape')
        expected=np.concatenate((np.asarray(q['emb'],np.float32).reshape(3,192),np.stack([x[-1] for x in self.projected])))
        require(pred.reshape(t+1,192).tobytes()==expected.tobytes(),'official full original3 plus actual successor tokens')
        e['output']=self.store.array(f'call_{call:02d}_predicted_emb.npy',pred);self.ledger.end(e);self.query=None
        return pred


def compare(a,b):
    require(isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype==np.float32 and
        np.isfinite(a).all() and np.isfinite(b).all(),'finite original float32 compare no casting/tolerance')
    def ordered(x):
        u=np.ascontiguousarray(x).view(np.uint32).astype(np.int64)
        return np.where(u>=0x80000000,0x80000000-(u&0x7fffffff),0x80000000+u)
    return dict(shape=list(a.shape),bitwise=a.tobytes()==b.tobytes(),numeric_equal=bool(np.array_equal(a,b)),
        maxabs=float(np.max(np.abs(a.astype(float)-b.astype(float)))),maxULP=int(np.max(np.abs(ordered(a)-ordered(b)))))


def evaluate(q,pred,reference,entries,goal,load_new,load_old):
    t=validate_query(q);require(pred.dtype==np.float32 and pred.shape==(1,1,t+1,192),'full official saved rollout')
    require(reference['identity']==q['identity'] and reference['kind']==q['kind'],'own source reference')
    z=np.asarray(q['emb'],np.float32).reshape(3,192);finals=[load_old(e) for e in reference['finals']]
    expected=np.concatenate((z,np.stack(finals)))[None,None]
    # Module trace comparison includes common sequence windows of one AE(T) vs repeated AE(3).
    ae=next(e for e in entries if e['resource']=='action_encoder');aeout=load_new(ae['output']);modules=[]
    for step,old in enumerate(reference['module_entries']):
        oldae,api,op,oq=old
        npred=next(e for e in entries if e['resource']=='predictor' and e['step']==step)
        nproj=next(e for e in entries if e['resource']=='pred_proj' and e['step']==step)
        modules.append(dict(action_embedding=compare(aeout[:,step:step+3],load_old(oldae['output'])),
            latent_input=compare(load_new(npred['inputs'][0]),load_old(op['inputs'][0])),
            action_input=compare(load_new(npred['inputs'][1]),load_old(op['inputs'][1])),
            predictor_output=compare(load_new(npred['output']),load_old(op['output'])),
            full3_projected=compare(load_new(nproj['output']),load_old(oq['output']))))
    g=core.literal(goal,(192,)).astype(float);last=pred[0,0,-1].astype(float);oldlast=finals[-1].astype(float)
    costs=lambda x:dict(mean192_float64=float(np.mean((x-g)**2)),sum192_float64=float(np.sum((x-g)**2)))
    return dict(full_predicted_emb=compare(pred,expected),final=compare(pred[0,0,-1],finals[-1]),
        successor_comparisons=[compare(pred[0,0,3+j],a) for j,a in enumerate(finals)],modules=modules,
        cost_native=costs(last),cost_reference=costs(oldlast),criterion_API_called=False,
        criterion_note='official float32 mse then sum192; these are saved float64 arithmetic, not criterion bitwise output')


def initial_rows(p):
    return [dict(identity=deepcopy(q['identity']),kind=q['kind'],status='not_started',predicted_emb=None,final=None,comparison=None) for q in p['queries']]


def execute(p,store,ledger,forward,rows,check=lambda:None,load_new=None,load_old=None):
    require(len(p['queries'])==len(p['references'])==len(rows)==24 and len(p['goals'])==12,'all24 queries/12 tasks')
    store.put('initial_rows.json',rows);store.event(dict(kind='run_started',contract=p['contract']))
    for n,(q,row) in enumerate(zip(p['queries'],rows,strict=True)):
        require(q['identity']['index']==n//2 and q['kind']==KINDS[n%2] and row['identity']==q['identity'],'fixed whole cohort')
        check();row['status']='attempted';ledger.current=n;validate_query(q)
        store.put(f'query_{n:02d}.json',q);store.event(dict(kind='rollout_attempt',call_index=n,identity=q['identity'],query_kind=q['kind']))
        pred=forward(deepcopy(q));require(pred.dtype==np.float32 and pred.shape==(1,1,validate_query(q)+1,192) and np.isfinite(pred).all(),'original official output')
        row.update(status='predicted',predicted_emb=store.array(f'prediction_{n:02d}.npy',pred),final=store.array(f'final_{n:02d}.npy',pred[0,0,-1]))
        store.event(dict(kind='rollout_saved',call_index=n,receipt=deepcopy(row)))
    ledger.current=None
    require(not ledger.blocked and ledger.counts()=={k:dict(attempted=v,completed=v) for k,v in contract()['resources'].items()},'actual24/24/36 before any evaluation')
    seal=store.put('predictions_seal.json',dict(schema=NAME+'.predictions_seal',rows=deepcopy(rows),ledger=deepcopy(ledger.entries),resources=ledger.counts()))
    store.event(dict(kind='all_predictions_sealed',sha256=seal))
    load_new=(lambda e:horizon.load_array(store.output,e)) if load_new is None else load_new
    # Reference predictions and module labels FIRST decode after whole24 rollout seal.
    load_old=(lambda e:horizon.load_array(REFERENCE.parent,e)) if load_old is None else load_old
    for n,(q,ref,row) in enumerate(zip(p['queries'],p['references'],rows,strict=True)):
        check();goal=p['goals'][n//2];require(goal['identity']==q['identity'],'same original goal, evaluator only')
        row['comparison']=evaluate(q,load_new(row['predicted_emb']),ref,[e for e in ledger.entries if e['call_index']==n],goal['goal'],load_new,load_old)
        row['status']='compared';store.event(dict(kind='rollout_compared',call_index=n,receipt=deepcopy(row)))
    return seal


def run(prepared,expected,output):
    started=time.monotonic();p=verify(prepared,expected);store=Store(output);rows=initial_rows(p);errors=[]
    model=None;before=after=host=None;source_ok=False;seal=None;ledger=Ledger(store)
    def check():
        require(time.monotonic()-started<300,'whole300s including verify/load/final')
        if model is not None:require(torch.cuda.memory_allocated(device)<=3*1024**3 and torch.cuda.memory_reserved(device)<=4*1024**3,'CUDA3/4GiB')
        require(sum(f.stat().st_size for f in store.output.iterdir() if f.is_file())<CAP-FINAL and shutil.disk_usage(store.output).free>=FLOOR,'8MiB/200MiB')
    ledger.check=check
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));check()
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        from PIL import Image
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');check()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);verify_class(model)
        core.moving.low.frozen(model);before=core.moving.low.tensor_hash(model);require(before==p['expected_model_hash'],'original model tensor SHA')
        transform=p0.image_transform()
        def pixels(receipts):
            frames=[]
            for r in receipts:
                require(sha(r['path'])==r['file_sha256'],'own actual past PNG SHA')
                with Image.open(r['path']) as im:rgb=np.asarray(im).copy()
                require(rgb.dtype==np.uint8 and rgb.shape==(224,224,3) and core.saved.sha_bytes(rgb.tobytes())==r['pixel_sha256'],'real original past pixels')
                frames.append(p0._image_tensor(rgb,transform=transform,device=device)[0,0])
            return torch.stack(frames)[None,None]
        with core.no_ad(torch,ledger,errors),torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH),NativeRollout(model,torch,device,ledger,store,pixels) as forward:
            seal=execute(p,store,ledger,forward,rows,check)
    except BaseException as e:
        errors.append(core.error('execution',e))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary:errors.append(core.error('error_journal',secondary))
    finally:
        ledger.current=None
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:core.moving.low.frozen(model);after=core.moving.low.tensor_hash(model)
            except BaseException as e:errors.append(core.error('final_model',e))
        try:verify(prepared,expected);source_ok=True;check()
        except BaseException as e:errors.append(core.error('final_source_or_limits',e))
        complete=not errors and not ledger.blocked and source_ok and before is not None and before==after and seal is not None and all(r['status']=='compared' for r in rows)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:files=[];errors.append(core.error('final_listing',e));complete=False
        for f in files:
            try:hashes[f.name]=sha(f)
            except BaseException as e:errors.append(core.error('final_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='native_rollout_comparison_recorded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,rows=rows,errors=errors,first_error=errors[0] if errors else None,
            resources=ledger.counts(),ledger=ledger.entries,blocked=ledger.blocked,predictions_seal_sha256=seal,artifact_sha256=hashes,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,source_verified=source_ok,
            contract=contract(),fixed_episode_denominator=12,fixed_rollout_denominator=24,independent_tasks=2,
            training=False,selection=False,P1_admission=False,P4_admission=False,new_AD=0,new_encoder=0,new_physics=0,host=host,wall_seconds=time.monotonic()-started)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(core.error('final_report',e));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial saved evidence no retry: '+str(errors));return report


def validate_trace(q,pred,entries,load):
    t=validate_query(q);plan=operation_plan();call=entries[0]['call_index']
    require([(e['call_index'],e['resource'],e['step']) for e in entries]==[x for x in plan if x[0]==call] and all(e['completed'] for e in entries),'whole official call trace')
    ae=next(e for e in entries if e['resource']=='action_encoder');actions=load(ae['inputs'][0]);embedding=load(ae['output'])
    require(actions.shape==(1,t,10) and actions.tobytes()==np.asarray(q['action_sequence'],np.float32).reshape(1,t,10).tobytes() and
        embedding.ndim==3 and embedding.shape[:2]==(1,t),'actual once-only AE full sequence')
    z=np.asarray(q['emb'],np.float32).reshape(3,192);require(pred.shape==(1,1,t+1,192) and pred[0,0,:3].tobytes()==z.tobytes(),'official original cached frames')
    for step in range(t-2):
        api=next(e for e in entries if e['resource']=='predict_API' and e['step']==step)
        p=next(e for e in entries if e['resource']=='predictor' and e['step']==step)
        proj=next(e for e in entries if e['resource']=='pred_proj' and e['step']==step)
        require(len(p['inputs'])==2 and len(proj['inputs'])==1,'native module arity')
        pz,pa,po=load(p['inputs'][0]),load(p['inputs'][1]),load(p['output']);qi,qo=load(proj['inputs'][0]),load(proj['output'])
        require(pz.shape==(1,3,192) and pz[0].tobytes()==pred[0,0,step:step+3].tobytes() and
            pa.tobytes()==embedding[:,step:step+3].tobytes() and pa.shape==embedding[:,step:step+3].shape and
            po.shape==(1,3,192) and qi.shape==qo.shape==(3,192) and qi.tobytes()==po.reshape(3,192).tobytes() and
            load(api['output']).reshape(3,192).tobytes()==qo.tobytes() and pred[0,0,3+step].tobytes()==qo[-1].tobytes(),'actual official loop context/slice/native outputs')
    rollout=entries[0];require(load(rollout['output']).tobytes()==pred.tobytes() and
        load(rollout['action_prefix']).shape==(1,1,3,10) and
        load(rollout['action_prefix']).tobytes()==np.asarray(q['action_sequence'],np.float32)[:,:,:3].tobytes(),'actual API full predicted_emb and H3 prefix')


def consume(output,report_sha):
    root=Path(output).resolve();r=core.saved.parse(core.saved.read_bytes(root/'report.json',report_sha));p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='native_rollout_comparison_recorded_not_admitted' and not r['errors'] and not r['blocked'] and
        r['source_verified'] is True and r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['contract']==contract() and r['fixed_episode_denominator']==12 and r['fixed_rollout_denominator']==24 and len(r['rows'])==24 and
        r['resources']=={k:dict(attempted=v,completed=v) for k,v in contract()['resources'].items()},'complete native comparison')
    require({f.name for f in root.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'all actual artifacts')
    for name,h in r['artifact_sha256'].items():require(sha(core.saved.local(root,name))==h,'actual artifact SHA')
    seal=core.saved.parse(core.saved.read_bytes(root/'predictions_seal.json',r['predictions_seal_sha256']))
    require(seal['schema']==NAME+'.predictions_seal' and seal['ledger']==r['ledger'] and seal['resources']==r['resources'] and len(seal['rows'])==24,'whole24 seal')
    raw=(root/'events.jsonl').read_bytes();events=core.saved.journal(raw,core.saved.parse(raw.splitlines()[-1])['event_sha256'])
    seals=[e for e in events if e['kind']=='all_predictions_sealed'];saved=[e for e in events if e['kind']=='rollout_saved'];compared=[e for e in events if e['kind']=='rollout_compared']
    require(len(seals)==1 and seals[0]['sha256']==r['predictions_seal_sha256'] and len(saved)==len(compared)==24 and
        [e['call_index'] for e in saved]==list(range(24)) and all(e['index']<seals[0]['index'] for e in saved) and
        all(e['index']>seals[0]['index'] for e in compared),'all24 sealed before any reference evaluation')
    attempts=[e['entry'] for e in events if e['kind']=='operation_attempt'];returns=[e['entry'] for e in events if e['kind']=='operation_return']
    require(len(attempts)==len(returns)==len(r['ledger'])==156 and sorted(returns,key=lambda e:e['index'])==r['ledger'] and
        [(e['call_index'],e['resource'],e['step']) for e in attempts]==operation_plan(),'actual all156 operations durable')
    load=lambda e:horizon.load_array(root,e);old=lambda e:horizon.load_array(REFERENCE.parent,e)
    for n,(q,row,ref) in enumerate(zip(p['queries'],r['rows'],p['references'],strict=True)):
        require(row['identity']==q['identity'] and row['kind']==q['kind'] and row['status']=='compared' and compared[n]['receipt']==row and
            seal['rows'][n]==saved[n]['receipt'] and seal['rows'][n]['predicted_emb']==row['predicted_emb'] and
            core.saved.parse((root/f'query_{n:02d}.json').read_bytes())==q,'own exact saved rollout request/row')
        pred=load(row['predicted_emb']);require(load(row['final']).tobytes()==pred[0,0,-1].tobytes(),'own final192')
        entries=[e for e in r['ledger'] if e['call_index']==n];validate_trace(q,pred,entries,load)
        call=core.saved.parse((root/f'call_{n:02d}_input.json').read_bytes())
        require(call['identity']==q['identity'] and call['kind']==q['kind'] and call['rgb_history']==q['rgb_history'] and
            call['pixels_shape']==[1,1,3,3,224,224] and call['pixels_dtype']=='float32' and call['history_size']==3 and
            load(call['emb']).tobytes()==np.asarray(q['emb'],np.float32).tobytes() and
            load(call['action_sequence']).tobytes()==np.asarray(q['action_sequence'],np.float32).tobytes(),'actual shapes/cached data/causal RGB provenance')
        require(row['comparison']==evaluate(q,pred,ref,entries,p['goals'][n//2]['goal'],load,old),'exact saved numerical comparisons')
    return dict(schema=NAME+'.consume',report_sha256=report_sha,rows=deepcopy(r['rows']),resources=r['resources'],
        fixed_episode_denominator=12,fixed_rollout_denominator=24,P1_admission=False,P4_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('prepared','prepare-sha','output','report-sha'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':value=prepare(a.output)
    elif a.command=='verify':value=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command=='consume':value=consume(a.output,a.report_sha)
    else:value=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(value).decode(),end='')


if __name__=='__main__':main()
