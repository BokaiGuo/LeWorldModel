"""Fixed known-task Execute/Observe/Recompile calibration; MAIN CUDA only.

Four sealed prefixes, actual prefix RGB latents, one bounded-ridge H1 each.
Old suffix is diagnostic only, NEVER a dynamic selection alternative.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import moving_history_planners_cuda_20261027_v1 as planner
from experiments.spectral_reachability import p1_constructive_moving_history_rgb_cuda_20261027_v1 as observed

NAME='moving_history_observed_suffix_cuda_20261027_v1'
ROOT,ARMS=planner.ROOT,planner.ARMS
blind=planner.blind
d2p=blind.d2p
compiler=d2p.compiler
core=planner.core
require,sha,canonical=planner.require,planner.sha,planner.canonical
read=observed.read
CAP,FLOOR,FINAL=blind.CAP,blind.FLOOR,blind.FINAL
PLANNER_SHA='f8f1d5507e2474fbbbfa56cae98f7bfa99f5929829946a2d7760c2c30ccc03c1'
RGB_REPORT=ROOT/'results/p1_constructive_moving_history_rgb_cuda_20261027_v1_cuda_main01/report.json'
RGB_REPORT_SHA='7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71'
RGB_SOURCE_SHA='ed9b80bebb212434d60ddd55d3a3cba01e5cc55306170a0ea6f9a167bb156469'
RGB_PREPARE_SHA='5b5de956f6a6bc45f9569e40d85546f809efd06334f6a8fc5aeaf2fb66947d7a'
STAGES=('nominal','jacobian','old_suffix','new_suffix','new_suffix_replay')
Store=blind.Store


def budget():return dict(predict=5,action_encoder=5,pred_proj=5,jacobian=1,svd=1)


def contract():
    return dict(branches=4,roots=2,arms=list(ARMS),order=[[s,a] for s in (0,1) for a in ARMS],stages=list(STAGES),
        per_branch=budget(),total={k:4*v for k,v in budget().items()},AD_API_per_branch=dict(grad=1,jacobian=1),
        mathematical_VJP_per_branch=192,mathematical_VJP_total=768,ADforward_in_predict_per_branch=1,
        observed_context='original latent_history[1:] + actual selected prefix192; warmup raw[5:] + own selected prefixraw5',
        horizon=1,packet_steps=5,goal='unchanged original goal',Q='original identity192 no fit',
        damping=compiler.DAMPING,packet_logit_cap=compiler.CAP,
        dispatch='always new bounded-ridge suffix; old suffix diagnostic only, never dynamic reselect',
        model='same frozen native CUDA checkpoint',numerics=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),
        inherited_costs='source/old planner/physical/RGB8 paid separately; not equal total compute',
        no_retries=True,no_dedup=True,new_images=0,new_physics=0,training=False,new_Q=False,CPU_neural=False,
        calibration='fixed known-task observation-conditioned replanning; not independent success or non-greedy evidence',
        P1_admission=False,P4_admission=False,output_cap_bytes=CAP,disk_floor_bytes=FLOOR)


def load_npy(path,entry):
    require(entry['shape']==[192] and entry['dtype']=='float32' and sha(path)==entry['sha256'],'actual prefix NPY binding')
    require(Path(path).stat().st_size<=4096,'bounded native192 NPY')
    raw=Path(path).read_bytes();stream=BytesIO(raw);a=np.load(stream,allow_pickle=False)
    observed.enc.native(a)
    require(stream.tell()==len(raw) and observed.enc.digest(raw)==entry['sha256'],'single exact prefix NPY bytes')
    return a,raw


def branch_input(slot,arm,i,plan,item,z,prefix_npy):
    """Pure input compiler. No physical labels or terminal actual latents enter."""
    i=planner.interface(i);t=i['gn_task'];observed.enc.native(z)
    require(type(slot) is int and slot in (0,1) and arm in ARMS and
        plan['source_slot']==slot and type(plan['source_slot']) is int and plan['arm']==arm and plan['task_id']==i['task_id'] and
        plan['task_export']==planner.EXPORTS[slot] and plan['task_sha256']==planner.EXPORT_SHAS[slot] and
        item['source_slot']==slot and type(item['source_slot']) is int and item['root_slot']==slot and
        item['arm']==arm and item['role']=='prefix5' and item['task_id']==i['task_id'] and
        item['task_export']==plan['task_export'] and item['task_sha256']==plan['task_sha256'] and
        item['blind_sha256']==plan['blind_sha256'] and item['planner_selection_sha256']==plan['selection_sha256'] and
        item['checkpoint']['step']==5 and item['checkpoint']['phase']=='plan',
        'literal own moving root/arm/task/selected prefix binding')
    body={k:v for k,v in plan.items() if k not in ('selection_sha256','whole_run_report_sha256')}
    require(plan['schema']==planner.NAME+'.model_plan' and core.digest(body)==plan['selection_sha256'] and
        plan['with_target'] is False and plan['selection_uses_physics'] is False and plan['P1_admission'] is False,
        'frozen own selection before observing, not outcome replacement')
    raw=np.asarray(plan['commands']);require(raw.shape==(10,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and
        (abs(raw)<=1).all() and np.array_equal(raw,raw.astype(np.float32)), 'literal float32 raw10, no clip')
    past=np.concatenate((planner.fixed_raw()[5:],raw[:5].astype(np.float32)))
    root=t['root_context'];context=dict(root,latent_history=deepcopy(root['latent_history'][1:])+[z.astype(float).tolist()],
        action_history=core.normalized_tokens(t,past).astype(float).tolist())
    core.context(context)
    p=dict(source_slot=slot,arm=arm,parent_task_id=i['task_id'],task_export=plan['task_export'],task_sha256=plan['task_sha256'],
        blind_sha256=plan['blind_sha256'],planner_selection_sha256=plan['selection_sha256'],
        physical_selection_digest=item['selection_digest'],checkpoint=deepcopy(item['checkpoint']),
        prefix_npy=deepcopy(prefix_npy),prefix_commands=raw[:5].tolist(),old_suffix_commands=raw[5:].tolist(),
        observed_raw_history=past.astype(float).tolist(),observed_context=context,goal=deepcopy(t['goal']),q=deepcopy(i['q']),
        action_mean=deepcopy(t['action_mean']),action_std=deepcopy(t['action_std']),packet_logit_cap=t['packet_logit_cap'])
    p['branch_id']=core.digest(p);return p


def build(rgb_report,rgb_sha,rgb_source_sha):
    for h in (rgb_sha,rgb_source_sha):core.hash_value(h)
    require(str(Path(rgb_report).absolute())==str(RGB_REPORT) and rgb_sha==RGB_REPORT_SHA and rgb_source_sha==RGB_SOURCE_SHA,'MAIN fixed actual RGB report/source')
    require(sha(planner.__file__)==PLANNER_SHA and sha(observed.__file__)==rgb_source_sha,'frozen own planner and externally pinned actual RGB source')
    rp=Path(rgb_report).absolute();require(rp.name=='report.json','explicit actual RGB report.json')
    r=read(rp,rgb_sha)
    require(r['schema']==observed.NAME+'.report' and r['status']=='moving_selected_rgb_encoded_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==observed.enc.MODEL_TENSOR_SHA and
        r['resources']=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')},'actual complete RGB8 with unchanged model')
    bindings={str(rp):rgb_sha,str(Path(observed.__file__).resolve()):rgb_source_sha}
    bp=rp.parent/'binding.json';binding=read(bp);require(binding['prepare_sha256']==r['prepare_sha256']==RGB_PREPARE_SHA,'RGB actual prepare binding')
    prepared=observed.verify(binding['prepared'],r['prepare_sha256'])  # Saved-only own new schema.
    require(prepared['arguments']==r['inputs'] and prepared['contract']==r['contract'] and len(prepared['images'])==8,'same original eight fixed images')
    bindings.update(prepared['bindings']);bindings.update({str(bp):sha(bp),str(Path(binding['prepared']).absolute()):r['prepare_sha256']})
    names={'binding.json','events.jsonl'}|{f'image_{k:02d}.native192.npy' for k in range(8)}
    require(set(r['artifact_sha256'])==names and {f.name for f in rp.parent.iterdir()}==names|{'report.json'},'exact actual RGB8 artifact census')
    for name,h in r['artifact_sha256'].items():
        path=observed.rgb.old.local(rp.parent,name);require(sha(path)==h,'RGB artifact SHA');bindings[str(path)]=h
    events=[blind.decode(line) for line in (rp.parent/'events.jsonl').read_bytes().splitlines()]
    require(len(events)>0 and (rp.parent/'events.jsonl').read_bytes().endswith(b'\n'),'complete RGB journal')
    attempts=[e['provenance'] for e in events if e['kind']=='image_attempt'];returns=[e['receipt'] for e in events if e['kind']=='image_saved']
    require(attempts==prepared['images'] and returns==r['rows'],'all eight actual image attempts/returns')
    require([(x['source_slot'],x['arm'],x['role']) for x in prepared['images']]==[(s,a,role) for s in (0,1) for a in ARMS for role in ('prefix5','terminal10')], 'fixed four prefixes, no favorable-root selection')
    selected=planner.consume(prepared['arguments']['planner_report'].rsplit('/',1)[0],prepared['arguments']['planner_sha'])['plans']
    indexed={(p['source_slot'],p['arm']):p for p in selected};require(len(indexed)==len(selected)==4,'all four sealed original plans')
    branches=[]
    for k,(row,item) in enumerate(zip(r['rows'],prepared['images'],strict=True)):
        require(row['status']=='encoded' and row['provenance']==item and row['index']==k and
            row['source_slot']==item['source_slot'] and row['arm']==item['arm'] and row['task_id']==item['task_id'] and
            row['role']==item['role'] and row['latent']['file']==f'image_{k:02d}.native192.npy' and
            row['latent']['sha256']==r['artifact_sha256'][row['latent']['file']], 'each fixed encoded image release')
        if item['role']!='prefix5':continue  # NEVER load/use old-suffix actual terminal arrays as targets.
        slot=item['source_slot'];arm=item['arm'];entry=row['latent'];path=rp.parent/entry['file']
        z,_=load_npy(path,entry);i=planner.load_export(planner.EXPORTS[slot],planner.EXPORT_SHAS[slot])
        branches.append(branch_input(slot,arm,i,indexed[slot,arm],item,z,dict(path=str(path),**entry)))
    require([(b['source_slot'],b['arm']) for b in branches]==[(s,a) for s in (0,1) for a in ARMS],'complete fixed four observed branches')
    files=blind.source_files()+observed.moving.source_files()+[Path(planner.__file__),Path(observed.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings.update({str(Path(f).resolve()):sha(f) for f in files})
    for path,h in bindings.items():require(sha(path)==h,'full input/source/model final TOCTOU')
    return dict(schema=NAME+'.prepare',arguments=dict(rgb_report=str(rp),rgb_sha=rgb_sha,rgb_source_sha=rgb_source_sha),
        branches=branches,bindings=bindings,policy=str(observed.enc.POLICY),expected_model_hash=observed.enc.MODEL_TENSOR_SHA,contract=contract())


def prepare(output,**arguments):
    p=build(**arguments);path=Path(output).absolute();raw=canonical(p)
    require(not path.exists() and len(raw)<2*1024**2 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive <=2MiB prepare (four explicit Q192 matrices) +200MiB')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'own external prepare SHA');p=blind.read(path)
    require(p==build(**p['arguments']),'complete own observed-context reconstruction');return p


def save_array(store,name,a):
    a=np.asarray(a);require(a.dtype in (np.dtype('float32'),np.dtype('float64')) and np.isfinite(a).all(),'finite real original arrays')
    b=BytesIO();np.save(b,a,allow_pickle=False);raw=b.getvalue();store._write(name,raw)
    return dict(file=name,sha256=observed.enc.digest(raw),dtype=str(a.dtype),shape=list(a.shape))


class Meter:
    def __init__(self,sink,sync=lambda:None):
        self.sink,self.sync=sink,sync;self.entries=[];self.failed=False;self.stage=None
        self.require_native=False;self.expected_input=None;self.last_native=None
    def call(self,resource,fn):
        require(not self.failed and resource in ('jacobian','svd') and sum(e['resource']==resource for e in self.entries)<1,'one nonretry math call')
        e=dict(index=len(self.entries),resource=resource,completed=False,stage=self.stage)
        if resource=='jacobian':e.update(shape=[192,10],mathematical_VJP=192)
        self.entries.append(e)
        try:
            self.sink(dict(kind='math_attempt',entry=deepcopy(e)));value=fn();self.sync();e['completed']=True
            self.sink(dict(kind='math_return',entry=deepcopy(e)));return value
        except BaseException:self.failed=True;raise
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in ('jacobian','svd')}


class SuffixAD(blind.ADTrace):
    def dispatch(self,kind,fn,args,kwargs):
        require(self.stage()=='jacobian' and kind in ('grad','jacobian') and not self.failed and
            sum(e['resource']==kind for e in self.events)<1 and (kind!='grad' or self.depth==1),'single H1 vectorized Jacobian API, no extra backward')
        if kind=='grad':require(kwargs.get('is_grads_batched') is True,'192-output batched VJP API')
        e=dict(index=len(self.events),resource=kind,completed=False,is_grads_batched=bool(kwargs.get('is_grads_batched',False)))
        self.events.append(e)
        if kind=='jacobian':self.depth+=1
        try:
            self.sink(dict(kind='AD_API_attempt',entry=deepcopy(e)))
            value=fn(*args,**kwargs);self.sync();e['completed']=True;self.sink(dict(kind='AD_API_completed',entry=deepcopy(e)));return value
        except BaseException:self.failed=True;raise
        finally:
            if kind=='jacobian':self.depth-=1


class Modules:
    """Actual complete native module inputs/outputs, before arrays survive failures."""
    ORDER=('action_encoder','predict','pred_proj')
    def __init__(self,model,store,stem,meter,sync=lambda:None):
        self.model,self.store,self.stem,self.meter,self.sync=model,store,stem,meter,sync
        self.entries=[];self.hooks=[];self.pending={};self.failed=False
        self.action_embedding=None;self.predictor_output=None
    def array(self,x):
        require(getattr(getattr(x,'device',None),'type',None)=='cuda','CUDA-only module arrays')
        a=x.detach().cpu().numpy();require(a.dtype==np.float32 and np.isfinite(a).all(),'native finite float32 module arrays');return a.copy()
    def pre(self,kind):
        def enter(module,args):
            require(not self.failed and not self.pending and len(self.entries)<15 and
                kind==self.ORDER[len(self.entries)%3] and self.meter.stage==STAGES[len(self.entries)//3], 'fixed5 native forwards/order/stage, no retry')
            e=dict(index=len(self.entries),resource=kind,stage=self.meter.stage,completed=False,inputs=[])
            self.entries.append(e);self.pending[module]=e
            self.store.event(dict(kind='module_attempt',entry=deepcopy(e)))
            for n,x in enumerate(args):
                a=self.array(x)
                if kind=='action_encoder':require(a.shape==(1,3,10) and self.meter.expected_input is not None and
                    np.array_equal(a[0],self.meter.expected_input['action_history']),'exact full3 actual observed/raw action tokens')
                elif kind=='predict':
                    require(a.shape[:2]==(1,3) and (n!=0 or (a.shape==(1,3,192) and
                        np.array_equal(a[0],self.meter.expected_input['latent_history']))),'exact full3 actual observed latent input')
                    if n==1:require(np.array_equal(a,self.action_embedding),'predictor receives actual full action embeddings')
                else:require(a.shape==(3,192) and self.predictor_output is not None and
                    np.array_equal(a,self.predictor_output.reshape(3,192)),'full3 unmodified predictor->pred_proj')
                e['inputs'].append(save_array(self.store,f'{self.stem}.module_{e["index"]:02d}.input_{n}.npy',a))
            self.store.event(dict(kind='module_inputs_saved',entry=deepcopy(e)))
        return enter
    def post(self,module,args,out):
        e=self.pending.get(module)
        if e is None:return
        if out is None:self.failed=True;return
        self.sync();a=self.array(out);kind=e['resource']
        if kind=='action_encoder':require(a.ndim==3 and a.shape[:2]==(1,3),'all3 action embeddings');self.action_embedding=a.copy()
        elif kind=='predict':require(a.shape==(1,3,192),'all3 predictor outputs');self.predictor_output=a.copy()
        else:require(a.shape==(3,192),'all3 projected outputs');self.meter.last_native=a[-1].astype(float).copy()
        e['output']=save_array(self.store,f'{self.stem}.module_{e["index"]:02d}.output.npy',a)
        e['completed']=True;self.pending.pop(module)
        self.store.event(dict(kind='module_return',entry=deepcopy(e)))
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in self.ORDER}
    def __enter__(self):
        def forbidden(*args):self.failed=True;raise ValueError('no new image encoder/projector calls')
        try:
            for m,k in ((self.model.action_encoder,'action_encoder'),(self.model.predictor,'predict'),(self.model.pred_proj,'pred_proj')):
                self.hooks.append(m.register_forward_pre_hook(self.pre(k)));self.hooks.append(m.register_forward_hook(self.post,always_call=True))
            for m in (self.model.encoder,self.model.projector):self.hooks.append(m.register_forward_pre_hook(forbidden))
            return self
        except BaseException as e:self.__exit__(type(e),e,e.__traceback__);raise
    def __exit__(self,typ,exc,tb):
        failures=[]
        for h in reversed(self.hooks):
            try:h.remove()
            except BaseException as e:failures.append(e)
        if failures:
            if exc is not None:exc.add_note('module cleanup: '+repr(failures))
            else:raise failures[0]


def evaluate(branch,backend,meter,store,row):
    """Backend-independent fixed H1 compiler; NumPy mocks are NOT neural evidence."""
    stem=f'branch_{row["index"]:02d}';context=branch['observed_context'];goal=np.asarray(branch['goal']);q=np.asarray(branch['q'])
    row['status']='attempted';row['arrays']={};row['evaluations']=[]
    for k,a in (('observed_latents',context['latent_history']),('observed_tokens',context['action_history']),
            ('observed_raw_history',branch['observed_raw_history']),('goal',branch['goal']),('prefix_commands',branch['prefix_commands']),('old_suffix',branch['old_suffix_commands'])):
        row['arrays'][k]=save_array(store,stem+'.'+k+'.npy',np.asarray(a,np.float32))
    def stage(name,raw):
        meter.stage=name;meter.last_native=None
        meter.expected_input=planner.incoming(dict(action_mean=branch['action_mean'],action_std=branch['action_std']),context,raw)
        store.event(dict(kind='stage_started',branch_id=branch['branch_id'],stage=name,commands=np.asarray(raw).astype(float).tolist(),expected_input=meter.expected_input))
    stage('nominal',np.zeros((5,2),np.float32));nominal=np.asarray(backend.nominal(deepcopy(context)))
    require(nominal.shape==(192,) and np.isfinite(nominal).all(),'full192 nominal')
    if meter.require_native:require(np.array_equal(nominal,meter.last_native),'nominal from actual native pred_proj')
    row['arrays']['nominal']=save_array(store,stem+'.nominal.npy',nominal)
    store.event(dict(kind='nominal_saved',receipt=deepcopy(row)))
    stage('jacobian',np.zeros((5,2),np.float32));jac=np.asarray(backend.jacobian(deepcopy(context),meter.call))
    require(jac.shape==(192,10) and np.isfinite(jac).all(),'original H1 192x10 Jacobian')
    if meter.require_native:require(np.array_equal(nominal,meter.last_native),'fresh AD nominal exactly equals independent forward')
    row['arrays']['jacobian']=save_array(store,stem+'.jacobian.npy',jac)
    store.event(dict(kind='jacobian_saved',receipt=deepcopy(row)))
    solve=d2p.with_svd_receipts(lambda:compiler.bounded_ridge(q@jac,q@(goal-nominal),damping=compiler.DAMPING,cap=branch['packet_logit_cap']),meter.call)
    row['solve']=solve;params=np.asarray(solve['params']);new=np.tanh(params.astype(np.float32)).reshape(5,2)
    row['arrays']['new_logits']=save_array(store,stem+'.new_logits.npy',params)
    row['arrays']['new_suffix']=save_array(store,stem+'.new_suffix.npy',new)
    store.event(dict(kind='new_suffix_frozen_before_evaluation',receipt=deepcopy(row)))
    outputs=[]
    for name,raw in (('old_suffix',np.asarray(branch['old_suffix_commands'],np.float32)),('new_suffix',new),('new_suffix_replay',new)):
        stage(name,raw);e=dict(stage=name,commands=raw.astype(float).tolist(),status='attempted');row['evaluations'].append(e)
        store.event(dict(kind='suffix_evaluation_attempt',receipt=deepcopy(e)))
        after=backend.step(deepcopy(context),raw);planner.advance(dict(action_mean=branch['action_mean'],action_std=branch['action_std']),context,raw,after)
        core.context(after);z=np.asarray(after['latent_history'][-1])
        if meter.require_native:require(np.array_equal(z,meter.last_native),'returned successor from actual native full module output')
        e.update(status='completed',context=after,cost=float(np.mean((q@(z-goal))**2)),
            latent_array=save_array(store,stem+'.'+name+'.context_latents.npy',np.asarray(after['latent_history'],np.float32)),
            token_array=save_array(store,stem+'.'+name+'.context_tokens.npy',np.asarray(after['action_history'],np.float32)))
        store.event(dict(kind='suffix_evaluation_return',receipt=deepcopy(e)));outputs.append(after)
    require(outputs[1]==outputs[2],'fresh new suffix exact independent full-context replay')
    row.update(status='suffix_compiled',new_suffix_commands=new.astype(float).tolist(),prefix_commands=deepcopy(branch['prefix_commands']),
        observed_context_sha256=core.digest(context),dispatch='new_suffix_unconditionally',equal_new_replay=True)
    return row


def check_result(branch,row):
    require(row['status']=='suffix_compiled' and row['branch_id']==branch['branch_id'] and row['equal_new_replay'] is True and
        row['dispatch']=='new_suffix_unconditionally' and [e['stage'] for e in row['evaluations']]==list(STAGES[2:]),'all three fixed evaluations, no dynamic selection')
    require(row['prefix_commands']==branch['prefix_commands'] and row['observed_context_sha256']==core.digest(branch['observed_context']),'exact actual observed context/prefix')
    params=np.asarray(row['solve']['params']);new=np.tanh(params.astype(np.float32)).reshape(5,2)
    require(params.shape==(10,) and np.linalg.norm(params)<=branch['packet_logit_cap']+1e-12 and
        row['new_suffix_commands']==new.astype(float).tolist() and row['solve']['radius']==branch['packet_logit_cap'],'original bounded ridge domain/raw mapping')
    expected=[branch['old_suffix_commands'],row['new_suffix_commands'],row['new_suffix_commands']]
    for e,raw in zip(row['evaluations'],expected,strict=True):
        require(e['status']=='completed' and e['commands']==raw,'old diagnostic/new/replay commands')
        planner.advance(dict(action_mean=branch['action_mean'],action_std=branch['action_std']),branch['observed_context'],raw,e['context'])
        z=np.asarray(e['context']['latent_history'][-1]);require(e['cost']==float(np.mean((np.asarray(branch['q'])@(z-np.asarray(branch['goal'])))**2)),'same original Q/goal score')
    require(row['evaluations'][1]['context']==row['evaluations'][2]['context'],'actual new suffix replay exact')


def saved_arrays(branch,row,load):
    """Pure release validation of actual module inputs/outputs; no neural calls."""
    a={k:load(v) for k,v in row['arrays'].items()}
    for k,v in (('observed_latents',branch['observed_context']['latent_history']),
            ('observed_tokens',branch['observed_context']['action_history']),
            ('observed_raw_history',branch['observed_raw_history']),('goal',branch['goal']),
            ('prefix_commands',branch['prefix_commands']),('old_suffix',branch['old_suffix_commands'])):
        require(np.array_equal(a[k],np.asarray(v,np.float32)),'saved literal input '+k)
    require(a['jacobian'].shape==(192,10) and a['nominal'].shape==(192,) and
        np.array_equal(a['new_logits'],row['solve']['params']) and
        np.array_equal(a['new_suffix'],row['new_suffix_commands']),'saved original Jacobian/solve/raw')
    entries=row['module_ledger'];require(len(entries)==15,'five complete full native forwards')
    raws=[np.zeros((5,2),np.float32)]*2+[branch['old_suffix_commands'],row['new_suffix_commands'],row['new_suffix_commands']]
    for n,(stage,raw) in enumerate(zip(STAGES,raws,strict=True)):
        group=entries[3*n:3*n+3];incoming=planner.incoming(branch,branch['observed_context'],raw)
        ins=[];outs=[]
        for j,(e,kind) in enumerate(zip(group,Modules.ORDER,strict=True)):
            require(e['index']==3*n+j and e['resource']==kind and e['stage']==stage and e['completed'] is True,
                'native complete order/stage')
            ins.append([load(v) for v in e['inputs']]);outs.append(load(e['output']))
        require(len(ins[0])==1 and ins[0][0].shape==(1,3,10) and
            np.array_equal(ins[0][0][0],incoming['action_history']) and outs[0].shape[:2]==(1,3), 'native exact three action tokens')
        require(len(ins[1])==2 and ins[1][0].shape==(1,3,192) and
            np.array_equal(ins[1][0][0],incoming['latent_history']) and np.array_equal(ins[1][1],outs[0]) and
            outs[1].shape==(1,3,192),'native observed prefix/actual action embeddings')
        require(len(ins[2])==1 and np.array_equal(ins[2][0],outs[1].reshape(3,192)) and outs[2].shape==(3,192),
            'native whole predictor/projector path')
        expected=a['nominal'] if n<2 else row['evaluations'][n-2]['context']['latent_history'][-1]
        require(np.array_equal(outs[2][-1],expected),'native actual returned successor')
        if n>=2:
            ev=row['evaluations'][n-2]
            require(np.array_equal(load(ev['latent_array']),ev['context']['latent_history']) and
                np.array_equal(load(ev['token_array']),ev['context']['action_history']),'literal full evaluated context arrays')
    require(row['math_ledger']==[dict(index=0,resource='jacobian',completed=True,stage='jacobian',shape=[192,10],mathematical_VJP=192),
        dict(index=1,resource='svd',completed=True,stage='jacobian')],'single ordered Jacobian/SVD')
    require(row['AD_API_ledger']==[dict(index=0,resource='jacobian',completed=True,is_grads_batched=False),
        dict(index=1,resource='grad',completed=True,is_grads_batched=True)],'actual vectorized AD engine census')


def compile_all(p,store,runner,rows):
    for branch,row in zip(p['branches'],rows,strict=True):
        require(row['branch_id']==branch['branch_id'],'four branch fixed order')
        row['status']='attempted';stem=f'branch_{row["index"]:02d}'
        entry=branch['prefix_npy'];_,raw=load_npy(entry['path'],entry)
        store._write(stem+'.original_prefix.native192.npy',raw)
        store.event(dict(kind='branch_started',branch=deepcopy(branch)))
        runner(branch,store,row)
        check_result(branch,row);blind.exact_counts(row['resources'],budget())
        blind.exact_counts(row['AD_API_resources'],dict(grad=1,jacobian=1))
        plan=dict(schema=NAME+'.model_plan',source_slot=branch['source_slot'],arm=branch['arm'],branch_id=branch['branch_id'],
            parent_task_id=branch['parent_task_id'],blind_sha256=branch['blind_sha256'],planner_selection_sha256=branch['planner_selection_sha256'],
            observed_context_sha256=core.digest(branch['observed_context']),prefix_commands=branch['prefix_commands'],
            suffix_commands=row['new_suffix_commands'],commands=branch['prefix_commands']+row['new_suffix_commands'],
            selection='new_suffix_unconditionally_old_diagnostic_only',P1_admission=False,P4_admission=False)
        plan['selection_sha256']=core.digest(plan);row['plan']=plan
        store.put(stem+'.result.json.gz',row);store.put(stem+'.plan.json',plan)
        store.event(dict(kind='branch_sealed',receipt=deepcopy(row)))


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(index=k,source_slot=b['source_slot'],arm=b['arm'],branch_id=b['branch_id'],status='not_started') for k,b in enumerate(p['branches'])]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        observed.enc.frozen(model);before=observed.enc.tensor_hash(model);require(before==p['expected_model_hash'],'same original frozen model')
        def runner(branch,store,row):
            sync=lambda:torch.cuda.synchronize(device)
            meter=Meter(store.event,sync);meter.require_native=True
            modules=Modules(model,store,f'branch_{row["index"]:02d}',meter,sync)
            ad=SuffixAD(torch,blind.ARMS[1],store.event,sync);ad.stage=lambda:meter.stage
            backend=d2p.TorchBackend(model,device)
            try:
                with ad,modules:evaluate(branch,backend,meter,store,row)
            finally:
                row['resources']={**modules.counts(),**meter.counts()};row['AD_API_resources']=ad.counts()
                row['module_ledger']=deepcopy(modules.entries);row['math_ledger']=deepcopy(meter.entries)
                row['AD_API_ledger']=deepcopy(ad.events)
        with sdpa_kernel(SDPBackend.MATH):compile_all(p,store,runner,rows)
    except BaseException as e:
        errors.append(observed.enc.error('execution',e))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(observed.enc.error('error_journal',e))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:observed.enc.frozen(model);after=observed.enc.tensor_hash(model)
            except BaseException as e:errors.append(observed.enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(observed.enc.error('final_source',e))
        try:store.close()
        except BaseException as e:errors.append(observed.enc.error('journal_finalize',e))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='suffix_compiled' and 'plan' in r for r in rows)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:errors.append(observed.enc.error('artifact_listing',e));complete=False;files=[]
        for f in files:
            try:require(f.is_file(),'flat regular output');hashes[f.name]=sha(f)
            except BaseException as e:errors.append(observed.enc.error('artifact_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='observed_suffix_calibration_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,first_error=errors[0] if errors else None,
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            commands_released=complete,contract=p['contract'],host=host,new_physics=0,new_images=0,training=False,P1_admission=False,P4_admission=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(observed.enc.error('final_report',e));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial preserved, no retry: '+str(errors));return report


def consume(output,report_sha):
    core.hash_value(report_sha);output=Path(output).absolute();r=read(output/'report.json',report_sha)
    require(r['schema']==NAME+'.report' and r['status']=='observed_suffix_calibration_sealed_not_admitted' and r['commands_released'] is True and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True and
        r['P1_admission'] is False and r['P4_admission'] is False,'complete own command release only')
    p=verify(r['prepared'],r['prepare_sha256']);require(p['contract']==r['contract'] and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'same frozen model/protocol')
    require(blind.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'own output/prepare binding')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'exact final artifact census')
    for n,h in r['artifact_sha256'].items():require(sha(observed.rgb.old.local(output,n))==h,'sealed artifact')
    journal,eof=blind.inflate((output/'events.jsonl.gz').read_bytes(),CAP*16)
    require(eof and journal.endswith(b'\n'),'complete durable own journal')
    events=[blind.decode(line) for line in journal.splitlines()]
    starts=[e['branch'] for e in events if e['kind']=='branch_started']
    sealed=[e['receipt'] for e in events if e['kind']=='branch_sealed']
    require(starts==p['branches'] and sealed==r['rows'],'all four journaled branches/selections')
    def load(entry):
        path=observed.rgb.old.local(output,entry['file'])
        require(entry['sha256']==r['artifact_sha256'][entry['file']],'array in report artifact closure')
        stream=BytesIO(path.read_bytes());a=np.load(stream,allow_pickle=False)
        require(stream.tell()==path.stat().st_size and str(a.dtype)==entry['dtype'] and list(a.shape)==entry['shape'] and
            a.dtype in (np.dtype('float32'),np.dtype('float64')) and np.isfinite(a).all(),'literal finite original saved array')
        return a
    require(len(r['rows'])==4,'all four branches');plans=[]
    for index,(b,row) in enumerate(zip(p['branches'],r['rows'],strict=True)):
        require(row['index']==index and row['source_slot']==b['source_slot'] and row['arm']==b['arm'],'fixed branch identity/order')
        check_result(b,row);blind.exact_counts(row['resources'],budget());blind.exact_counts(row['AD_API_resources'],dict(grad=1,jacobian=1))
        stem=f'branch_{row["index"]:02d}'
        require(blind.read(output/(stem+'.result.json.gz'))==row and blind.read(output/(stem+'.plan.json'))==row['plan'] and
            sha(output/(stem+'.original_prefix.native192.npy'))==b['prefix_npy']['sha256'],'original prefix bytes and own result/plan release')
        saved_arrays(b,row,load)
        left=next(k for k,e in enumerate(events) if e['kind']=='branch_started' and e['branch']['branch_id']==b['branch_id'])
        right=next(k for k,e in enumerate(events) if e['kind']=='branch_sealed' and e['receipt']['branch_id']==b['branch_id'])
        scoped=events[left:right+1]
        for attempt,returned,ledger in (('module_attempt','module_return',row['module_ledger']),
                ('math_attempt','math_return',row['math_ledger']),('AD_API_attempt','AD_API_completed',row['AD_API_ledger'])):
            initial=[dict(e,completed=False) for e in ledger]
            if attempt=='module_attempt':initial=[{k:v for k,v in dict(e,inputs=[]).items() if k!='output'} for e in initial]
            returns=[e['entry'] for e in scoped if e['kind']==returned]
            require(returns==(list(reversed(ledger)) if attempt=='AD_API_attempt' else ledger) and
                [e['entry'] for e in scoped if e['kind']==attempt]==initial,'actual attempt/return journal matches ledger')
        plan=deepcopy(row['plan']);ident=plan.pop('selection_sha256')
        require(plan['schema']==NAME+'.model_plan' and plan['source_slot']==b['source_slot'] and plan['arm']==b['arm'] and
            plan['parent_task_id']==b['parent_task_id'] and plan['blind_sha256']==b['blind_sha256'] and plan['planner_selection_sha256']==b['planner_selection_sha256'] and
            plan['observed_context_sha256']==core.digest(b['observed_context']) and plan['P1_admission'] is False and plan['P4_admission'] is False and
            plan['selection']=='new_suffix_unconditionally_old_diagnostic_only' and
            core.digest(plan)==ident and plan['branch_id']==b['branch_id'] and plan['prefix_commands']==b['prefix_commands'] and
            plan['suffix_commands']==row['new_suffix_commands'] and plan['commands']==b['prefix_commands']+row['new_suffix_commands'], 'own full raw prefix/new suffix seal')
        plans.append(dict(plan,selection_sha256=ident,whole_run_report_sha256=report_sha))
    require(sha(output/'report.json')==report_sha,'final report TOCTOU')
    return dict(schema=NAME+'.consume',status='observed_suffix_commands_verified_not_admitted',errors=[],
        plans=plans,new_physics=0,P1_admission=False,P4_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('rgb-report','rgb-sha','rgb-source-sha','prepared','prepare-sha','output','report-sha'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output,rgb_report=a.rgb_report,rgb_sha=a.rgb_sha,rgb_source_sha=a.rgb_source_sha)
    elif a.command=='verify':r=dict(status='verified_no_execution',branches=len(verify(a.prepared,a.prepare_sha)['branches']))
    elif a.command=='consume':r=consume(a.output,a.report_sha)
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
