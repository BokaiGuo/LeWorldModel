"""Two CEM raw15 trajectories x cp0/5/10/15, original HOST CUDA encode only.

Actual physical authorities supplied externally; no future SHA constants.
No new task, prediction, AD, physics, training, outcome ranking or admission.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_lewm_bounded_cem_h3_physical_20261029_v1 as producer
from experiments.spectral_reachability import agent_p3_same_time_cp10_rgb_bridge_20261029_v1 as primitive
from experiments.spectral_reachability import agent_p2_two_program_full30_rgb_encode_20261029_v1 as codec

bridge,physical=codec.bridge,codec.physical
require,sha,raw=producer.require,producer.sha,primitive.raw
ROOT=producer.ROOT
NAME='agent_lewm_bounded_cem_h3_rgb_20261029_v1'
AUTH='MAIN_BOUNDED_CEM_H3_RGB8_HOST_CUDA_ONLY'
RECORDED='all8_CEM_H3_actual_RGB_sealed_NOT_admitted'
CP=(0,5,10,15)
CAP,FINAL,METADATA,FLOOR,WALL=64*1024**2,8*1024**2,16*1024**2,200*1024**2,600
META=16*1024**2
SHAPES=deepcopy(primitive.SHAPES)
FLAGS={k:False for k in ('P1_admission','P2_admission','P2_mechanism','C3','scientific_admission',
    'scientific_pass','physical_release','fresh','equal_FLOPs','equal_total_compute','training','fair_ranking','official_reproduction')}
def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def read(path):
    return producer.read(producer.native.absolute(str(path)),META)


def shape_census(images=8):
    require(type(images) is int and 0<=images<=8,'actual available images inside fixed8 denominator')
    n=[int(np.prod(s))*np.dtype(d).itemsize for s,d in SHAPES]
    upper=lambda x:x+(x>>12)+(x>>14)+(x>>25)+31
    gz=images*sum(upper(x+128) for x in n)
    return dict(images=images,fixed_denominator=8,arrays=images*7,per_image_array_bytes=n,raw_stored_array_bytes=images*sum(n),
        NPY_header_bytes=images*7*128,gzip_NPY_upper_bytes=gz,
        native_module_IO_bytes=images*sum(n[2:6]),metadata_cap_bytes=METADATA,
        final_reserve_bytes=FINAL,total_upper_bytes=gz+METADATA+FINAL,output_cap_bytes=CAP,
        metadata_scope='ALL repeated JSON: initial roster, event records, seal, report, completion, failures',
        shapes=[[list(s),d] for s,d in SHAPES])


def contract():
    return dict(schema=NAME+'.contract',fixed_denominator=8,trajectory_denominator=2,
        checkpoints=list(CP),absolute_steps=[10,15,20,25],known_roots=2,fresh_roots=0,batch=1,
        max_calls={k:8 if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        AD=0,new_physics=0,new_predict=0,new_optimizer=0,new_task=0,no_dedup=True,no_reselection=True,
        inherited='original CEM H3 saved actualmean predictions, goal and warm latents; no extra prediction/goal encode',
        native_hidden=[1,257,192],latent=192,normalization='original image transform exactly once; no latent renormalization',
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH_requested_and_flag_guarded',
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        wall_seconds=WALL,output_cap_bytes=CAP,metadata_cap_bytes=METADATA,final_reserve_bytes=FINAL,
        disk_floor_bytes=FLOOR,physical_consume_rebuilt=False,CEM_numerical_fullstrict_rebuilt=False,total_FLOPs=None,shape_census=shape_census(),**FLAGS)


def model_metadata():
    policy=bridge.rgb.POLICY
    return str(policy),bridge.rgb.config_shapes(bridge.rgb.read(policy/'config.json')),deepcopy(bridge.rgb.MODELS)



PHYSICAL_SOURCE_SHA='717c400db322c0789c2ea4a7dd68dfc0499224703e9b8bb87b3b1153657a0610'
ARGUMENTS={'physical_prepared','physical_prepare_sha','physical_output','physical_report_sha',
    'physical_completion_sha','physical_receipt','physical_receipt_sha'}
ACTUAL=dict(physical_prepared=str(ROOT/'results'/(producer.NAME+'_prepare_MAIN01.json')),
    physical_prepare_sha='d17ffc349dedd7d310764f8cb6d7c4c615c9d70bc2e85ee85cd3a5991db431a8',
    physical_output=str(ROOT/'results'/(producer.NAME+'_physics_MAIN01')),
    physical_report_sha='8705d1877a37d81b835b04935aadaf42af4a78a4e1fb4b9524fadb2f516f59d1',
    physical_completion_sha='c9310caa342d89637e7f49cd58c3258e2279bb75c027af0ad345183dfe81a63a',
    physical_receipt=str(ROOT/'results'/(producer.NAME+'_consume_MAIN01.json')),
    physical_receipt_sha='1e69a2fe082af3c5da32ec28ac8ba10b2375b5a593fa0bf69a60367d7cbc9b86')


def source_files():
    return own_files()+bridge.source_files()+[Path(codec.__file__),Path(primitive.__file__),Path(producer.__file__)]


def rehash(bindings):
    bridge.verify_bindings(bindings)


def normalized_history(slot,commands):
    a=np.asarray(commands)
    require(a.dtype==np.float32 and a.shape==(10,2) and np.isfinite(a).all() and np.all(np.abs(a)<=1),
        'literal raw10 actual/planned history')
    mean=np.asarray(slot['mean'],np.float32);std=np.asarray(slot['std'],np.float32)
    require(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),
        'original task engineering stats')
    # Exactly the original P2 subtract then divide, once, time-major5x2.
    x=np.subtract(a.reshape(2,10),np.tile(mean,5),dtype=np.float32)
    return np.divide(x,np.tile(std,5),dtype=np.float32)


def fixed_rows(prep,strict,modelprep,futures,output):
    producer.validate_execution(prep)
    require(strict['schema']==producer.NAME+'.consume' and strict['status']==producer.RECORDED
        and strict['fixed_denominator']==strict['known_roots']==2 and strict['fresh_roots']==0
        and strict['errors']==[] and strict['plans']==prep['plans'] and len(strict['rows'])==2
        and strict['planner_authority']==prep['planner_authority']
        and all(strict[k] is False for k in ('P1_admission','P2_admission','scientific_admission','physical_release','fair_ranking')),
        'complete actual two-root CEM native receipt, no old plan schema')
    require(modelprep['schema']==producer.PRODUCER+'.prepare' and modelprep['slots']==prep['model_slots']
        and modelprep['expected_model_hash']==prep['expected_model_hash'] and len(futures)==2,
        'original CEM model/normalizer/history authority')
    rows=[]
    for i,(plan,actual,slot,future) in enumerate(zip(prep['plans'],strict['rows'],modelprep['slots'],futures,strict=True)):
        key=producer.ORDER[i]
        require(type(actual['index']) is type(actual['root_slot']) is int
            and (actual['index'],actual['root_slot'],actual['condition'],actual['arm'])==(i,*key)
            and actual['plan']==plan and actual['selection_digest']==plan['selection_digest']
            and actual['task_id']==plan['task_id']==slot['task_id'] and plan['task_sha256']==slot['task_sha256']
            and actual['status'] in ('horizon_completed','native_stopped')
            and type(actual['own_returned']) is int and 0<=actual['own_returned']<=15
            and (actual['initial_exact'] is True or actual['own_returned']==0),
            'fixed original source/physical two-root whole roster')
        commands=producer.raw15(plan['commands']);context=slot['packet']['root_context']
        emb=np.asarray(context['latent_history'],np.float32);goal=bridge.latent(slot['packet']['goal'])
        require(emb.shape==(3,192) and np.isfinite(emb).all() and slot['q']==np.eye(192).tolist()
            and future.dtype==np.float32 and future.shape==(1,1,6,192) and np.isfinite(future).all()
            and future[0,0,:3].tobytes()==emb.tobytes()
            and np.asarray(slot['emb'],np.float32).tobytes()==emb.tobytes()
            and np.asarray(slot['goal_emb'],np.float32).tobytes()==goal.tobytes(),
            'paid actualmean whole6 frames/original context/goal/Q')
        warm=np.asarray(slot['warm_raw10'],np.float32)
        require(warm.tobytes()==producer.moving.warmup_commands().tobytes()
            and normalized_history(slot,warm).tobytes()==np.asarray(context['action_history'],np.float32).tobytes(),
            'root actual warm10 tokens from original normalizer exactly once')
        require(actual['missing_checkpoints']==[cp for cp in CP if cp not in [c['step'] for c in actual['checkpoints']]],
            'exact all four missing-checkpoint slots')
        for j,step in enumerate(CP):
            cps=[x for x in actual['checkpoints'] if x['step']==step]
            frames=[x for x in actual['trajectory'] if x['phase']=='plan' and x['step']==step and x['overlay_index']==0]
            responses=[x for x in actual['responses'] if x['step']==step]
            present=actual['initial_exact'] is True and step<=actual['own_returned']
            require(len(cps)==len(frames)==int(present) and len(responses)==int(present and step>0)
                and (step in actual['missing_checkpoints']) is (not present),
                'unique same-time actual off checkpoint, no last-frame padding')
            src=metrics=response=None
            if present:
                f=frames[0]
                require(cps[0]=={k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}
                    and f['condition']=='off' and f['state']==f['snapshot']['state']
                    and Path(f['png']).name==f['png'],'actual PNG/full7 snapshot same time')
                if step==0:
                    require(f['state']==actual['root_state']==prep['source_roots'][i]['root_state']
                        and actual['root_sha256']==plan['native_root_sha256'],'actual cp0 root, not warmup or forecast')
                else:
                    response=responses[0];require(response['state']==f['state'],'native transition same time')
                metrics=producer.native.metrics(plan['native_goal'],f['state'])
                if response is not None:
                    require(response['native_metrics']==metrics['native'] and response['reward']==metrics['native7_reward']
                        and response['native_success'] is response['terminated'] is metrics['native_success'],
                        'unchanged native goal labels, not RGB-Q success')
                src=dict(cps[0],path=str(Path(output)/cps[0]['png']))
            history=np.concatenate((warm,commands[:step]))[-10:]
            pred=None if step==0 else future[0,0,j+2].copy()
            rows.append(dict(index=len(rows),plan_index=i,root_slot=i,condition='off',arm=key[2],
                checkpoint=step,absolute_step=10+step,task_id=plan['task_id'],task_sha256=plan['task_sha256'],
                selection_digest=plan['selection_digest'],model_hash=prep['expected_model_hash'],
                root_context=deepcopy(context),native_goal=deepcopy(plan['native_goal']),goal192=goal.tolist(),
                Q_sha256=producer.digest(slot['q']),raw15_float32_bits=commands.tobytes().hex(),
                planned_raw10=history.tolist(),planned_raw10_bits=history.tobytes().hex(),
                planned_normalized_history=normalized_history(slot,history).tolist(),
                causal_raw10_bits=history.tobytes().hex() if present else None,
                actual_normalized_history=normalized_history(slot,history).tolist() if present else None,
                predicted192=None if pred is None else pred.tolist(),
                observed_anchor192=emb[-1].tolist() if step==0 else None,
                model_Q=None if pred is None else bridge.mse(pred,goal),
                source=src,native_metrics=metrics,response=response,
                physical_status=actual['status'],actual_returned=actual['own_returned'],
                missing_reason=None if present else 'native_stop_before_checkpoint',
                status='not_started' if present else 'missing_physical_checkpoint',latent=None))
    require(len(rows)==8,'fixed two roots x cp0/5/10/15; no outcome filtering')
    return rows


def load_cem_future(report,prep,root):
    row=report['rows'][root];slot=prep['model_slots'][root]
    require(type(row['root']) is int and row['root']==root
        and all(row[k]==slot[k] for k in ('task_id','task_sha256')),'original paid CEM root/task')
    d=row['delivery'];rec=d['layouts']['whole']['predicted_emb'];name=rec['file']
    require(d['root']==root and d['status']=='paid_finalmean_evaluated'
        and d['layouts']['whole']['attempted'] is d['layouts']['whole']['returned'] is True
        and Path(name).name==name and rec['dtype']=='<f4' and rec['shape']==[1,1,6,192] and rec['nbytes']==4608
        and report['artifacts'][name]=={k:rec[k] for k in ('file_bytes','sha256')},
        'actual paid whole6 NPY descriptor, no new forecast')
    f=producer.PLANNER_OUTPUT/name
    require(prep['bindings'][str(f)]==sha(f)==rec['sha256'] and f.stat().st_size==rec['file_bytes'],
        'paid whole6 NPY file authority inside full physical prepare closure')
    with f.open('rb') as stream:
        a=np.load(stream,allow_pickle=False);require(not stream.read(),'no future NPY trailing payload')
    require(a.dtype==np.float32 and a.shape==(1,1,6,192) and np.isfinite(a).all(),'literal full paid future f32')
    return a


def build(**arguments):
    require(arguments==ACTUAL and set(arguments)==ARGUMENTS,'only MAIN completed actual four authorities; no root filtering')
    for k,v in arguments.items():
        if k.endswith('_sha'):producer.native.actual_hash(v)
        else:producer.native.absolute(v)
    require(sha(producer.__file__)==PHYSICAL_SOURCE_SHA,'frozen physical source before saved reconstruction')
    output=Path(arguments['physical_output']);rp=output/'report.json'
    require(sha(rp)==arguments['physical_report_sha'] and sha(output/'completion.json')==arguments['physical_completion_sha']
        and sha(arguments['physical_prepared'])==arguments['physical_prepare_sha']
        and sha(arguments['physical_receipt'])==arguments['physical_receipt_sha'],'actual four-authority SHA bindings')
    # MAIN already ran the complete native consumer: inherit its entire persisted
    # receipt and SHA closure. Never run producer.consume/verify here again.
    strict=read(arguments['physical_receipt'])
    report=read(rp);p=read(arguments['physical_prepared'])
    require(report['schema']==producer.NAME+'.report' and report['status']==strict['status']==producer.RECORDED
        and report['errors']==strict['errors']==[] and report['inputs_final_verified'] is True
        and report['execution_binding']['prepared']==arguments['physical_prepared']
        and report['execution_binding']['prepare_sha256']==arguments['physical_prepare_sha']
        and report['rows']==strict['rows'] and report['counts']==strict['counts']
        and strict['physical_report_sha256']==arguments['physical_report_sha']
        and strict['completion_sha256']==arguments['physical_completion_sha']
        and report['execution_binding']['planner_authority']==p['planner_authority'],
        'exact physical prepare/report/completion/whole receipt identity')
    completion=read(output/'completion.json')
    require(completion==strict['completion'] and completion['schema']==producer.NAME+'.completion'
        and completion['status']=='published_within_cap' and completion['report_sha256']==arguments['physical_report_sha']
        and completion['prepared']==arguments['physical_prepared'] and completion['prepare_sha256']==arguments['physical_prepare_sha']
        and 0<=report['wall_seconds']<=completion['inclusive_wall_seconds']<producer.WALL
        and set(completion['artifact_sha256'])|{'completion.json'}=={f.name for f in output.iterdir()}
        and all(strict['bindings'][str(output/name)]==h for name,h in completion['artifact_sha256'].items()),
        'MAIN full receipt completion/inclusive wall/full artifact roster')
    require({str(f) for f in output.iterdir()}=={name for name in strict['bindings'] if Path(name).parent==output}
        and all(f.is_file() and not f.is_symlink() for f in output.iterdir())
        and strict['bindings'][arguments['physical_prepared']]==arguments['physical_prepare_sha']
        and strict['bindings'][str(output/'completion.json')]==arguments['physical_completion_sha'],
        'whole MAIN source inventory and prepare/completion byte authority')
    mr=read(producer.PLANNER_OUTPUT/'report.json');modelprep=read(producer.PLANNER_PREP)
    require(p['bindings'][str(producer.PLANNER_PREP)]==sha(producer.PLANNER_PREP)==producer.PREP_SHA
        and p['bindings'][str(producer.PLANNER_OUTPUT/'report.json')]==sha(producer.PLANNER_OUTPUT/'report.json')==producer.REPORT_SHA,
        'original CEM actual model report/preparation closure')
    futures=[load_cem_future(mr,p,i) for i in (0,1)]
    rows=fixed_rows(p,strict,modelprep,futures,output)
    bindings=deepcopy(strict['bindings'])
    producer.native.merge(bindings,{arguments['physical_prepared']:arguments['physical_prepare_sha'],
        str(rp):arguments['physical_report_sha'],str(output/'completion.json'):arguments['physical_completion_sha'],
        arguments['physical_receipt']:arguments['physical_receipt_sha']})
    producer.native.merge(bindings,{str(Path(f).resolve()):sha(f) for f in source_files()})
    policy,shapes,models=model_metadata()
    require(Path(policy).resolve()==Path(modelprep['policy']).resolve()
        and shapes['hidden']==[1,257,192] and shapes['projected']==[1,192]
        and modelprep['expected_model_hash']==bridge.rgb.MODEL_TENSOR_SHA,
        'original frozen model/transform/257tokens/192CLS, not Hi training scaler')
    producer.native.merge(bindings,models)
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    rehash(bindings);n=sum(r['source'] is not None for r in rows)
    require(shape_census()['total_upper_bytes']<CAP and len(raw(rows))<METADATA,'full8 actual native transport proof')
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(arguments),rows=rows,bindings=bindings,
        policy=policy,shapes=shapes,expected_model_hash=p['expected_model_hash'],
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},actual_shape_census=shape_census(n),
        inherited_physical_counts=deepcopy(strict['counts']),inherited_model_resources=deepcopy(p['planner_authority']['resources']),
        inherited_CEM=deepcopy(p['planner_authority']),inherited_model_wall_seconds=mr['elapsed_seconds'],
        physical_report_sha256=arguments['physical_report_sha'],physical_receipt_sha256=arguments['physical_receipt_sha'],
        physical_completion_sha256=arguments['physical_completion_sha'],
        physical_consume_rebuilt=False,CEM_numerical_fullstrict_rebuilt=False,
        upstream_trust='MAIN full saved receipt + whole rows/accounting/publication/artifact SHA closure; no upstream strict rerun')



def score(spec,row):
    value=row['actual192'];z=None if value is None else bridge.latent(value)
    pred=spec['predicted192'];goal=spec['goal192']
    actual=None if z is None else bridge.mse(z,goal)
    model=spec['model_Q']
    # Diagnostic only: original root re-encode can differ across job/batch kernels.
    anchor=spec['observed_anchor192'];anchor_check=None
    if anchor is not None and z is not None:
        a=bridge.latent(anchor)
        def ordered(x):
            u=x.view(np.uint32).astype(np.int64)
            return np.where(u>=0x80000000,0x100000000-u,u+0x80000000)
        anchor_check=dict(bitwise=a.tobytes()==z.tobytes(),maxabs=float(np.max(np.abs(a.astype(np.float64)-z.astype(np.float64)))),
            max_ULP=int(np.max(np.abs(ordered(a)-ordered(z)))),MSE=bridge.mse(a,z),parity_upgraded=False)
    return dict(index=spec['index'],root_slot=spec['root_slot'],checkpoint=spec['checkpoint'],absolute_step=spec['absolute_step'],
        task_id=spec['task_id'],role='observed_anchor_reencode' if spec['checkpoint']==0 else 'paid_CEM_same_raw_H3_prediction',
        model_Q=model,actual_encoded_Q=actual,prediction_vs_encoded_MSE=None if pred is None or z is None else bridge.mse(pred,z),
        model_minus_actual_Q=None if model is None or actual is None else model-actual,
        anchor_reencode=anchor_check,latent_missing=z is None,native_metrics=spec['native_metrics'],
        native_success=None if spec['native_metrics'] is None else spec['native_metrics']['native_success'],
        missing_reason=spec['missing_reason'],scientific_pass=False,physical_benefit_claim=False,fair_ranking=False)


def write(path,value):
    path=Path(producer.native.absolute(str(path)));body=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(body)<=META
        and shutil.disk_usage(path.parent).free>=FLOOR+len(body),'own exclusive bounded metadata')
    with path.open('xb') as f:
        require(f.write(body)==len(body),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent)
    return dict(path=str(path),sha256=sha(path))


def verify(path,expected):
    producer.native.actual_hash(expected);path=producer.native.absolute(str(path))
    require(sha(path)==expected,'external own prepare SHA');p=read(path)
    require(p['schema']==NAME+'.prepare' and raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,
        'complete own pinned saved prepare reconstruction')
    return p


class Budget(primitive.Budget):
    def __init__(self,clock=None,free=None):
        super().__init__(clock=clock or time.monotonic,free=free or (lambda:shutil.disk_usage(ROOT).free))


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        path=Path(producer.native.absolute(str(output)))
        require(path.name.startswith(NAME+'_') and shutil.disk_usage(path.parent).free>=CAP+FLOOR,'own full64MiB space')
        super().__init__(path,budget);self.index=0;self.arrays=0;self.metadata_bytes=0
    def write(self,name,value,final=False):
        require(Path(name).name==name and name not in ('','.','..') and type(value) is bytes,'flat byte artifacts')
        if name.endswith('.npy.gz'):
            require(not final and self.arrays<56 and len(value)<=max_array_file_bound(),'56 full arrays, no reserve borrowing')
        else:
            require(name.endswith('.json') and value==raw(producer.native.decode(value)),'canonical all metadata')
            require(self.metadata_bytes+len(value)<=METADATA,'aggregate16MiB ALL metadata copies')
        h=super().write(name,value,final)
        if name.endswith('.npy.gz'):self.arrays+=1
        else:self.metadata_bytes+=len(value)
        return h
    def put(self,name,value,final=False):return self.write(name,raw(value),final)
    def event(self,e,final=False):
        self.put(f'event_{self.index:04}.json',e,final);self.index+=1
    # Policy-free lossless codec only, NOT the old cp10 task constructor.
    def array(self,name,a):return primitive.Store.array(self,name,a)


def max_array_file_bound():
    n=max(shape_census()['per_image_array_bytes'])+128
    return n+(n>>12)+(n>>14)+(n>>25)+31


class Ledger(bridge.native.Ledger):
    def __init__(self,images,*args,**kw):
        require(type(images) is int and 0<=images<=8,'exact available-image guard budget')
        self.images=images;super().__init__((images+3)//4,*args,**kw)
    def dispatch(self,kind,tensor,callback,*args,**kw):
        require(len(self.entries)<2*self.images and self.image==len(self.entries)//2,'exact available image guard/no retries')
        return super().dispatch(kind,tensor,callback,*args,**kw)


def process(p,rows,store,encode,check=lambda:None):
    require(rows==p['rows'] and len(rows)==8,'fixed whole8 immutable roster')
    ordinal=0
    for row in rows:
        if row['source'] is None:continue
        check();scope=dict(row=row['index'],step=row['checkpoint'],phase='encode');row['status']='attempted'
        store.event(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        pixels=bridge.load_pixels(row['source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pixels)
        z=encode(pixels,row,scope,ordinal);bridge.latent(z)
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)));ordinal+=1


precision_state=primitive.precision_state
precision_contract=codec.precision_contract
model_inventory=primitive.model_inventory


def publish(store,r,started,clock=None):
    clock=clock or time.monotonic
    store.put('report.json',r,final=True)
    hashes={p.name:sha(p) for p in store.path.iterdir()}
    elapsed=clock()-started
    require(type(elapsed) in (int,float) and np.isfinite(elapsed) and 0<=elapsed<=WALL,'publication whole wall')
    completion=dict(schema=NAME+'.completion',report_sha256=hashes['report.json'],
        prepared=r['prepared'],prepare_sha256=r['prepare_sha256'],artifact_sha256=hashes,
        inclusive_wall_seconds=elapsed,boundary='verify start through durable report and all artifact hashes',**FLAGS)
    store.put('completion.json',completion,final=True)
    if clock()-started>WALL:
        store.put('terminal_failure.json',dict(stage='completion_wall'),final=True)
        raise ValueError('completion publication exceeded cap')
    return dict(output=str(store.path),status=r['status'],report_sha256=hashes['report.json'],
        completion_sha256=sha(store.path/'completion.json'))

def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA before I/O')
    budget=Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);precision=None;errors=[];trace=guard=None;model=None;before=after=None;source_ok=False
    numeric_before=numeric_after=None;classes_before=classes_after=None;peak=None
    def check():
        store.check()
        require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and
            torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'actual CUDA memory caps')
        require(precision_state(torch)==precision_contract(),'actual TF32/autocast/MATH contract')
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats(device)
        precision=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model')
        classes_before=model_inventory(model,p['bindings'])
        trace=bridge.Trace(model,torch,store,check,p['expected_calls'])
        guard=Ledger(p['expected_calls']['encoder'],lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pixels,row,scope,ordinal):
            guard.image=ordinal
            with physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pixels,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pixels),'original preprocessing exactly')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'native192 only')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,bridge.native.guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            numeric_before=precision_state(torch);check()
            process(p,rows,store,encode,check)
            torch.cuda.synchronize(device);check();numeric_after=precision_state(torch)
            peak=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
    except BaseException as e:errors.append(physical.kernel.error('encode',e))
    finally:
        if precision is not None:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=precision
        try:
            if model is not None:
                bridge.native.frozen(model);after=bridge.native.tensor_hash(model);classes_after=model_inventory(model,p['bindings'])
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash']
            and classes_before==classes_after and numeric_before==numeric_after==precision_contract()
            and peak is not None and all(v<=contract()[k] for v,k in zip(peak,('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)) and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        # native seal precedes any analysis of the new encoded outputs.
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        store.event(dict(kind='all8_native_sealed',sha256=h,encoded=sum(x['status']=='encoded' for x in rows)),final=True)
        report=dict(schema=NAME+'.report',status=RECORDED if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,precision_before=numeric_before,precision_after=numeric_after,
            model_inventory_before=classes_before,model_inventory_after=classes_after,CUDA_peak_bytes=peak,
            known_roots=2,fixed_denominator=8,physical_consume_rebuilt=False,CEM_numerical_fullstrict_rebuilt=False,**FLAGS)
        try:publication=publish(store,report,budget.started)
        except BaseException as e:
            store.put('terminal_failure.json',dict(stage='publication',error=repr(e)),final=True)
            raise
    require(complete,'partial retained/no retries '+str(errors))
    return publication


def validate(p,r,events,load):
    require(p['contract']==r['contract']==contract() and len(p['rows'])==len(r['rows'])==8
        and r['errors']==r['blocked']==[],'whole8 clean encoding')
    modules=[];guards=[];want=[];ordinal=0;results=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')}==
            {k:v for k,v in spec.items() if k not in ('status','latent')},'literal immutable source/time/raw identity')
        if spec['source'] is None:
            require(row==spec,'missing checkpoint remains null without substitute')
            results.append(dict(spec,actual192=None));continue
        require(row['status']=='encoded','every available original image encoded once')
        pix=load(row['pixels']);bridge.same(pix,bridge.load_pixels(spec['source']),'actual CEM H3 image, never O replacement')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'original image transform once')
        z=load(row['latent']);bridge.latent(z)
        scope=dict(row=spec['index'],step=spec['checkpoint'],phase='encode')
        mods=row['modules']
        require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector']
            and all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),
            'only encoder/projector complete actual scoped native IO')
        en,proj=mods
        buffers=[pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z]
        require(all(x.dtype==np.float32 and np.isfinite(x).all() for x in buffers),'finite f32 raw native tensors')
        physical.saved.overlay.visual_bridge(*buffers,[1,257,192])
        bridge.same(buffers[1],pre.reshape(1,3,224,224),'preprocess to actual encoder bytes')
        bridge.same(buffers[3],buffers[2][:,0],'original full hidden CLS slice to projector')
        bridge.same(z,buffers[4][0],'projector output to original192, no renormalize')
        want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in mods:
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            a={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=a),
                dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),
                dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1
        results.append(dict(row,actual192=z.tolist()))
    want.append(dict(kind='all8_native_sealed',sha256=r['native_seal_sha256'],encoded=ordinal))
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(2*ordinal))
        and r['guards']==guards and events==want and r['counts']==bridge.counts(modules)=={
            k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},
        'ALL event chronology, forbidden API and paid counters; no filtered events')
    return results


def byte_census(files):
    arrays=sum(len(v) for n,v in files.items() if n.endswith('.npy.gz'))
    meta=sum(len(v) for n,v in files.items() if n.endswith('.json'))
    require(arrays<=shape_census()['gzip_NPY_upper_bytes'] and meta<=METADATA
        and arrays+meta==sum(map(len,files.values()))<=CAP,'full actual arrays/aggregate metadata/cap')
    return dict(array_file_bytes=arrays,all_metadata_bytes=meta,total_bytes=arrays+meta)


def consume(output,expected,completion_sha):
    output=Path(producer.native.absolute(str(output)))
    producer.native.actual_hash(expected);producer.native.actual_hash(completion_sha)
    require(output.is_dir() and not output.is_symlink(),'regular output directory')
    paths=list(output.iterdir())
    require(len(paths)<=142 and all(p.is_file() and not p.is_symlink() for p in paths)
        and sum(p.stat().st_size for p in paths)<=CAP,'bounded all output files before read')
    files={p.name:p.read_bytes() for p in paths};census=byte_census(files)
    require('terminal_failure.json' not in files and hashlib.sha256(files['report.json']).hexdigest()==expected
        and hashlib.sha256(files['completion.json']).hexdigest()==completion_sha,'external hashes/no failed publication')
    for name,body in files.items():
        if name.endswith('.json'):require(body==raw(producer.native.decode(body)),'canonical all JSON records')
        else:require(name.endswith('.npy.gz') and len(body)<=max_array_file_bound(),'only bounded exact array formats')
    decode=lambda n:producer.native.decode(files[n])
    r=decode('report.json');completion=decode('completion.json')
    require(set(completion)=={'schema','report_sha256','prepared','prepare_sha256','artifact_sha256',
        'inclusive_wall_seconds','boundary'}|set(FLAGS) and completion['schema']==NAME+'.completion'
        and completion['report_sha256']==expected and completion['prepared']==r['prepared']
        and completion['prepare_sha256']==r['prepare_sha256']
        and completion['boundary']=='verify start through durable report and all artifact hashes'
        and type(r['elapsed_seconds']) in (int,float) and type(completion['inclusive_wall_seconds']) in (int,float)
        and np.isfinite(r['elapsed_seconds']) and np.isfinite(completion['inclusive_wall_seconds'])
        and 0<=r['elapsed_seconds']<=completion['inclusive_wall_seconds']<=WALL
        and completion['artifact_sha256']=={n:hashlib.sha256(b).hexdigest() for n,b in files.items() if n!='completion.json'}
        and all(completion[k] is False for k in FLAGS),'completion-bound inclusive publication wall/census')
    p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']==RECORDED and r['fixed_denominator']==8
        and r['known_roots']==2 and r['physical_consume_rebuilt'] is False and r['CEM_numerical_fullstrict_rebuilt'] is False
        and all(r[k] is False for k in FLAGS) and r['source_verified'] is r['model_unchanged'] is True
        and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'complete original frozen model, no science')
    require(r['precision_before']==r['precision_after']==precision_contract()
        and r['model_inventory_before']==r['model_inventory_after'] and r['model_inventory_before'],
        'observed precision/class inventory before and after, not kernel-level parity proof')
    for e in r['model_inventory_before']:require(p['bindings'][e['path']]==e['sha256'],'bound actual model class source')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2 and
        all(type(v) is int and 0<=v<=contract()[k] for v,k in zip(r['CUDA_peak_bytes'],
            ('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'actual GPU axes')
    require(set(files)==set(r['artifact_sha256'])|{'report.json','completion.json'}
        and all(hashlib.sha256(files[n]).hexdigest()==h for n,h in r['artifact_sha256'].items()),
        'full artifact census/no orphan outputs')
    n=sum(x['source'] is not None for x in p['rows'])
    require(type(r['event_count']) is int and r['event_count']==10*n+1,'all image/native/seal events')
    metadata={'binding.json','initial_roster.json','native_seal.json'}|{f'event_{i:04}.json' for i in range(r['event_count'])}
    require({n for n in r['artifact_sha256'] if not n.endswith('.npy.gz')}==metadata,'no filtered metadata')
    require(decode('binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'])
        and decode('initial_roster.json')==p['rows'],'original binding/roster')
    require(hashlib.sha256(files['native_seal.json']).hexdigest()==r['native_seal_sha256']
        and decode('native_seal.json')=={k:r[k] for k in ('rows','modules','guards','blocked','errors')},
        'all encoding sealed before saved analysis/labels')
    used=set()
    def load(ref):
        used.add(ref['file']);return codec.load_array(ref,lambda n:files[n])
    rows=validate(p,r,[decode(f'event_{i:04}.json') for i in range(r['event_count'])],load)
    require(used=={name for name in files if name.endswith('.npy.gz')} and len(used)==7*n,'all7 native arrays per paid image')
    io_bytes=sum(load(e).nbytes for m in r['modules'] for e in m['inputs']+[m['output']])
    require(io_bytes==n*sum(shape_census()['per_image_array_bytes'][2:6]),'independent native input/output byte axis')
    bindings=deepcopy(p['bindings']);producer.native.merge(bindings,{r['prepared']:r['prepare_sha256']})
    producer.native.merge(bindings,{str(output/name):hashlib.sha256(b).hexdigest() for name,b in files.items()})
    rehash(bindings)
    require({f.name:sha(f) for f in output.iterdir()}=={name:hashlib.sha256(b).hexdigest() for name,b in files.items()},
        'unchanged final file census')
    return dict(schema=NAME+'.consume',status='all_available8_CEM_H3_RGB_verified_NOT_admitted',
        rows=rows,report_sha256=expected,completion_sha256=completion_sha,prepare_sha256=r['prepare_sha256'],
        fixed_denominator=8,trajectory_denominator=2,known_roots=2,fresh_roots=0,
        encoded=n,missing=8-n,counts=r['counts'],actual_shape_census=shape_census(n),
        sample_frame_counts={k:dict(API=n,samples=n,frames=n) for k in ('encoder','projector')},
        actual_native_module_IO_bytes=io_bytes,byte_census=census,shape_census=shape_census(),bindings=bindings,
        inherited_CEM=p['inherited_CEM'],inherited_physical_counts=p['inherited_physical_counts'],
        inherited_model_resources=p['inherited_model_resources'],inherited_model_wall_seconds=p['inherited_model_wall_seconds'],
        source_authorities=p['arguments'],completion=completion,full_upstream_rebuilt_here=False,
        own_native_encoding_saved_reconstructed_here=True,total_FLOPs=None,
        Q_analysis_here=True,score_scope='saved identity192 MSE only, native task success separate',
        comparisons=[score(spec,row) for spec,row in zip(p['rows'],rows,strict=True)],consume_new_neural_calls=0,consume_new_physics=0,**FLAGS)



def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);sub=q.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    for k in sorted(ARGUMENTS):a.add_argument('--'+k.replace('_','-'),required=True)
    for cmd in ('verify','run','consume'):
        a=sub.add_parser(cmd)
        if cmd!='consume':a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd in ('run','consume'):a.add_argument('--output',required=True)
        if cmd=='run':a.add_argument('--authorize-host-cuda',required=True,action='store_true')
        if cmd=='consume':
            a.add_argument('--report-sha',required=True);a.add_argument('--completion-sha',required=True);a.add_argument('--receipt',required=True)
    a=q.parse_args(argv)
    if a.command=='prepare':v=write(a.output,build(**{k:getattr(a,k) for k in ARGUMENTS}))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(verified=True,expected_calls=p['expected_calls'])
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:v=write(a.receipt,consume(a.output,a.report_sha,a.completion_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()

