"""Eight fixed I/F trajectories x cp20/25/30, original HOST CUDA encode only.

MAIN receipts are inherited trust anchors, not recursively recertified.
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
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_physical_20261029_v1 as producer
from experiments.spectral_reachability import agent_p3_same_time_cp10_rgb_bridge_20261029_v1 as primitive
from experiments.spectral_reachability import agent_p2_two_program_full30_rgb_encode_20261029_v1 as codec

bridge,physical=codec.bridge,codec.physical
require,sha,raw=producer.require,producer.sha,primitive.raw
ROOT=producer.ROOT
NAME='agent_p2_full30_feedback_rgb_20261029_v1'
AUTH='MAIN_FULL30_FEEDBACK24_RGB_HOST_CUDA_ONLY'
RECORDED='all24_IF_actual_RGB_sealed_NOT_admitted'
CP=(20,25,30)
CAP,FINAL,METADATA,FLOOR,WALL=64*1024**2,8*1024**2,16*1024**2,200*1024**2,600
META=16*1024**2
SHAPES=deepcopy(primitive.SHAPES)
FLAGS={k:False for k in ('P1_admission','P2_admission','P2_mechanism','C3','scientific_admission',
    'scientific_pass','physical_release','fresh','equal_FLOPs','equal_total_compute','training')}
RECEIPT_SHAS=[
 '565c432e08a7965079a10c709d90773edab4a967004ec07e43202b88a6c96d6a',
 'fc08ef91da4f158218d78a17da640a2664a8d1e63ebdf6e3b6fe6b6df5c2ee5e',
 '94157f79fd3ca2ea28e825914e56a9f364ba4f87e7871b6d09d9c9eeaa7b9c38',
 '8a66e6bb41c716c4d0b01b405745a7d356fcd4cfc7b6e17178e4084c27199f2d']
PINS={f'job{i}':dict(path=str(ROOT/'results'/(producer.NAME+f'_consume_J{i}_MAIN01.json')),sha256=h)
      for i,h in enumerate(RECEIPT_SHAS)}
PINS.update(joined=dict(path=str(ROOT/'results'/(producer.NAME+'_joined_MAIN01.json')),
 sha256='2842de482d30ff5898dc52915f58ddca292d0c4958f84198377d8c852c9ae101'),
 O_receipt=dict(path=str(ROOT/'results'/(codec.NAME+'_consume_MAIN01.json')),
 sha256='42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166'),
 O_report=dict(path=str(ROOT/'results'/(codec.NAME+'_cuda_MAIN01/report.json')),
 sha256='915b6760db8bf30abe16e4bfba347604e6359c17d08c1ff3f2a7a076b7714df4'))
CODE_PINS={str(Path(producer.__file__).resolve()):'ce57d99c85b03e1365b55c429e39cd129458a5ed01be22b232ce92309a17b844',
 str(Path(primitive.__file__).resolve()):'40d51e4d5ad5fce5db9417fecc639e0288f286b678a5e7b0eb4e22adea392b77',
 str(Path(codec.__file__).resolve()):'3d1f0821bf138699f28a7e49eaea78b5dacb54930eb07a430cd7d2e4166314a4'}
MODEL_PINS=deepcopy(producer.PINS)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def read(path):
    return producer.read(producer.native.absolute(str(path)),META)


def pinned(ref):
    producer.native.actual_hash(ref['sha256'])
    require(sha(ref['path'])==ref['sha256'],'actual pinned saved authority')
    return read(ref['path'])


def shape_census():
    n=[int(np.prod(s))*np.dtype(d).itemsize for s,d in SHAPES]
    upper=lambda x:x+(x>>12)+(x>>14)+(x>>25)+31
    gz=24*sum(upper(x+128) for x in n)
    return dict(images=24,arrays=168,per_image_array_bytes=n,raw_stored_array_bytes=24*sum(n),
        NPY_header_bytes=24*7*128,gzip_NPY_upper_bytes=gz,
        native_module_IO_bytes=24*sum(n[2:6]),metadata_cap_bytes=METADATA,
        final_reserve_bytes=FINAL,total_upper_bytes=gz+METADATA+FINAL,output_cap_bytes=CAP,
        metadata_scope='ALL repeated JSON: initial roster, event records, seal, report, completion, failures',
        shapes=[[list(s),d] for s,d in SHAPES])


def contract():
    return dict(schema=NAME+'.contract',fixed_denominator=24,trajectory_denominator=8,base_denominator=4,
        checkpoints=list(CP),absolute_steps=[30,35,40],known_roots=2,fresh_roots=0,batch=1,
        max_calls={k:24 if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        AD=0,new_physics=0,new_predict=0,new_optimizer=0,new_task=0,no_dedup=True,no_reselection=True,
        inherited='original four O and cp<=15 encoding; NEVER reuse O suffix latent for I/F actual frames',
        native_hidden=[1,257,192],latent=192,normalization='original image transform exactly once; no latent renormalization',
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH_requested_and_flag_guarded',
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        wall_seconds=WALL,output_cap_bytes=CAP,metadata_cap_bytes=METADATA,final_reserve_bytes=FINAL,
        disk_floor_bytes=FLOOR,full_upstream_rebuilt_here=False,total_FLOPs=None,shape_census=shape_census(),**FLAGS)


def model_metadata():
    policy=bridge.rgb.POLICY
    return str(policy),bridge.rgb.config_shapes(bridge.rgb.read(policy/'config.json')),deepcopy(bridge.rgb.MODELS)


def source_files():
    return own_files()+bridge.source_files()+[Path(codec.__file__),Path(primitive.__file__),Path(producer.__file__)]


def fixed_rows(prep,receipt,output,base):
    """Pure original-plan/time/frame join; no Q or outcome-dependent selection."""
    require(type(base) is int and base in range(4) and prep['job_index']==receipt['job_index']==base,
        'fixed job identity')
    producer.validate_execution(prep)  # Pure plan contracts only, NOT old saved strict.
    require(receipt['schema']==producer.NAME+'.consume' and receipt['status']==producer.RECORDED
        and receipt['errors']==[] and receipt['own_native_saved_reconstructed_here'] is True
        and receipt['fixed_denominator']==2 and receipt['global_policy_slots']==12 and
        receipt['plans']==prep['plans'] and len(receipt['rows'])==2 and
        all(receipt[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','physical_release')),
        'actual complete own physical authority/all2 slots')
    out=[]
    for i,(plan,row) in enumerate(zip(prep['plans'],receipt['rows'],strict=True)):
        require(row['index']==i and row['plan']==plan and row['policy']==('I','F')[i]
            and row['base_index']==base and row['root_slot']==base//2 and row['condition']=='off'
            and row['selection_digest']==plan['selection_digest'] and row['task_id']==plan['task_id']
            and row['status'] in ('horizon_completed','native_stopped'),'fixed source row/task/arm')
        require(type(row['own_returned']) is int and 0<=row['own_returned']<=30
            and row['warmup_returned']==10 and row['initial_exact'] is True,'literal source eventtime')
        if row['own_returned']>15:
            require(row['feedback_guard']==producer.feedback_guard_value(plan,prep),'exact accepted prefix guard')
        commands=producer.raw30(plan['commands']);w=plan['new_model_plan']
        require(commands.tobytes().hex()==w['raw_float32_bits'],'new I/F raw witness not original O')
        require(row['missing_checkpoints']==[s for s in producer.CP if s not in [c['step'] for c in row['checkpoints']]],
            'source missing time, never padding')
        for cp in CP:
            checkpoints=[x for x in row['checkpoints'] if x['step']==cp]
            frames=[x for x in row['trajectory'] if x['phase']=='plan' and x['step']==cp and x['overlay_index']==0]
            present=cp<=row['own_returned']
            require(len(checkpoints)==len(frames)==int(present),'unique same-time off checkpoint')
            source=None
            if present:
                f=frames[0];checkpoint=checkpoints[0]
                require(checkpoint=={k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}
                    and f['condition']=='off' and f['state']==f['snapshot']['state']
                    and Path(f['png']).name==f['png'],'actual I/F source frame/native identity')
                response=[x for x in row['responses'] if x['step']==cp]
                require(len(response)==1 and response[0]['state']==f['state'],'same-time native return')
                source=dict(checkpoint,path=str(Path(output)/checkpoint['png']))
            history=commands[cp-10:cp]
            out.append(dict(index=base*6+i*3+CP.index(cp),base_index=base,policy=plan['policy'],root_slot=base//2,
                arm=plan['arm'],condition='off',checkpoint=cp,absolute_step=10+cp,
                task_id=plan['task_id'],selection_digest=plan['selection_digest'],
                new_witness_sha256=w['new_witness_sha256'],parent_witness_sha256=w['parent_witness_sha256'],
                model_hash=prep['expected_model_hash'],goal192=deepcopy(w['goal']),
                raw30_float32_bits=commands.tobytes().hex(),causal_raw10_bits=history.tobytes().hex(),
                normalized_action_history=producer.normalization.normalized_history(history).tolist(),
                source=source,source_job=deepcopy(PINS[f'job{base}']),source_branch=i,
                physical_status=row['status'],actual_returned=row['own_returned'],
                status='not_started' if present else 'missing_physical_checkpoint',latent=None))
    return out


def build():
    require(shape_census()['total_upper_bytes']<CAP,'fixed transport proof fits original64MiB')
    producer.rehash(CODE_PINS)
    values={k:pinned(v) for k,v in PINS.items()}
    joined=values['joined']
    require(joined['schema']==producer.NAME+'.joined_jobs' and joined['fixed_denominator']==12
        and joined['inherited_O']==4 and joined['new_trajectories']==8 and
        joined['job_authorities']==[PINS[f'job{i}'] for i in range(4)],'whole fixed4 job join authorities')
    origin=values['O_receipt'];oreport=values['O_report']
    require(origin['schema']==codec.NAME+'.consume' and origin['status']=='all28_full30_RGB_verified_NOT_admitted'
        and origin['report_sha256']==PINS['O_report']['sha256'] and len(origin['rows'])==28
        and oreport['schema']==codec.NAME+'.report' and oreport['status']=='terminal_encoded_science_blocked',
        'original O encoder actual report/strict inheritance only')
    c=pinned(MODEL_PINS['prepared'])
    bindings=deepcopy(c['bindings_sha256']);rows=[];inherited_counts=[]
    producer.native.merge(bindings,{v['path']:v['sha256'] for v in list(PINS.values())+list(MODEL_PINS.values())})
    producer.native.merge(bindings,CODE_PINS)
    for base in range(4):
        u=values[f'job{base}'];b=u['bindings']
        prep_paths=[path for path,h in b.items() if h==u['prepare_sha256']]
        report_paths=[path for path,h in b.items() if h==u['physical_report_sha256']]
        require(len(prep_paths)==len(report_paths)==1,'unique actual saved prepare/report binding')
        pp,rp=prep_paths[0],report_paths[0];p=read(pp);r=read(rp);output=Path(rp).parent
        require(sha(pp)==u['prepare_sha256'] and sha(rp)==u['physical_report_sha256']
            and r['execution_binding']==dict(prepared=pp,prepare_sha256=u['prepare_sha256'],
                source_report_sha256=p['source_report_sha256'],planner_authority=p['planner_authority'])
            and r['rows']==u['rows'] and r['counts']==u['counts'] and r['errors']==u['errors']==[]
            and r['schema']==producer.NAME+'.report' and r['status']==u['status']==producer.RECORDED,
            'physical prepare/report/receipt same source')
        co=output/'completion.json';completion=read(co)
        require(sha(co)==u['completion_sha256'] and completion==u['completion']
            and completion['report_sha256']==u['physical_report_sha256'] and completion['journal_sha256']==r['journal_sha256']
            and completion['status']==producer.RECORDED and type(completion['wall_seconds']) in (int,float)
            and np.isfinite(completion['wall_seconds']) and 0<=completion['wall_seconds']<producer.WALL,
            'inherited physical completion authority')
        require({'report.json','completion.json','journal.jsonl','execution_binding.json'}<=
            {f.name for f in output.iterdir()} and
            {str(f) for f in output.iterdir()}=={path for path in b if Path(path).parent==output}
            and all(f.is_file() and not f.is_symlink() for f in output.iterdir()),'complete source inventory bound')
        producer.native.merge(bindings,b);producer.native.merge(bindings,p['bindings'])
        require(p['expected_model_hash']==c['expected_model_hash'] and p['arguments']==producer.PINS,
            'original model authority not replaced')
        rows.extend(fixed_rows(p,u,output,base));inherited_counts.append(deepcopy(u['counts']))
    policy,shapes,models=model_metadata()
    require(Path(policy).resolve()==Path(c['policy']).resolve() and c['expected_model_hash']==bridge.rgb.MODEL_TENSOR_SHA,
        'original checkpoint/encoder/normalizer identity')
    producer.native.merge(bindings,models)
    for f in source_files():
        producer.native.merge(bindings,{str(f.resolve()):sha(f)})
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    producer.rehash(bindings)
    require(len(rows)==24 and len(raw(rows))<METADATA,'whole immutable24 row metadata bounded')
    n=sum(row['source'] is not None for row in rows)
    return dict(schema=NAME+'.prepare',contract=contract(),authorities=deepcopy(PINS),
        model_authorities=deepcopy(MODEL_PINS),rows=rows,bindings=bindings,policy=policy,shapes=shapes,
        expected_model_hash=c['expected_model_hash'],
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        inherited_physical_counts=inherited_counts,inherited_encoding=dict(authorities={k:deepcopy(PINS[k]) for k in ('O_receipt','O_report')},
            original_O=4,IF_prefix_controls=[0,5,10,15],basis='physical exact-prefix guards; original O encoding inherited, NOT rerun',
            O_suffix_latents_reused_for_IF=False),
        full_upstream_rebuilt_here=False,upstream_trust='MAIN pinned complete receipts + all saved byte bindings, no fullstrict rerun')


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
    require(p['schema']==NAME+'.prepare' and raw(p)==raw(build()) and sha(path)==expected,
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
            require(not final and self.arrays<168 and len(value)<=max_array_file_bound(),'168 full arrays, no reserve borrowing')
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
    def __init__(self,*args,**kw):super().__init__(6,*args,**kw)
    def dispatch(self,kind,tensor,callback,*args,**kw):
        require(len(self.entries)<48 and self.image==len(self.entries)//2,'exact24 image guard/no retries')
        return super().dispatch(kind,tensor,callback,*args,**kw)


def process(p,rows,store,encode,check=lambda:None):
    require(rows==p['rows'] and len(rows)==24,'fixed whole24 immutable roster')
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
        guard=Ledger(lambda e:store.event(dict(kind='guard_event',event=e)),
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
        store.event(dict(kind='all24_native_sealed',sha256=h,encoded=sum(x['status']=='encoded' for x in rows)),final=True)
        report=dict(schema=NAME+'.report',status=RECORDED if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,precision_before=numeric_before,precision_after=numeric_after,
            model_inventory_before=classes_before,model_inventory_after=classes_after,CUDA_peak_bytes=peak,
            known_roots=2,fixed_denominator=24,full_upstream_rebuilt_here=False,**FLAGS)
        try:publication=publish(store,report,budget.started)
        except BaseException as e:
            store.put('terminal_failure.json',dict(stage='publication',error=repr(e)),final=True)
            raise
    require(complete,'partial retained/no retries '+str(errors))
    return publication


def validate(p,r,events,load):
    require(p['contract']==r['contract']==contract() and len(p['rows'])==len(r['rows'])==24
        and r['errors']==r['blocked']==[],'whole24 clean encoding')
    modules=[];guards=[];want=[];ordinal=0;results=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')}==
            {k:v for k,v in spec.items() if k not in ('status','latent')},'literal immutable source/time/raw identity')
        if spec['source'] is None:
            require(row==spec,'missing checkpoint remains null without substitute')
            results.append(dict(spec,actual192=None));continue
        require(row['status']=='encoded','every available original image encoded once')
        pix=load(row['pixels']);bridge.same(pix,bridge.load_pixels(spec['source']),'actual I/F image, never O replacement')
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
    want.append(dict(kind='all24_native_sealed',sha256=r['native_seal_sha256'],encoded=ordinal))
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
    require(len(paths)<=414 and all(p.is_file() and not p.is_symlink() for p in paths)
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
    require(r['schema']==NAME+'.report' and r['status']==RECORDED and r['fixed_denominator']==24
        and r['known_roots']==2 and r['full_upstream_rebuilt_here'] is False
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
    producer.rehash(bindings)
    require({f.name:sha(f) for f in output.iterdir()}=={name:hashlib.sha256(b).hexdigest() for name,b in files.items()},
        'unchanged final file census')
    return dict(schema=NAME+'.consume',status='all_available24_IF_RGB_verified_NOT_admitted',
        rows=rows,report_sha256=expected,completion_sha256=completion_sha,prepare_sha256=r['prepare_sha256'],
        fixed_denominator=24,trajectory_denominator=8,known_roots=2,fresh_roots=0,
        encoded=n,missing=24-n,counts=r['counts'],
        sample_frame_counts={k:dict(API=n,samples=n,frames=n) for k in ('encoder','projector')},
        actual_native_module_IO_bytes=io_bytes,byte_census=census,shape_census=shape_census(),bindings=bindings,
        inherited_encoding=p['inherited_encoding'],inherited_physical_counts=p['inherited_physical_counts'],
        source_authorities=p['authorities'],completion=completion,full_upstream_rebuilt_here=False,
        own_native_encoding_saved_reconstructed_here=True,total_FLOPs=None,
        Q_analysis_here=False,consume_new_neural_calls=0,consume_new_physics=0,**FLAGS)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);sub=q.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    for name in ('verify','run','consume'):
        p=sub.add_parser(name)
        if name!='consume':p.add_argument('--prepared',required=True);p.add_argument('--prepare-sha',required=True)
        if name in ('run','consume'):p.add_argument('--output',required=True)
        if name=='run':p.add_argument('--authorize-host-cuda',required=True,action='store_true')
        if name=='consume':
            p.add_argument('--report-sha',required=True);p.add_argument('--completion-sha',required=True)
            p.add_argument('--receipt',required=True)
    a=q.parse_args(argv)
    if a.command=='prepare':v=write(a.output,build())
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(verified=True,expected_calls=p['expected_calls'])
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:v=write(a.receipt,consume(a.output,a.report_sha,a.completion_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()
