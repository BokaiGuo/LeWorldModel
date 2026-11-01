"""Fixed ten U/Z2 cp15 RGB bridge; explicit future MAIN authorities, no physics.

Inherited physical authentication is joined, not rerun. CUDA imports are confined
to MAIN-authorized run. RAM fixtures are never HOST input certification.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p3_same_time_cp10_rgb_bridge_20261029_v1 as old
from experiments.spectral_reachability import agent_p3_same_time_capability_probe_physical_20261029_v2 as native

bridge,physical,planner=old.bridge,old.physical,old.planner
require,sha,raw=old.require,native.sha,old.raw
ROOT=old.ROOT
NAME='agent_p3_same_time_cp15_rgb_bridge_20261029_v1'
AUTH='MAIN_P3_TEN_CP15_RGB_HOST_CUDA_ONLY'
RECORDED='all10_cp15_slots_saved_RGB_bridge_NOT_admitted'
CAP,FINAL,FLOOR,WALL,META=64*1024**2,8*1024**2,200*1024**2,600,16*1024**2
NATIVE_SHA='f070b6afe4fe0ebba66e05d366f929467602b98805980f86f27e2344445a8344'
ORDER=[(root,role,probe) for root,role in old.ORDER for probe in ('U','Z2')]
FLAGS=deepcopy(native.FLAGS)
SHAPES=deepcopy(old.SHAPES)
ACTUAL_PINS=dict(physical_prepare='8fdecdc9fe5fcb1f00b03c30e9971566da03c2a523ccc652d86065c9b442e95f',
    physical_batch='b70022a70a32c0c5f743684276e42f9040e228ae8296bb6daeae5fb70ce28c1b',
    physical_receipt='4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1')


def actual_authorities():
    base=ROOT/'results';stem=native.NAME
    paths=dict(physical_prepare=base/(stem+'_prepare_MAIN01.json'),
        physical_batch=base/(stem+'_MAIN01')/'batch.json',physical_receipt=base/(stem+'_consume_MAIN01.json'))
    return {k:dict(path=str(v),sha256=ACTUAL_PINS[k]) for k,v in paths.items()}


def calls(n):return {k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS}


def census():
    sizes=[int(np.prod(s))*np.dtype(d).itemsize for s,d in SHAPES]
    # Generic deflate bound plus gzip wrapper overhead (deliberately redundant).
    def bound(n):return n+(n+7)//8+(n+63)//64+11+18
    arrays=10*sum(bound(n+128) for n in sizes)
    require(arrays+META+FINAL<CAP,'full70 arrays and every metadata copy fit64MiB')
    return dict(slots=10,maximum_arrays=70,array_shapes=[[s,d] for s,d in SHAPES],per_image_bytes=sizes,
        compressed_array_upper_bytes=arrays,total_upper_bytes=arrays+META+FINAL,
        output_cap_bytes=CAP,metadata_cap_bytes=META,final_reserve_bytes=FINAL)


def contract():
    return dict(schema=NAME+'.contract',order=[list(x) for x in ORDER],fixed_denominator=10,
        checkpoint=15,absolute_step=25,history_absolute_steps=[15,20,25],known_roots=2,fresh_roots=0,
        maximum_calls=calls(10),no_dedup=True,no_padding=True,no_goal_encode=True,no_history_reencode=True,
        new_physics=0,AD=0,Q_scoring=0,new_optimizer=0,raw_deployment=False,
        model='original native192 RGB224 encoder CLS projector, not Hi',hidden=[1,257,192],
        context='cp10 actual history last2 + new actual cp15; actual control6..15 normalize once',
        missing='fixed null slot; never encode an earlier terminal in place of cp15',
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH_requested_observed_flags_guard',
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        wall_seconds=WALL,disk_floor_bytes=FLOOR,shape_census=census(),total_FLOPs=None,**FLAGS)


def read(path):return native.read(path)


def descriptor(path,expected):
    p=Path(path)
    require(p.is_absolute() and p.is_file() and not p.is_symlink(),'explicit regular absolute authority')
    require(len(expected)==64 and all(c in '0123456789abcdef' for c in expected)
        and len(set(expected))>1 and sha(str(p))==expected,'actual external authority SHA, no placeholder')
    return dict(path=str(p),sha256=expected)


def fixed_rows(p,receipt,w,output):
    """Pure authenticated ABI join; no score or native outcome chooses availability."""
    require(p['schema']==native.NAME+'.prepare' and p['fixture_only'] is False and p['stageB_connected'] is True
        and len(p['jobs'])==10 and len(w['rows'])==5,'real stageC prepare and full5 witnesses')
    require(receipt['schema']==native.NAME+'.consume' and receipt['fixed_denominator']==10
        and receipt['status'] in ('all10_saved_native_jobs_reconstructed_NOT_admitted','saved_partial_transport_NOT_verified')
        and receipt['source_authorities']==p['authority']
        and receipt['inherited_stageB_resource_receipt']==p['inherited_stageB_resource_receipt']
        and all(receipt[k] is False for k in FLAGS),'entire actual MAIN StageC receipt')
    require(len(receipt['jobs'])==10,'all ten receipt rows, no subset')
    rows=[]
    for i,(job,entry) in enumerate(zip(p['jobs'],receipt['jobs'],strict=True)):
        root,role,probe=ORDER[i];r=w['rows'][i//2]
        require((job['index'],job['endpoint'],job['root_slot'],job['source_role'],job['probe'])==(i,i//2,root,role,probe)
            and (r['root_slot'],r['role'])==(root,role) and job['source_witness_sha256']==p['authority']['full_witness']['sha256'],
            'fixed ten source endpoint/arm identities')
        require(entry['output']==str(Path(output)/f'job{i:02}'),'same actual job directory')
        task=r['stageA_task']['task'];old.controls.bridge.typed.core.validate_task(task)
        require(task['binding']['model_sha256']==bridge.rgb.MODEL_TENSOR_SHA
            and np.array_equal(task['projection'],np.eye(192)),'original native192 identity task')
        commands=np.asarray(job['raw15'],dtype='<f4')
        native.bits(job['raw15'],15)
        require(commands[:10].tobytes()==np.asarray(r['control_raw10'],dtype='<f4').tobytes(),
            'unchanged causal first10 before probe')
        frame=response=None;actual=entry.get('row');reason=None
        if actual is None:
            require(entry['schema']==native.NAME+'.failure_consume'
                and entry['status']=='transport_failed_NOT_verified' and entry['endpoint15'] is None
                and entry['counts'] is None,'unverified transport remains missing')
            reason='physical_transport_unverified'
        else:
            require(entry['schema']==native.NAME+'.job_consume' and entry['fixture_only'] is False
                and actual['job']==job and actual['fixture_only'] is False,'full actual job identity')
            endpoint=actual['endpoint15']
            selected=[e['receipt'] for e in actual['observations'] if e['receipt']['phase']=='plan'
                and e['receipt']['step']==15 and e['receipt']['overlay_index']==0]
            cp=[x for x in actual['checkpoints'] if x['step']==15]
            responses=[x for x in actual['responses'] if x['step']==15]
            present=endpoint is not None
            require(len(cp)==len(selected)==len(responses)==int(present)
                and (15 in actual['missing_checkpoints']) is (not present),'exact cp15 missing/no padding')
            if present:
                require(actual['status'] in ('horizon_completed','native_stopped') and actual['guards']==[51,66]
                    and len(actual['observations'])==81
                    and actual['counts']=={k:dict(attempted=v,completed=v) for k,v in native.COUNTS.items()},
                    'all66 guarded actual horizon before encoding cp15')
                require(cp[0]==selected[0]==endpoint and endpoint['condition']=='off'
                    and endpoint['state']==endpoint['snapshot']['state']
                    and responses[0]['state']==endpoint['state'],'same cp15 RGB/snapshot/native response')
                frame=deepcopy(endpoint);require(Path(frame['png']).name==frame['png'],'flat original native PNG')
                frame['path']=str(Path(output)/f'job{i:02}'/frame['png']);response=deepcopy(responses[0])
            else:reason='physical_cp15_missing'
        incoming=commands[-10:].copy()
        tokens=old.controls.bridge.typed.tokens(incoming,np.asarray(task['action_mean'],np.float32),
            np.asarray(task['action_std'],np.float32)).reshape(2,10)
        rows.append(dict(index=i,root_slot=root,role=role,probe=probe,endpoint=i//2,
            checkpoint=15,absolute_step=25,history_absolute_steps=[15,20,25],source=frame,
            physical_receipt_entry_sha256=old.digest(entry),physical_status=None if actual is None else actual['status'],
            physical_counts=None if actual is None else deepcopy(actual['counts']),
            physical_response=response,missing_reason=reason,
            raw15=deepcopy(job['raw15']),raw15_bits=commands.tobytes().hex(),
            causal_raw10=incoming.tolist(),causal_raw10_bits=incoming.tobytes().hex(),
            normalized_action_history=tokens.tolist(),normalized_float32_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
            cp10_task=deepcopy(task),cp10_identity=deepcopy(r['stageA_task']['identity']),
            selection_digest=job['selection_digest'],unknown_semantics=deepcopy(job['unknown_semantics']),
            status='not_started' if frame is not None else 'missing_physical_cp15',latent=None))
    return rows


def job_authority(p,pp,i,seal,entry,report,completion):
    require(seal['report_sha256']==entry['report_sha256'] and seal['completion_sha256']==entry['completion_sha256'],
        'full physical report/completion seals')
    binding=report['binding']
    require(set(report)=={'schema','binding','row','journal_tail','journal_sha256','elapsed_seconds'}
        and set(completion)=={'schema','report_sha256','artifact_sha256','bytes_before_completion','inclusive_wall_seconds','wall_scope'}
        and report['schema']==native.NAME+'.report' and report['row']==entry['row']
        and binding['prepared']==pp['path'] and binding['prepare_sha256']==pp['sha256']
        and binding['job']==p['jobs'][i] and binding['fixture_only'] is False
        and binding['authority']==p['authority'],'whole report row and authority identity')
    require(completion['schema']==native.NAME+'.completion' and completion['report_sha256']==entry['report_sha256']
        and completion['artifact_sha256']['journal.jsonl']==report['journal_sha256']
        and 0<=report['elapsed_seconds']<=completion['inclusive_wall_seconds']<=native.WALL
        and completion['wall_scope']=='through report hash; completion self-write excluded','physical completion/journal join')


def build(authorities):
    require(set(authorities)=={'physical_prepare','physical_batch','physical_receipt'},'three explicit actual MAIN authorities')
    require(authorities==actual_authorities(),'only sealed actual MAIN01 ten-slot authority, no substitution')
    d={k:descriptor(v['path'],v['sha256']) for k,v in authorities.items()}
    p,b,r=(read(d[k]['path']) for k in ('physical_prepare','physical_batch','physical_receipt'))
    require(sha(native.__file__)==NATIVE_SHA,'frozen StageC producer source')
    pp=d['physical_prepare'];output=Path(d['physical_batch']['path']).parent
    require(Path(d['physical_batch']['path']).name=='batch.json' and b['schema']==native.NAME+'.batch'
        and b['prepared']==r['prepared']==pp['path'] and b['prepare_sha256']==r['prepare_sha256']==pp['sha256']
        and len(b['jobs'])==10,'whole batch/receipt/prepare join')
    bindings=deepcopy(p['bindings'])
    def bind(path,h):
        path=str(path);require(path not in bindings or bindings[path]==h,'no source authority collision')
        require(sha(path)==h,'all authenticated source bytes');bindings[path]=h
    for v in d.values():bind(v['path'],v['sha256'])
    fw=p['authority']['full_witness'];bind(fw['path'],fw['sha256']);w=read(fw['path'])
    rows=fixed_rows(p,r,w,str(output))
    for i,(seal,entry) in enumerate(zip(b['jobs'],r['jobs'],strict=True)):
        directory=output/f'job{i:02}'
        require(seal['output']==entry['output']==str(directory),'literal own branch')
        if entry.get('status')=='transport_failed_NOT_verified':
            require(seal['failure_sha256']==entry['failure_sha256'],'failure bytes exact join')
            bind(directory/'failure.json',entry['failure_sha256'])
            continue
        bind(directory/'report.json',entry['report_sha256']);bind(directory/'completion.json',entry['completion_sha256'])
        report=read(directory/'report.json');completion=read(directory/'completion.json')
        job_authority(p,pp,i,seal,entry,report,completion)
        for name,h in completion['artifact_sha256'].items():
            require(Path(name).name==name,'flat original physical artifacts');bind(directory/name,h)
        if rows[i]['source'] is not None:bridge.load_pixels(rows[i]['source'])
    for f,h in bridge.rgb.MODELS.items():bind(f,h)
    own=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    for f in bridge.source_files()+own+[Path(old.__file__).resolve(),Path(native.__file__).resolve()]:bind(f,sha(str(f)))
    bridge.verify_bindings(bindings)
    return dict(schema=NAME+'.prepare',authorities=d,contract=contract(),rows=rows,bindings=bindings,
        expected_calls=calls(sum(x['source'] is not None for x in rows)),fixture_only=False,
        policy=str(bridge.rgb.POLICY),expected_model_hash=bridge.rgb.MODEL_TENSOR_SHA,
        upstream_fullstrict_reexecuted=False,upstream_trust='explicit MAIN whole saved receipt plus local exact joins',
        inherited_physical_counts=[x.get('row',{}).get('counts') for x in r['jobs']])


def write(path,value):
    p=Path(path);body=raw(value)
    require(p.is_absolute() and p.parent.is_dir() and not p.exists() and not p.is_symlink()
        and len(body)<=native.fw.CAP,'exclusive bounded own metadata')
    with p.open('xb') as f:require(f.write(body)==len(body),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(p.parent);return dict(path=str(p),sha256=sha(str(p)))


def verify(path,expected):
    descriptor(path,expected);p=read(path)
    require(p['fixture_only'] is False and raw(p)==raw(build(p['authorities'])) and sha(path)==expected,
        'entire own prepare reconstruction, no fixture input')
    return p


class Budget(bridge.rgb.Budget):
    def check(self,size=0,final=False):
        require(final or self.clock()-self.started<=WALL,'600s whole job')
        require(self.used+size<=CAP-(0 if final else FINAL) and self.free()>=FLOOR+size,'64MiB reserve/floor')


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        super().__init__(output,budget);self.index=0;self.arrays=0;self.metadata=0
    def write(self,name,value,final=False):
        require(Path(name).name==name,'flat own output')
        if name.endswith('.npy.gz'):require(self.arrays<70,'maximum70 arrays')
        else:require(name.endswith('.json') and self.metadata+len(value)<=META,'all serialized metadata copies quota')
        h=super().write(name,value,final)
        if name.endswith('.npy.gz'):self.arrays+=1
        else:self.metadata+=len(value)
        return h
    def put(self,name,value,final=False):return self.write(name,raw(value),final)
    def event(self,e,final=False):self.put(f'event_{self.index:04}.json',e,final);self.index+=1
    def array(self,name,a):
        a=np.ascontiguousarray(a)
        require((list(a.shape),str(a.dtype)) in SHAPES and np.isfinite(a).all(),'fixed native shape dtype')
        f=io.BytesIO();np.save(f,a,allow_pickle=False);require(len(f.getvalue())==a.nbytes+128,'NPY1 exact128 header')
        body=gzip.compress(f.getvalue(),mtime=0);n=a.nbytes+128
        require(len(body)<=n+(n+7)//8+(n+63)//64+11+18,'generic gzip bound')
        file=name+'.npy.gz';h=self.write(file,body)
        return dict(file=file,sha256=h,array_sha256=hashlib.sha256(a.tobytes()).hexdigest(),shape=list(a.shape),dtype=str(a.dtype))


class Ledger(bridge.native.Ledger):
    # Parent capacity is 4*n images; owned dispatch narrows to exact availability.
    def __init__(self,*args,available=10,**kwargs):
        require(type(available) is int and 0<=available<=10,'exact available image count')
        self.available=available
        super().__init__(max(1,(available+3)//4),*args,**kwargs)
    def dispatch(self,kind,tensor,callback,*args,**kwargs):
        require(len(self.entries)<2*self.available and self.image==len(self.entries)//2,'exact available image ordinal guard')
        return super().dispatch(kind,tensor,callback,*args,**kwargs)


def process(p,rows,store,encode,check=lambda:None):
    require(rows==p['rows'] and len(rows)==10,'fixed10 input roster')
    ordinal=0
    for row in rows:
        if row['source'] is None:
            store.event(dict(kind='image_missing',index=row['index'],reason=row['missing_reason']));continue
        check();scope=dict(row=row['index'],step=15,phase='encode');row['status']='attempted'
        store.event(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        pix=bridge.load_pixels(row['source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pix)
        z=encode(pix,row,scope,ordinal);bridge.latent(z)
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)));ordinal+=1


def tasks_for(p,rows,load):
    tasks=[]
    for spec,row in zip(p['rows'],rows,strict=True):
        if row['status']!='encoded':tasks.append(None);continue
        z=bridge.latent(load(row['latent']));t=deepcopy(spec['cp10_task'])
        ctx=deepcopy(t['root_context']);ctx['latent_history']=ctx['latent_history'][1:]+[z.tolist()]
        ctx['action_history']=deepcopy(spec['normalized_action_history'])
        identity=dict(namespace=NAME,index=spec['index'],root_slot=spec['root_slot'],role=spec['role'],probe=spec['probe'],
            history_absolute_steps=[15,20,25],source=spec['source'],cp10_identity=spec['cp10_identity'],
            raw10_sha256=hashlib.sha256(bytes.fromhex(spec['causal_raw10_bits'])).hexdigest(),
            encoded_latent=row['latent'],cp10_task_sha256=old.digest(spec['cp10_task']))
        t['root_context']=ctx
        t['binding'].update(context_sha256=old.digest(ctx),task_id=old.digest([NAME,identity]),
            task_sha256=old.digest([spec['cp10_task'],ctx,identity]),task_contract_sha256=old.digest(identity))
        old.controls.bridge.typed.core.validate_task(t)
        tasks.append(dict(index=spec['index'],task=t,task_sha256=old.digest(t),identity=identity,**FLAGS))
    return tasks


def publish(store,r,started,clock=time.monotonic):
    store.put('report.json',r,final=True);hashes={p.name:sha(str(p)) for p in store.path.iterdir()}
    elapsed=clock()-started
    c=dict(schema=NAME+'.completion',report_sha256=hashes['report.json'],prepared=r['prepared'],
        prepare_sha256=r['prepare_sha256'],artifact_sha256=hashes,
        bytes_before_completion=sum(p.stat().st_size for p in store.path.iterdir()),inclusive_wall_seconds=elapsed,
        status='published_within_cap' if elapsed<=WALL else 'publication_wall_failed',**FLAGS)
    store.put('completion.json',c,final=True)
    if clock()-started>WALL:
        store.put('completion_failure.json',dict(status='whole_wall_failed'),final=True)
        raise ValueError('wall failed; partial artifacts retained')
    return dict(output=str(store.path),report_sha256=hashes['report.json'],completion_sha256=sha(str(store.path/'completion.json')))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'explicit MAIN HOST CUDA authority before I/O')
    budget=Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];trace=guard=model=None;before=after=None
    precision=None;numeric_before=numeric_after=classes_before=classes_after=peak=None;source_ok=False
    store.put('binding.json',dict(prepared=prepared,prepare_sha256=expected));store.put('initial_roster.json',rows)
    def check():
        store.check();require(torch.cuda.memory_allocated(device)<=3*1024**3 and torch.cuda.memory_reserved(device)<=4*1024**3,'GPU3/4GiB')
        require(old.precision_state(torch)==planner.precision_contract(),'actual precision flags')
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
        classes_before=old.model_inventory(model,p['bindings'])
        trace=bridge.Trace(model,torch,store,check,p['expected_calls'])
        guard=Ledger(lambda e:store.event(dict(kind='guard_event',event=e)),lambda:torch.cuda.synchronize(device),
            lambda:not torch.is_grad_enabled(),available=p['expected_calls']['encoder'])
        transform=p0.image_transform()
        def encode(pix,row,scope,ordinal):
            guard.image=ordinal
            with physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pix,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pix),'native preprocess exact')
                z=model.encode({'pixels':x})['emb'];require(tuple(z.shape)==(1,1,192),'native192')
                return z[0,0].detach().cpu().numpy().copy()
        with trace,bridge.native.guard(model,torch,guard,errors),torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            numeric_before=old.precision_state(torch);check();process(p,rows,store,encode,check)
            torch.cuda.synchronize(device);check();numeric_after=old.precision_state(torch)
            peak=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
    except BaseException as e:errors.append(physical.kernel.error('cp15_encode',e))
    finally:
        if precision is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=precision
        try:
            if model is not None:
                bridge.native.frozen(model);after=bridge.native.tensor_hash(model);classes_after=old.model_inventory(model,p['bindings'])
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(physical.kernel.error('final_closure',e))
        modules=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash']
            and classes_before==classes_after and numeric_before==numeric_after==planner.precision_contract()
            and peak is not None and all(n<=cap for n,cap in zip(peak,(3*1024**3,4*1024**3),strict=True))
            and bridge.counts(modules)=={k:dict(attempted=n,completed=n) for k,n in p['expected_calls'].items()})
        seal=dict(rows=rows,modules=modules,guards=guards,blocked=blocked,errors=errors)
        seal_sha=store.put('native_seal.json',seal,final=True)
        store.event(dict(kind='all10_native_sealed',sha256=seal_sha,complete=complete),final=True)
        tasks=[None]*10
        try:
            if complete:tasks=tasks_for(p,rows,lambda e:bridge.rgb.load_array(e,lambda n:(store.path/n).read_bytes()))
        except BaseException as e:errors.append(physical.kernel.error('context_derivation',e));complete=False
        tasks_sha=store.put('cp15_tasks.json',tasks,final=True)
        store.event(dict(kind='cp15_tasks_published',sha256=tasks_sha,complete=complete),final=True)
        r=dict(schema=NAME+'.report',status=RECORDED if complete else 'partial_or_failed',prepared=prepared,prepare_sha256=expected,
            contract=contract(),**seal,counts=bridge.counts(modules),model_hash_before=before,model_hash_after=after,
            model_inventory_before=classes_before,model_inventory_after=classes_after,source_verified=source_ok,
            precision_before=numeric_before,precision_after=numeric_after,CUDA_peak_bytes=peak,
            native_seal_sha256=seal_sha,tasks_sha256=tasks_sha,artifact_sha256=deepcopy(store.hashes),
            event_count=store.index,elapsed_seconds=time.monotonic()-budget.started,**FLAGS)
        result=publish(store,r,budget.started)
    require(complete,'partial saved; no retry '+str(errors));return result


def validate(p,r,events,load):
    require(r['contract']==p['contract']==contract() and len(r['rows'])==len(p['rows'])==10
        and r['errors']==r['blocked']==[],'complete fixed ten encoder transport')
    want=[];mods=[];guards=[];ordinal=0
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')}==
            {k:v for k,v in spec.items() if k not in ('status','latent')},'entire immutable input row')
        if spec['source'] is None:
            require(row==spec,'missing remains untouched null')
            want.append(dict(kind='image_missing',index=spec['index'],reason=spec['missing_reason']));continue
        require(row['status']=='encoded','present cp15 encoded exactly once')
        pix=load(row['pixels']);bridge.same(pix,bridge.load_pixels(spec['source']),'actual cp15 PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'original preprocess')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=15,phase='encode')
        en,proj=row['modules']
        require([en['resource'],proj['resource']]==['encoder','projector']
            and all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in (en,proj)),
            'only encoder/projector native APIs')
        buffers=[pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z]
        require(all(a.dtype==np.float32 and np.isfinite(a).all() for a in buffers),'all full f32 module arrays')
        physical.saved.overlay.visual_bridge(*buffers,[1,257,192])
        bridge.same(buffers[1],pre.reshape(1,3,224,224),'encoder input');bridge.same(buffers[3],buffers[2][:,0],'literal CLS')
        bridge.same(z,buffers[4][0],'literal projected external latent')
        want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in (en,proj):
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want.extend([dict(kind='module_attempt',entry=attempt),
                dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)])
        mods.extend((en,proj));want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1
    want.extend([dict(kind='all10_native_sealed',sha256=r['native_seal_sha256'],complete=True),
        dict(kind='cp15_tasks_published',sha256=r['tasks_sha256'],complete=True)])
    require(events==want and r['modules']==mods and r['guards']==guards
        and [m['index'] for m in mods]==list(range(2*ordinal))
        and r['counts']==bridge.counts(mods)=={k:dict(attempted=n,completed=n) for k,n in p['expected_calls'].items()},
        'full events/modules/actual costs, no hidden or refunded calls')
    return tasks_for(p,r['rows'],load)


def consume(output,report_sha,completion_sha):
    out=Path(output);descriptor(str(out/'report.json'),report_sha);descriptor(str(out/'completion.json'),completion_sha)
    paths=list(out.iterdir());require(len(paths)<=185 and all(f.is_file() and not f.is_symlink() for f in paths)
        and sum(f.stat().st_size for f in paths)<=CAP,'bounded flat entire RGB output')
    files={f.name:f.read_bytes() for f in paths};hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    require('completion_failure.json' not in files and sum(len(b) for n,b in files.items() if n.endswith('.json'))<=META,'metadata cap/final transport')
    decode=lambda n:old.controls.native.decode(files[n])
    require(all(n.endswith(('.json','.npy.gz')) and (not n.endswith('.json') or files[n]==raw(decode(n))) for n in files),'canonical owned formats')
    r,c=decode('report.json'),decode('completion.json')
    require(set(r)=={'schema','status','prepared','prepare_sha256','contract','rows','modules','guards','blocked','errors',
        'counts','model_hash_before','model_hash_after','model_inventory_before','model_inventory_after','source_verified',
        'precision_before','precision_after','CUDA_peak_bytes','native_seal_sha256','tasks_sha256','artifact_sha256',
        'event_count','elapsed_seconds'}|set(FLAGS)
        and set(c)=={'schema','report_sha256','prepared','prepare_sha256','artifact_sha256','bytes_before_completion',
        'inclusive_wall_seconds','status'}|set(FLAGS),'exact owned report/completion fields')
    require(c['schema']==NAME+'.completion' and c['status']=='published_within_cap' and c['report_sha256']==report_sha
        and c['artifact_sha256']=={n:h for n,h in hashes.items() if n!='completion.json'}
        and c['bytes_before_completion']==sum(len(b) for n,b in files.items() if n!='completion.json')
        and c['prepared']==r['prepared'] and c['prepare_sha256']==r['prepare_sha256']
        and 0<=r['elapsed_seconds']<=c['inclusive_wall_seconds']<=WALL,'all publication bytes/time/binding')
    p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']==RECORDED and r['source_verified'] is True
        and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and r['model_inventory_before']==r['model_inventory_after'] and r['model_inventory_before']
        and r['precision_before']==r['precision_after']==planner.precision_contract()
        and all(r[k] is c[k] is False for k in FLAGS),'frozen CUDA/source/precision all flags')
    for e in r['model_inventory_before']:require(p['bindings'][e['path']]==e['sha256'],'all original model class sources')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2
        and all(type(n) is int and 0<=n<=cap for n,cap in zip(r['CUDA_peak_bytes'],(3*1024**3,4*1024**3),strict=True)),'GPU actual axes')
    require(set(files)==set(r['artifact_sha256'])|{'report.json','completion.json'}
        and all(hashes[n]==h for n,h in r['artifact_sha256'].items())
        and decode('binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'])
        and decode('initial_roster.json')==p['rows'],'whole input/artifact census')
    require(hashes['native_seal.json']==r['native_seal_sha256']
        and decode('native_seal.json')=={k:r[k] for k in ('rows','modules','guards','blocked','errors')},'all outputs sealed before context derivation')
    used=set()
    def load(e):used.add(e['file']);return bridge.rgb.load_array(e,lambda n:files[n])
    n=p['expected_calls']['encoder'];require(r['event_count']==10+9*n+2,'fixed missing plus full native events')
    tasks=validate(p,r,[decode(f'event_{i:04}.json') for i in range(r['event_count'])],load)
    require(tasks==decode('cp15_tasks.json') and hashes['cp15_tasks.json']==r['tasks_sha256']
        and used=={n for n in files if n.endswith('.npy.gz')} and len(used)==7*n,'all tasks and full seven arrays per actual image')
    io_bytes=sum(load(e).nbytes for m in r['modules'] for e in m['inputs']+[m['output']])
    require(io_bytes==n*sum(census()['per_image_bytes'][2:6]),'full actual module IO byte census')
    bridge.verify_bindings(p['bindings']);require({f.name:sha(str(f)) for f in out.iterdir()}==hashes,'stable full bytes')
    return dict(schema=NAME+'.consume',status=RECORDED,report_sha256=report_sha,completion_sha256=completion_sha,
        rows=r['rows'],tasks=tasks,fixed_denominator=10,encoded=n,missing=10-n,known_roots=2,fresh_roots=0,
        counts=r['counts'],sample_frame_counts={k:dict(API=n,samples=n,frames=n) for k in ('encoder','projector')},
        actual_module_IO_bytes=io_bytes,
        source_authorities=p['authorities'],consume_new_neural_calls=0,consume_new_physics=0,total_FLOPs=None,**FLAGS)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);s=q.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('prepare');a.add_argument('--output',required=True)
    for key in ('physical-prepare','physical-batch','physical-receipt'):
        a.add_argument('--'+key,required=True);a.add_argument('--'+key+'-sha',required=True)
    for cmd in ('verify','run','consume'):
        a=s.add_parser(cmd)
        if cmd!='consume':a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd in ('run','consume'):a.add_argument('--output',required=True)
        if cmd=='run':a.add_argument('--authorization',required=True)
        if cmd=='consume':
            for k in ('report-sha','completion-sha','receipt'):a.add_argument('--'+k,required=True)
    a=q.parse_args(argv)
    if a.cmd=='prepare':
        authorities={k:dict(path=getattr(a,k),sha256=getattr(a,k+'_sha')) for k in ('physical_prepare','physical_batch','physical_receipt')}
        result=write(a.output,build(authorities))
    elif a.cmd=='verify':p=verify(a.prepared,a.prepare_sha);result=dict(verified=True,slots=10,calls=p['expected_calls'])
    elif a.cmd=='run':result=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:result=write(a.receipt,consume(a.output,a.report_sha,a.completion_sha))
    print(old.controls.native.raw(result).decode())


if __name__=='__main__':main()
