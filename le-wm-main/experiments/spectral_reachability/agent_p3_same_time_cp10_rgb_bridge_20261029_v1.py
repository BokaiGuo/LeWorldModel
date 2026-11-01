"""Five fixed native cp10 RGB -> actual abs10/15/20 task bridge, MAIN CUDA only.

No physics, prediction, AD, training, Q scoring or scientific admission.
Inherited MAIN full receipts are explicit trust anchors, not recursively rerun.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import inspect
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p3_secant_controls_first5_physical_20261029_v1 as controls
from experiments.spectral_reachability import agent_p3_h3_witness_encoding_20261029_v1 as kernel

bridge,physical,planner=kernel.bridge,kernel.physical,kernel.planner
require,sha,raw=controls.require,controls.sha,controls.raw
digest=controls.producer_digest
ROOT=controls.ROOT
NAME='agent_p3_same_time_cp10_rgb_bridge_20261029_v1'
AUTH='MAIN_P3_FIVE_CP10_RGB_HOST_CUDA_ONLY'
RECORDED='all5_cp10_RGB_and_actual_history_tasks_sealed_NOT_admitted'
ORDER=[(0,'C'),(0,'Z'),(0,'P'),(1,'Z'),(1,'P')]
CAP,FINAL,FLOOR,WALL,META=64*1024**2,8*1024**2,200*1024**2,600,128*1024**2
METADATA=16*1024**2
FLAGS=deepcopy(controls.FLAGS)
PROTOCOL=controls.PROTOCOL
PROTOCOL_SHA=controls.PROTOCOL_SHA
ZP=controls.NAME
AUTHORITIES=deepcopy(controls.AUTHORITIES)
AUTHORITIES.update({
 'ZP_prepare':('results/'+ZP+'_prepare_MAIN01.json','54adc4f5c5183815f9e78c660ec0c3026f52f5ecab8056c322307d6c5fad7f5a'),
 'ZP_report':('results/'+ZP+'_physics_MAIN01/report.json','459c30ef31d2b062fa8ddcafbddaa6d125972706edd7458811bab51a654bc646'),
 'ZP_completion':('results/'+ZP+'_physics_MAIN01/completion.json','4dfd3f22f78469a01d6b71dbba8fbf921a8bd35730727cc7d8ff844123c87eef'),
 'ZP_receipt':('results/'+ZP+'_consume_MAIN01.json','c58ca1e43439841a3e592edb1128f95e2b5b8375af8f686052ab662930e74b50'),
 'ZP_source':('experiments/spectral_reachability/'+ZP+'.py','e9403bd9608dcd43bde780b538e60d24711c51ca568e012083d855d47e47096c'),
 'encoder_source':('experiments/spectral_reachability/'+kernel.NAME+'.py','c3c7f4421ffe843acc845ba9ef1bc19424ae66e4d08fa96b7a3fc63f22bb4748')})
SHAPES=[([224,224,3],'uint8'),([1,1,3,224,224],'float32'),([1,3,224,224],'float32'),
    ([1,257,192],'float32'),([1,192],'float32'),([1,192],'float32'),([192],'float32')]


def descriptors():return {k:dict(path=str(ROOT/p),sha256=h) for k,(p,h) in AUTHORITIES.items()}
def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def shape_census():
    n=[int(np.prod(s))*np.dtype(d).itemsize for s,d in SHAPES]
    # np.save v1 header 128B; gzip wrapper18B + compressBound31B (conservative).
    def upper(x):return x+(x>>12)+(x>>14)+(x>>25)+13+18
    gz=5*sum(upper(x+128) for x in n)
    return dict(images=5,arrays=35,per_image_array_bytes=n,raw_stored_array_bytes=5*sum(n),
        native_module_IO_bytes=5*sum(n[2:6]),gzip_NPY_upper_bytes=gz,metadata_cap_bytes=METADATA,
        working_upper_bytes=gz+METADATA,final_reserve_bytes=FINAL,total_upper_bytes=gz+METADATA+FINAL,
        output_cap_bytes=CAP,shape_dtype=[[s,d] for s,d in SHAPES],
        metadata_scope='every JSON copy, including events, roster, seal, tasks, report, completion; aggregate hard quota')


def contract():
    return dict(schema=NAME+'.contract',order=[list(x) for x in ORDER],fixed_denominator=5,
        known_roots=2,fresh_roots=0,source_conditions=['off']*5,absolute_steps=[10,15,20],control_steps=[0,5,10],
        new_calls={k:5 if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        sample_frame_axes='each encoder and projector: API5 samples5 frames5; nested axes not summed',
        native_hidden=[1,257,192],no_goal_encode=True,no_history_reencode=True,no_dedup=True,
        action_history='literal prefix5 + role raw5, original f32 mean/std once, (2,10)',
        context='inherit actual cp5 last2 latent slots abs10/15; append newly encoded actual abs20',
        task_namespace=NAME,cp5_task_id_reuse=False,Q_scoring=0,AD=0,new_physics=0,
        scientific_scope='known2 input bridge only; not ability/non-greedy benefit or task success',
        upstream_trust='pinned MAIN full C/ZP receipts, whole rows/census/SHA closure; no nested old fullstrict',
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH_requested_plus_observed_flags_guard',
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        protocol_sha256=PROTOCOL_SHA,shape_census=shape_census(),total_FLOPs=None,**FLAGS)


def physical_authority(prep,report,completion,receipt,prefix,name,status):
    d=descriptors()
    require(prep['schema']==name+'.prepare' and report['schema']==name+'.report'
        and receipt['schema']==name+'.consume' and completion['schema']==name+'.completion'
        and report['status']==receipt['status']==status,'exact completed physical own schema')
    require(receipt['physical_report_sha256']==completion['report_sha256']==d[prefix+'_report']['sha256']
        and receipt['completion_sha256']==d[prefix+'_completion']['sha256'] and receipt['completion']==completion
        and report['execution_binding']['prepared']==completion['prepared']==d[prefix+'_prepare']['path']
        and report['execution_binding']['prepare_sha256']==completion['prepare_sha256']==d[prefix+'_prepare']['sha256']
        and completion['status']=='published_within_cap'
        and 0<=report['wall_seconds']<=completion['inclusive_wall_seconds']<=controls.WALL,
        'four physical authorities/report publication bound')
    require(report['rows']==receipt['rows'] and len(report['rows'])==4
        and report['counts']==receipt['counts'] and report['errors']==receipt['errors']==[]
        and report['inputs_final_verified'] is True and receipt['fixed_denominator']==4
        and receipt['planner_authority']==report['execution_binding']['planner_authority']==prep['planner_authority'],
        'whole persisted physical receipt rows/accounting match actual report')
    require(all(all(v[k] is False for k in FLAGS) for v in (report,receipt,completion)),
        'physical evidence no science authorization')
    for i,row in enumerate(report['rows']):
        require(row['index']==i and row['plan']==prep['plans'][i],'all original physical plans retained')
    for key in ('prepare','report','completion'):
        require(receipt['bindings'][d[prefix+'_'+key]['path']]==d[prefix+'_'+key]['sha256'],
            'receipt-bound physical authority '+key)


def fixed_rows(cp,cr,zp,zr):
    """Pure source join; no rankings, outcomes or Q enter an input rule."""
    controls.validate_execution(zp)
    require(cr['schema']==controls.CNAME+'.consume' and zr['schema']==controls.NAME+'.consume'
        and len(cr['rows'])==len(zr['rows'])==4,'all four C and ZP denominator rows')
    require([r['status'] for r in cr['rows']]==['horizon_completed']+['no_witness']*3
        and all(r['status']=='horizon_completed' for r in zr['rows']),'fixed full5 available; no subset preparation')
    rows=[]
    for i,(root,role) in enumerate(ORDER):
        source_kind='C' if role=='C' else 'ZP'
        slot=0 if role=='C' else i-1
        prepared=cp if role=='C' else zp;strict=cr if role=='C' else zr
        plan=prepared['plans'][slot];actual=strict['rows'][slot];ref=prepared['prefix_reference'][slot]
        require(actual['plan']==plan and actual['root_slot']==root and actual['arm']==('H1' if role=='C' else role)
            and actual['task_id']==plan['task_id'] and actual['selection_digest']==plan['selection_digest']
            and plan['root_slot']==root and actual['initial_exact'] is actual['prefix_reference_exact'] is True
            and actual['own_returned']==10 and actual['PNG']==66 and actual['missing_checkpoints']==[]
            and actual['prefix51']['complete'] is True and len(actual['prefix51']['frames'])==51,
            'actual full guarded known-root cp10, not cp5/earlystop/another arm')
        task=plan['model_task'];controls.bridge.typed.core.validate_task(task)
        require(task['binding']['model_sha256']==bridge.rgb.MODEL_TENSOR_SHA
            and np.array_equal(task['projection'],np.eye(192)),'original native192 identity task/model')
        same_root=zp['plans'][0 if root==0 else 2]
        require(all(controls.bridge.typed.same(plan[k],same_root[k]) for k in
            ('model_task','base_task','provenance','prefix_commands','native_goal','native_goal_pose','native_root_sha256','native_goal_pixel_sha256')),
            'C/Z/P share exact original root goal normalizer actual cp5 context')
        commands=controls.f32(plan['commands'],(10,2))
        require(commands[:5].tobytes()==controls.f32(ref['prefix_raw5'],(5,2)).tobytes()
            and controls.bytehash(commands)==plan['raw_float32_sha256'],'literal committed first5 and role5')
        require([x['absolute_step'] for x in ref['history_frames']]==[5,10,15]
            and controls.bridge.typed.same(ref['actual_context'],task['root_context']),'inherited actual history chronology')
        cp10=[x for x in actual['checkpoints'] if x['step']==10]
        selected=[f for f in actual['trajectory'] if f['phase']=='plan' and f['overlay_index']==0 and f['step'] in (0,5,10)]
        require(len(cp10)==1 and [f['step'] for f in selected]==[0,5,10]
            and cp10[0]=={k:selected[-1][k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')},
            'real off cp0/5/10 source frames')
        for frame,anchor in zip(selected[:2],ref['history_frames'][1:],strict=True):
            require(all(controls.bridge.typed.same(frame[k],anchor[k]) for k in
                ('file_sha256','pixel_sha256','state','snapshot')),'abs10/15 originals authenticate inherited last2 latents')
        for frame in selected:
            require(frame['condition']=='off' and frame['state']==frame['snapshot']['state'],'same-frame native state identity')
        response=actual['responses'][-1]
        require(response['step']==10 and controls.bridge.typed.same(response['state'],selected[-1]['state']),
            'actual terminal native response belongs to this same cp10 image')
        d=descriptors();out=Path(d[source_kind+'_report']['path']).parent
        frame=deepcopy(cp10[0]);frame['path']=str(out/frame['png'])
        require(Path(frame['png']).name==frame['png'],'flat original actual PNG')
        normalized=controls.bridge.typed.tokens(commands,np.asarray(task['action_mean'],np.float32),
            np.asarray(task['action_std'],np.float32)).reshape(2,10)
        hist=deepcopy(ref['history_frames'][1:])
        hist.append(dict(absolute_step=20,plan_step=10,png=frame['path'],
            **{k:deepcopy(frame[k]) for k in ('file_sha256','pixel_sha256','state','snapshot')}))
        rows.append(dict(index=i,root_slot=root,role=role,checkpoint=10,absolute_steps=[10,15,20],
            source_slot=slot,source_kind=source_kind,source_receipt=deepcopy(d[source_kind+'_receipt']),
            source_report=deepcopy(d[source_kind+'_report']),selection_digest=plan['selection_digest'],
            source=frame,history_sources=hist,cp5_task=deepcopy(task),
            base_task_id=plan['task_id'],old_cp5_task_id=task['binding']['task_id'],
            original_native_goal=deepcopy(plan['native_goal']),native_response=deepcopy(actual['responses'][-1]),
            causal_raw10=commands.tolist(),causal_raw10_bits=commands.tobytes().hex(),
            normalized_action_history=normalized.tolist(),normalized_float32_sha256=controls.bytehash(normalized),
            status='not_started',latent=None))
    return rows


def build():
    require(sha(PROTOCOL)==PROTOCOL_SHA and shape_census()['total_upper_bytes']<CAP,'fixed protocol/shape proof')
    d=descriptors();v={}
    for key,ref in d.items():
        require(sha(ref['path'])==ref['sha256'],'fixed original source/authority SHA '+key)
        if not key.endswith('_source'):v[key]=controls.read(ref['path'],META)
    cp,cr,zp,zr=(v[k] for k in ('C_prepare','C_receipt','ZP_prepare','ZP_receipt'))
    physical_authority(cp,v['C_report'],v['C_completion'],cr,'C',controls.CNAME,'secant_commit5_native_recorded_NOT_admitted')
    physical_authority(zp,v['ZP_report'],v['ZP_completion'],zr,'ZP',controls.NAME,controls.RECORDED)
    rows=fixed_rows(cp,cr,zp,zr)
    require(zr['endpoints']==v['ZP_report']['endpoints']==controls.endpoint_rows(zr['rows'])
        and all(x['available'] for x in zr['endpoints']) and zr['saved_PNG_verified']==264,
        'all4 ZP available endpoints, complete PNG authority')
    bindings=deepcopy(zr['bindings']);controls.native.merge(bindings,cr['bindings'])
    controls.native.merge(bindings,{x['path']:x['sha256'] for x in d.values()})
    controls.native.merge(bindings,bridge.rgb.MODELS)
    files=bridge.source_files()+own_files()+[Path(controls.__file__).resolve(),Path(kernel.__file__).resolve(),PROTOCOL]
    controls.native.merge(bindings,{str(p.resolve()):sha(p) for p in files})
    for row in rows:
        bridge.load_pixels(row['source'])
        for f in row['history_sources']:require(sha(f['png'])==f['file_sha256'],'actual history RGB file identity')
    bridge.verify_bindings(bindings)
    return dict(schema=NAME+'.prepare',contract=contract(),rows=rows,bindings=bindings,
        expected_calls=deepcopy(contract()['new_calls']),policy=str(bridge.rgb.POLICY),
        shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json')),
        expected_model_hash=bridge.rgb.MODEL_TENSOR_SHA,authorities=d,
        inherited_physical_counts=dict(C=deepcopy(cr['counts']),ZP=deepcopy(zr['counts'])),
        upstream_fullstrict_reconstructed_here=False,
        upstream_trust='MAIN pinned complete receipts, exact whole rows/closure; not historical recertification')


def write(path,value):
    path=Path(controls.native.absolute(str(path)));body=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists()
        and not path.is_symlink() and len(body)<=META and shutil.disk_usage(path.parent).free>=FLOOR+len(body),
        'exclusive own bounded metadata')
    with path.open('xb') as f:require(f.write(body)==len(body),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path))


def verify(path,expected):
    controls.native.actual_hash(expected);require(sha(path)==expected,'external prepare SHA')
    p=controls.read(path,META);require(raw(p)==raw(build()) and sha(path)==expected,'entire own prepare rejoin/source identity')
    return p


class Budget(bridge.rgb.Budget):
    def check(self,size=0,final=False):
        require(final or self.clock()-self.started<=WALL,'600s whole run')
        require(self.used+size<=CAP-(0 if final else FINAL) and self.free()>=FLOOR+size,'cap/reserve/floor')


class Store(bridge.rgb.Store):
    def __init__(self,output,budget):
        path=Path(controls.native.absolute(str(output)))
        require(path.name.startswith(NAME+'_') and shutil.disk_usage(path.parent).free>=CAP+FLOOR,'own new full-cap directory')
        super().__init__(path,budget);self.index=0;self.arrays=0;self.metadata_bytes=0
    def write(self,name,value,final=False):
        if name.endswith('.npy.gz'):
            require(self.arrays<35 and len(value)<=max_array_file_bound(),'full35 native arrays only')
        else:
            require(name.endswith('.json') and value==raw(controls.native.decode(value)),'canonical JSON')
            require(self.metadata_bytes+len(value)<=METADATA,'aggregate all metadata copies quota')
        h=super().write(name,value,final)
        if name.endswith('.npy.gz'):self.arrays+=1
        else:self.metadata_bytes+=len(value)
        return h
    def put(self,name,value,final=False):return self.write(name,raw(value),final)
    def event(self,e,final=False):self.put(f'event_{self.index:04}.json',e,final=final);self.index+=1
    def array(self,name,a):
        a=np.ascontiguousarray(a)
        require((list(a.shape),str(a.dtype)) in SHAPES and np.isfinite(a).all(),'only exact full native encoding array census')
        stream=io.BytesIO();np.save(stream,a,allow_pickle=False)
        require(len(stream.getvalue())==a.nbytes+128,'observed fixed NPY1 header size')
        body=gzip.compress(stream.getvalue(),mtime=0)
        n=a.nbytes+128;bound=n+(n>>12)+(n>>14)+(n>>25)+13+18
        require(len(body)<=bound,'lossless actual gzip bound')
        file=name+'.npy.gz';h=self.write(file,body)
        return dict(file=file,sha256=h,array_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            shape=list(a.shape),dtype=str(a.dtype))


def max_array_file_bound():
    n=max(shape_census()['per_image_array_bytes'])+128
    return n+(n>>12)+(n>>14)+(n>>25)+13+18


class Ledger(bridge.native.Ledger):
    def __init__(self,*args,**kwargs):super().__init__(2,*args,**kwargs)
    def dispatch(self,kind,tensor,callback,*args,**kwargs):
        require(len(self.entries)<10 and self.image==len(self.entries)//2,'exact five image guard, no sixth')
        return super().dispatch(kind,tensor,callback,*args,**kwargs)


def process(p,rows,store,encode,check=lambda:None):
    require(rows==p['rows'] and len(rows)==5,'fixed all5 immutable source roster')
    for row in rows:
        check();scope=dict(row=row['index'],step=10,phase='encode');row['status']='attempted'
        store.event(dict(kind='image_attempt',scope=scope,guard_image=row['index']))
        pixels=bridge.load_pixels(row['source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pixels)
        z=encode(pixels,row,scope,row['index']);bridge.latent(z)
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)))


def task_for(spec,latent,latent_receipt):
    old=spec['cp5_task'];controls.bridge.typed.core.validate_task(old)
    z=bridge.latent(latent);incoming=controls.f32(spec['causal_raw10'],(10,2))
    tokens=controls.bridge.typed.tokens(incoming,np.asarray(old['action_mean'],np.float32),
        np.asarray(old['action_std'],np.float32)).reshape(2,10)
    require(tokens.tobytes()==np.asarray(spec['normalized_action_history'],np.float32).tobytes()
        and controls.bytehash(tokens)==spec['normalized_float32_sha256'],'once-normalized causal packets, no warmup tail at cp10')
    ctx=deepcopy(old['root_context'])
    ctx['latent_history']=deepcopy(ctx['latent_history'][1:])+[z.tolist()]
    ctx['action_history']=tokens.tolist()
    identity=dict(namespace=NAME,index=spec['index'],root_slot=spec['root_slot'],role=spec['role'],
        absolute_steps=[10,15,20],history_sources=deepcopy(spec['history_sources']),
        raw10_float32_sha256=controls.bytehash(incoming),old_cp5_task_sha256=digest(old),
        source_report=spec['source_report'],source_receipt=spec['source_receipt'],
        encoded_latent=deepcopy(latent_receipt))
    t=deepcopy(old);t['root_context']=ctx
    t['binding'].update(context_sha256=digest(ctx),task_id=digest([old['binding']['task_id'],identity]),
        task_sha256=digest([old,ctx,identity]),task_contract_sha256=digest([NAME,identity]))
    controls.bridge.typed.core.validate_task(t)
    require(t['binding']['task_id']!=old['binding']['task_id'],'cp10 never borrows cp5 task ID')
    return dict(index=spec['index'],root_slot=spec['root_slot'],role=spec['role'],task=t,
        task_sha256=digest(t),identity=identity,base_task_id=spec['base_task_id'],
        old_cp5_task_id=old['binding']['task_id'],absolute_steps=[10,15,20],
        source=deepcopy(spec['source']),causal_raw10=deepcopy(spec['causal_raw10']),
        context_scope='actual observations, not predictions; no forecast or science gate',**FLAGS)


def assemble_tasks(p,rows,load):
    require(len(rows)==len(p['rows'])==5,'all5 rows before task construction')
    return [None if row['status']!='encoded' else task_for(spec,load(row['latent']),row['latent'])
        for spec,row in zip(p['rows'],rows,strict=True)]


def publish(store,r,started,clock=None):
    clock=time.monotonic if clock is None else clock
    store.put('report.json',r,final=True)
    hashes={p.name:sha(p) for p in store.path.iterdir()}
    elapsed=clock()-started
    c=dict(schema=NAME+'.completion',report_sha256=hashes['report.json'],
        prepare_sha256=r['prepare_sha256'],prepared=r['prepared'],artifact_sha256=hashes,
        bytes_before_completion=sum(p.stat().st_size for p in store.path.iterdir()),
        inclusive_wall_seconds=elapsed,status='published_within_cap' if elapsed<=WALL else 'publication_wall_failed',
        wall_scope='through report fsync and artifact hash; completion self-publication excluded',**FLAGS)
    store.put('completion.json',c,final=True);h=sha(store.path/'completion.json')
    end=clock()-started
    if end>WALL:store.put('completion_failure.json',dict(wall_seconds=end,report_sha256=hashes['report.json']),final=True)
    require(end<=WALL,'inclusive publication wall exceeded; durable failure retained')
    return dict(status=r['status'],output=str(store.path),report_sha256=hashes['report.json'],completion_sha256=h)



def precision_state(torch):
    return dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
        autocast=torch.is_autocast_enabled('cuda'),math=torch.backends.cuda.math_sdp_enabled(),
        flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())


def model_inventory(model,bindings):
    result=[]
    for module in model.modules():
        path=str(Path(inspect.getfile(type(module))).resolve())
        require(path in bindings and sha(path)==bindings[path],'presealed installed model class source')
        result.append(dict(type=type(module).__module__+'.'+type(module).__qualname__,path=path,sha256=bindings[path]))
    require(result,'complete module inventory');return result


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA before I/O')
    budget=Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);precision=None;errors=[];trace=guard=None;model=None;before=after=None;source_ok=False
    numeric_before=numeric_after=None;classes_before=classes_after=None;peak=None
    def check():
        store.check()
        require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and
            torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'actual CUDA memory caps')
        require(precision_state(torch)==planner.precision_contract(),'actual TF32/autocast/MATH contract')
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
            and classes_before==classes_after and numeric_before==numeric_after==planner.precision_contract()
            and peak is not None and all(v<=contract()[k] for v,k in zip(peak,('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)) and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        # native seal precedes any analysis of the new encoded outputs.
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        store.event(dict(kind='all5_native_sealed',sha256=h,encoded=sum(x['status']=='encoded' for x in rows)),final=True)
        tasks=[None]*5
        try:
            if complete:
                tasks=assemble_tasks(p,rows,lambda e:bridge.rgb.load_array(e,lambda n:(store.path/n).read_bytes()))
                require(all(x is not None for x in tasks),'all5 cp10 tasks, no subset')
        except BaseException as e:errors.append(physical.kernel.error('task_binding',e));complete=False
        tasks_sha=store.put('cp10_tasks.json',tasks,final=True)
        store.event(dict(kind='cp10_tasks_published',sha256=tasks_sha,all5=complete),final=True)
        report=dict(schema=NAME+'.report',status=RECORDED if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,precision_before=numeric_before,precision_after=numeric_after,
            model_inventory_before=classes_before,model_inventory_after=classes_after,CUDA_peak_bytes=peak,
            tasks_sha256=tasks_sha,known_roots=2,fixed_denominator=5,**FLAGS)
        publication=publish(store,report,budget.started)
    require(complete,'partial retained/no retries '+str(errors))
    return publication



def validate(p,r,events,load):
    require(r['contract']==p['contract']==contract() and len(p['rows'])==len(r['rows'])==5
        and r['errors']==r['blocked']==[],'all5 clean original inputs retained')
    modules=[];guards=[];want=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')}==
            {k:v for k,v in spec.items() if k not in ('status','latent')},'literal original frame/task/causal input roster')
        require(row['status']=='encoded','all5 exactly once')
        pix=load(row['pixels']);bridge.same(pix,bridge.load_pixels(spec['source']),'actual physical PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'original preprocessing exact')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=10,phase='encode')
        mods=row['modules']
        require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector']
            and all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),
            'only scoped encoder and projector, no prediction')
        en,proj=mods
        buffers=[pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z]
        require(all(x.dtype==np.float32 and np.isfinite(x).all() for x in buffers),'full finite native f32 arrays')
        physical.saved.overlay.visual_bridge(*buffers,[1,257,192])
        bridge.same(buffers[1],pre.reshape(1,3,224,224),'preprocess to encoder literal bytes')
        bridge.same(buffers[3],buffers[2][:,0],'hidden CLS to projector literal bytes')
        bridge.same(z,buffers[4][0],'projector to external latent literal bytes')
        want.append(dict(kind='image_attempt',scope=scope,guard_image=row['index']))
        for mod in mods:
            g=dict(index=len(guards),image=row['index'],resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=attempt),
                dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row))
    want += [dict(kind='all5_native_sealed',sha256=r['native_seal_sha256'],encoded=5),
        dict(kind='cp10_tasks_published',sha256=r['tasks_sha256'],all5=True)]
    require(modules==r['modules'] and [x['index'] for x in modules]==list(range(10))
        and r['guards']==guards and events==want and r['counts']==bridge.counts(modules)=={
            k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},'all native/guard chronology and zero forbidden API')
    return assemble_tasks(p,r['rows'],load)


def consume(output,expected,completion_sha):
    output=Path(controls.native.absolute(str(output)))
    controls.native.actual_hash(expected);controls.native.actual_hash(completion_sha)
    require(output.is_dir() and not output.is_symlink(),'regular output directory')
    paths=list(output.iterdir())
    require(len(paths)<=94 and all(p.is_file() and not p.is_symlink() for p in paths)
        and sum(p.stat().st_size for p in paths)<=CAP,'aggregate bounded artifacts before reads')
    files={p.name:p.read_bytes() for p in paths}
    require('completion_failure.json' not in files and hashlib.sha256(files['report.json']).hexdigest()==expected
        and hashlib.sha256(files['completion.json']).hexdigest()==completion_sha,'external report/completion SHA, no publication failure')
    require(sum(len(b) for n,b in files.items() if n.endswith('.json'))<=METADATA,'independent metadata byte census')
    for n,b in files.items():
        require(n.endswith(('.json','.npy.gz')),'no hidden artifact format')
        if n.endswith('.json'):require(b==raw(controls.native.decode(b)),'canonical all JSON')
        else:require(len(b)<=max_array_file_bound(),'native file compressed bound')
    r=controls.native.decode(files['report.json']);c=controls.native.decode(files['completion.json'])
    require(c['schema']==NAME+'.completion' and c['status']=='published_within_cap'
        and c['report_sha256']==expected and c['prepared']==r['prepared'] and c['prepare_sha256']==r['prepare_sha256']
        and c['artifact_sha256']=={n:hashlib.sha256(b).hexdigest() for n,b in files.items() if n!='completion.json'}
        and c['bytes_before_completion']==sum(len(b) for n,b in files.items() if n!='completion.json')
        and c['wall_scope']=='through report fsync and artifact hash; completion self-publication excluded'
        and 0<=r['elapsed_seconds']<=c['inclusive_wall_seconds']<=WALL
        and all(c[k] is False for k in FLAGS),'inclusive wall/publication and complete artifact census')
    p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']==RECORDED and r['fixed_denominator']==5
        and r['known_roots']==2 and all(r[k] is False for k in FLAGS)
        and r['source_verified'] is r['model_unchanged'] is True
        and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'complete frozen HOST CUDA only')
    require(r['precision_before']==r['precision_after']==planner.precision_contract()
        and r['model_inventory_before']==r['model_inventory_after'] and r['model_inventory_before'],
        'observed TF32/autocast/SDP flags + original bound class source; not a kernel-level MATH certificate')
    for e in r['model_inventory_before']:require(p['bindings'][e['path']]==e['sha256'],'installed original model class source')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2
        and all(type(v) is int and 0<=v<=contract()[k] for v,k in zip(r['CUDA_peak_bytes'],
            ('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'actual GPU memory axes')
    require(set(files)==set(r['artifact_sha256'])|{'report.json','completion.json'}
        and all(hashlib.sha256(files[n]).hexdigest()==h for n,h in r['artifact_sha256'].items()),
        'no filtered or orphan artifacts')
    decode=lambda n:controls.native.decode(files[n])
    require(decode('binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'])
        and decode('initial_roster.json')==p['rows'],'full initial binding/roster unchanged')
    require(hashlib.sha256(files['native_seal.json']).hexdigest()==r['native_seal_sha256']
        and decode('native_seal.json')=={k:r[k] for k in ('rows','modules','guards','blocked','errors')},
        'all5 encoder outputs sealed before constructing tasks')
    used=set()
    def load(e):
        used.add(e['file'])
        return bridge.rgb.load_array(e,lambda n:files[n])
    require(r['event_count']==52,'all5 complete event denominator')
    tasks=validate(p,r,[decode(f'event_{i:04}.json') for i in range(r['event_count'])],load)
    require(tasks==decode('cp10_tasks.json') and hashlib.sha256(files['cp10_tasks.json']).hexdigest()==r['tasks_sha256'],
        'independent actual task/context/binding reconstruction')
    require(used=={n for n in files if n.endswith('.npy.gz')} and len(used)==35,'all35 native arrays used; no hidden output')
    io_bytes=sum(load(e).nbytes for m in r['modules'] for e in m['inputs']+[m['output']])
    require(io_bytes==shape_census()['native_module_IO_bytes'],'actual module tensor byte axis')
    bindings=deepcopy(p['bindings'])
    controls.native.merge(bindings,{r['prepared']:r['prepare_sha256']})
    controls.native.merge(bindings,{str(output/n):hashlib.sha256(b).hexdigest() for n,b in files.items()})
    bridge.verify_bindings(bindings)
    require({p.name:sha(p) for p in output.iterdir()}=={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},'final stable output census')
    return dict(schema=NAME+'.consume',status=RECORDED,report_sha256=expected,completion_sha256=completion_sha,
        tasks=tasks,fixed_denominator=5,known_roots=2,fresh_roots=0,counts=r['counts'],
        sample_frame_counts={k:dict(API=5,samples=5,frames=5) for k in ('encoder','projector')},
        actual_module_IO_bytes=io_bytes,actual_array_file_bytes=sum(len(b) for n,b in files.items() if n.endswith('.npy.gz')),
        actual_metadata_bytes=sum(len(b) for n,b in files.items() if n.endswith('.json')),
        shape_census=shape_census(),bindings=bindings,completion=c,
        inherited_physical_counts=p['inherited_physical_counts'],source_authorities=p['authorities'],
        consume_new_neural_calls=0,consume_new_physics=0,total_FLOPs=None,**FLAGS)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);sub=q.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--fixed-authorities',required=True,action='store_true')
    for name in ('verify','run','consume'):
        p=sub.add_parser(name)
        if name!='consume':p.add_argument('--prepared',required=True);p.add_argument('--prepare-sha',required=True)
        if name in ('run','consume'):p.add_argument('--output',required=True)
        if name=='run':p.add_argument('--authorize-host-cuda',action='store_true',required=True)
        if name=='consume':
            p.add_argument('--report-sha',required=True);p.add_argument('--completion-sha',required=True);p.add_argument('--receipt',required=True)
    a=q.parse_args(argv)
    if a.command=='prepare':v=write(a.output,build())
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=contract()['new_calls'])
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:v=write(a.receipt,consume(a.output,a.report_sha,a.completion_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()



