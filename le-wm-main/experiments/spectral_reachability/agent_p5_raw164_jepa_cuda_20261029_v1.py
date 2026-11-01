"""Fixed164 actual-frame15-conditioned factual H1 diagnostic, HOST CUDA only.

Same factual H1 timing as actual-cp5 FD164, but extra native agent proprio.
All164 full3 outputs are sealed before any successor image decode/encode.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np

from experiments.spectral_reachability import agent_p5_raw164_jepa_input_bridge_20261029_v1 as bridge

NAME='agent_p5_raw164_jepa_cuda_20261029_v1'
ROOT=bridge.ROOT
AUTH='MAIN_P5_RAW164_JEPA_FACTUAL_H1_CUDA_ONLY'
CAP,FLOOR,FINAL=4*1024**3,200*1024**2,64*1024**2
WALL,META,ARRAY_CAP=900,32*1024**2,4*1024**2
BRIDGE_SHA='d70352f2fc574ca8a226522fc3dda9f015305894c83a1d1826f05808c005da0f'
require,sha,digest,raw=bridge.require,bridge.sha,bridge.digest,bridge.raw
KEYS=('official_encoder','proprio_encoder','predictor')


def protocol():
    return dict(schema=NAME+'.protocol',requests=164,pairs=bridge.factor.schedule(),
        diagnostic='one-step factual successor conditioned on actual frame15',context_global_frames=[5,10,15],target_global_frame=20,
        alignment_to_actual_FD164=dict(context=[5,10,15],open_loop_packets=1,same_factual_H1_timing=True,
            extra_agent_proprio=True,fully_matched_input=False),
        not_root_H2='original root0/5/10 open-loop15/20 is a separate prediction contract',
        RGB_only=False,known_roots=2,fresh=False,conditions=['off','on'],
        counters='actual top-module pre/post hooks; batch_samples and frame_samples separate; no nested double counting',
        caps=dict(official_encoder=dict(api=328,batch_samples=656,frame_samples=656),
            proprio_encoder=dict(api=164,batch_samples=164,frame_samples=492),predictor=dict(api=164,batch_samples=164,frame_samples=492)),
        order='164*(context encoder,proprio encoder,predictor), durable full prediction seal,164 label encoder',
        encoder_dedup=False,encoder_dedup_reason='fixed per-request calls; duplicates still paid and metered',
        target='own factual observed successor; not counterfactual physical ground truth',
        score='float64 NumPy C-order global patch MSE; final token only; persistence separately',
        pairing='all80 signed original raw perturbation pairs; off/on separate; no significance or independent164 claim',
        cap_bytes=CAP,floor_bytes=FLOOR,final_reserve_bytes=FINAL,wall_seconds=WALL,
        wall_scope='dispatch and source/model final checks; bounded terminal fsync cleanup may finish afterwards',
        initial_cuda_free_floor_bytes=4*1024**3,
        array_cap_bytes=ARRAY_CAP,metadata_cap_bytes=META,
        neural_tensor_io_upper_bound_bytes=2*1024**3,
        tensor_bound_scope='all3 modules input/output tensor snapshots; not a GPU activation upper bound',
        backend='HOST CUDA offline, MATH attention, TF32off, frozen eval; no CPU neural fallback',
        new_physics=0,new_AD=0,new_training=0,P1_admission=False,P5_admission=False,
        matched_cross_model_performance=False,cross_architecture_MSE_magnitude_comparable=False)


def build(prepared,expected):
    require(sha(bridge.__file__)==BRIDGE_SHA,'frozen bridge source')
    q=bridge.verify(prepared,expected)
    bindings=deepcopy(q['bindings']);bindings[str(Path(prepared).absolute())]=expected
    for path in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(path)]=sha(path)
    return dict(schema=NAME+'.prepare',status='prepared_no_neural_execution',protocol=protocol(),
        bridge_prepared=str(Path(prepared).absolute()),bridge_prepare_sha256=expected,bridge=q,bindings=bindings)


def verify(prepared,expected):
    m=bridge.read(prepared,expected,META)
    require(m==build(m['bridge_prepared'],m['bridge_prepare_sha256']),'full rebuilt prepared closure')
    return m


def write_new(path,value):
    path=Path(path).absolute();body=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and
        not any(p.is_symlink() for p in (path,*path.parents)) and len(body)<=META and shutil.disk_usage(path.parent).free>=FLOOR+len(body),
        'new own bounded metadata/floor')
    with path.open('xb') as f:require(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())
    bridge.physical.paired.fsync_dir(path.parent)
    return dict(path=str(path),sha256=sha(path),status=value['status'])


class Store:
    """Exactly-one-LF canonical JSON, incremental byte/hash accounting, fsync."""
    def __init__(self,path):
        self.started=time.monotonic();self.path=Path(path).absolute()
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and not self.path.exists() and
            not any(p.is_symlink() for p in (self.path,*self.path.parents)) and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,
            'exclusive NEW4GiB+200MiB floor')
        self.path.mkdir();bridge.physical.paired.fsync_dir(self.path.parent)
        self.bytes=0;self.hashes={};self.sizes={};self.seq=0;self.previous=None;self.array_index=0;self.tensor_bytes=0

    def check(self,size=0,final=False):
        require(type(size) is int and size>=0 and self.bytes+size<=CAP-(0 if final else FINAL) and
            shutil.disk_usage(self.path).free>=FLOOR+size,'actual disk bytes/cap/floor/reserve')
        require(final or time.monotonic()-self.started<WALL,'full job900s before dispatch/write')

    def write(self,name,body,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and type(body) is bytes,'flat owned byte artifact')
        path=self.path/name;self.check(len(body),final)
        require(not path.is_symlink() and (not path.exists() or append and name in self.hashes and path.stat().st_size==self.sizes[name]),'exclusive file/owned append')
        before=self.sizes.get(name,0)
        try:
            with path.open('ab' if append else 'xb') as f:require(f.write(body)==len(body),'complete write');f.flush();os.fsync(f.fileno())
        finally:
            after=path.stat().st_size if path.exists() else 0;self.bytes+=after-before;self.sizes[name]=after
        h=self.hashes.setdefault(name,hashlib.sha256());h.update(body);bridge.physical.paired.fsync_dir(self.path)
        return dict(file=name,sha256=h.hexdigest(),file_bytes=self.sizes[name])

    def put(self,name,value,final=False):return self.write(name,raw(value),final=final)

    def emit(self,kind,final=False,**fields):
        e=dict(sequence=self.seq,previous_sha256=self.previous,kind=kind,**deepcopy(fields));e['event_sha256']=digest(e)
        self.write('journal.jsonl',raw(e),append=True,final=final)
        self.seq+=1;self.previous=e['event_sha256'];return e

    def array(self,value):
        require(isinstance(value,np.ndarray) and value.dtype==np.float32 and np.isfinite(value).all(),'unaltered finite float32 module tensor')
        require(self.tensor_bytes+value.nbytes<=protocol()['neural_tensor_io_upper_bound_bytes'],'actual complete tensor IO2GiB, no dropping')
        buf=io.BytesIO();np.save(buf,np.ascontiguousarray(value),allow_pickle=False);body=buf.getvalue()
        require(len(body)<=ARRAY_CAP,'per-tensor4MiB, never truncate')
        name=f'tensor_{self.array_index:05d}.npy';self.array_index+=1
        rec=dict(self.write(name,body),shape=list(value.shape),dtype='float32',nbytes=value.nbytes)
        self.tensor_bytes+=value.nbytes;return rec

    def inventory(self):return {n:dict(sha256=h.hexdigest(),file_bytes=self.sizes[n]) for n,h in self.hashes.items()}


def journal(path):
    previous=None
    with Path(path).open('rb') as f:
        i=0
        while True:
            line=f.readline(META+1)
            if not line:break
            require(len(line)<=META and line.endswith(b'\n') and not line.endswith(b'\n\n'),'one bounded LF journal line')
            value=json.loads(line,object_pairs_hook=bridge.pairs)
            require(raw(value)==line and value['sequence']==i and value['previous_sha256']==previous and
                digest({k:v for k,v in value.items() if k!='event_sha256'})==value['event_sha256'],'strict canonical full hashchain')
            previous=value['event_sha256'];i+=1;yield value


def tree(value,save,is_tensor):
    if is_tensor(value):return {'$array':save(value)}
    if isinstance(value,dict):
        require(all(type(k) is str for k in value),'string module dict keys')
        return {'$dict':{k:tree(v,save,is_tensor) for k,v in sorted(value.items())}}
    if isinstance(value,(tuple,list)):return {'$tuple' if isinstance(value,tuple) else '$list':[tree(v,save,is_tensor) for v in value]}
    require(value is None or type(value) in (str,int,float,bool),'literal module IO, no repr guessing')
    raw(value);return value


def restore(value,load):
    if not isinstance(value,dict):return value
    require(len(value)==1,'tagged IO tree')
    if '$array' in value:return load(value['$array'])
    if '$dict' in value:return {k:restore(v,load) for k,v in value['$dict'].items()}
    if '$tuple' in value:return tuple(restore(v,load) for v in value['$tuple'])
    require('$list' in value,'known module IO tag');return [restore(v,load) for v in value['$list']]


def units(key,shape):
    require(key in KEYS and all(type(n) is int and n>=0 for n in shape),'real module shape')
    require(len(shape)>0 and shape[0]>0,'nonempty batch')
    return dict(api=1,batch_samples=shape[0],frame_samples=shape[0] if key=='official_encoder' else shape[0]*shape[1])


def empty_counts():return {phase:{key:dict(api=0,batch_samples=0,frame_samples=0) for key in KEYS} for phase in ('attempted','returned')}


def add_counts(counts,phase,key,unit):
    for axis,v in unit.items():counts[phase][key][axis]+=v


def call_schedule():
    return [(i,'prediction',k) for i in range(164) for k in KEYS]+[(i,'label','official_encoder') for i in range(164)]


class Trace:
    def __init__(self,store):
        self.store=store;self.calls=[];self.counts=empty_counts();self.where=None;self.pending=None;self.handles=[]

    def install(self,models,torch):
        def save(t):
            require(t.is_cuda and t.dtype==torch.float32 and not t.requires_grad and torch.isfinite(t).all(),'actual finite frozen CUDA module IO')
            return self.store.array(t.detach().contiguous().cpu().numpy())
        def pre(key):
            def hook(module,args,kwargs):
                require(not torch.is_grad_enabled() and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32 and
                    self.pending is None and self.where is not None,'inference-only TF32off attributed nonnested top modules')
                i=len(self.calls);require(i<len(call_schedule()) and (*self.where,key)==call_schedule()[i],'fixed real module order/cap before dispatch')
                self.store.check();unit=units(key,list(args[0].shape));add_counts(self.counts,'attempted',key,unit)
                require(all(self.counts['attempted'][key][a]<=v for a,v in protocol()['caps'][key].items()),'pre-dispatch actual module cap')
                rec=dict(index=i,branch=self.where[0],phase=self.where[1],module=key,units=unit,status='attempted')
                self.calls.append(rec);self.pending=rec;self.store.emit('module_dispatch_intent',call=rec)
                rec['inputs']=tree((args,kwargs),save,torch.is_tensor);self.store.emit('module_inputs_saved',call_index=i,inputs=rec['inputs'])
            return hook
        def post(key):
            def hook(module,args,kwargs,output):
                rec=self.pending;require(rec is not None and rec['module']==key,'actual top module return')
                add_counts(self.counts,'returned',key,rec['units']);rec['status']='returned'
                self.store.emit('module_returned',call_index=rec['index'])
                rec['outputs']=tree(output,save,torch.is_tensor);self.store.emit('module_outputs_saved',call_index=rec['index'],outputs=rec['outputs'])
                self.pending=None
            return hook
        for key in KEYS:
            self.handles.append(models[key].register_forward_pre_hook(pre(key),with_kwargs=True))
            self.handles.append(models[key].register_forward_hook(post(key),with_kwargs=True))

    def close(self):
        errors=[]
        for h in self.handles:
            try:h.remove()
            except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
        return errors


class CudaBackend:
    def __init__(self,manifest,trace):
        # bridge verifies its complete own loaded-source closure before torch.
        import torch
        require(torch.cuda.is_available(),'explicit HOST CUDA, no fallback')
        require(torch.cuda.mem_get_info()[0]>=protocol()['initial_cuda_free_floor_bytes'],'initial4GiB CUDA free scratch policy')
        torch.cuda.reset_peak_memory_stats()  # includes load/model construction, not only forwards
        loaded=bridge.load_host_cuda(manifest['bridge_prepared'],manifest['bridge_prepare_sha256'],'MAIN_P5_RAW164_JEPA_HOSTCUDA_ONLY')
        from app.plan_common.datasets.transforms import make_transforms
        self.torch=torch;self.models=loaded['models'];self.before=loaded['model_before'];self.strict=loaded['strict'];self.sources=loaded['sources']
        self.math=loaded['attention_context_factory'];self.trace=trace
        self.transform=make_transforms(img_size=224,random_horizontal_flip=False,random_resize_aspect_ratio=(1.,1.),
            random_resize_scale=(1.,1.),reprob=0.,auto_augment=False,motion_shift=False)
        # Include newly imported transform modules, not only loader-time imports.
        for name,module in tuple(sys.modules.items()):
            file=getattr(module,'__file__',None)
            if file and name.split('.')[0] in ('app','src','dinov2'):
                path=str(Path(file).resolve());require(path in manifest['bindings'] and sha(path)==manifest['bindings'][path],'all actual native/preprocessing imports sealed')
                self.sources[name]=path
        trace.install(self.models,torch)

    def encode(self,rgb):
        t=self.torch;pixels=t.from_numpy(rgb.copy()).permute(0,3,1,2).to('cuda',t.float32)/255.
        return self.models['official_encoder'](self.transform(pixels),is_training=True)['x_norm_patchtokens']

    def predict(self,item):
        t=self.torch;self.trace.where=(item['index'],'prediction')
        with t.inference_mode(),self.math():
            patch=self.encode(item['rgb']);require(tuple(patch.shape)==(3,256,384),'actual context patch3')
            p=t.from_numpy(item['normalized_proprio']).unsqueeze(0).to('cuda')
            features=self.models['proprio_encoder'](p);require(tuple(features.shape)==(1,3,1,16),'actual proprio embedding')
            pred=self.models['predictor'](patch.reshape(1,3,1,16,16,384),t.from_numpy(item['packets']).unsqueeze(0).to('cuda'),features.expand(1,3,256,16))
            require(tuple(pred[0].shape)==(1,3,256,384),'actual full3 prediction, no extra replay')
        return self.trace.calls[-1]['outputs']['$tuple'][0]['$array']

    def label(self,index,rgb):
        self.trace.where=(index,'label')
        with self.torch.inference_mode(),self.math():
            patch=self.encode(rgb[None]);require(tuple(patch.shape)==(1,256,384),'actual label patch1')

    def receipt(self):
        # Reuse the original loader's exact state-hash function, no redefinition.
        from experiments.spectral_reachability.p5_jepa_real_forward_20261027_v4 import model_receipt
        return {k:model_receipt(m) for k,m in self.models.items()}

    def memory(self):
        self.torch.cuda.synchronize()
        free,total=self.torch.cuda.mem_get_info()
        return dict(cuda_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),
            cuda_free_bytes=free,cuda_total_bytes=total,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def predictions_seal(store,manifest,rows):
    require(len(rows)==164 and all(r['prediction_status']=='sealed' and r['prediction'] is not None for r in rows),'all164 genuine full outputs before label access')
    require(len({r['prediction']['file'] for r in rows})==164,'164 unique original module outputs, not shared receipts')
    seal=dict(schema=bridge.NAME+'.prediction_seal',prepared_semantic_sha256=digest(manifest['bridge']),
        predictions=[dict(index=r['index'],path=str(store.path/r['prediction']['file']),sha256=r['prediction']['sha256']) for r in rows])
    rec=store.put('predictions_seal.json',seal);store.emit('predictions_sealed',receipt=rec)
    return rec


def load_array(directory,rec):
    path=Path(directory)/rec['file']
    require(Path(rec['file']).name==rec['file'] and not path.is_symlink() and path.is_file() and path.stat().st_size==rec['file_bytes']<=ARRAY_CAP and sha(path)==rec['sha256'],'actual flat NPY SHA/bytes')
    a=np.load(path,allow_pickle=False)
    require(a.dtype==np.float32 and rec['dtype']=='float32' and list(a.shape)==rec['shape'] and a.nbytes==rec['nbytes'] and np.isfinite(a).all(),'full original finite float32 NPY')
    return a


def arithmetic(pred,context,target):
    require(pred.shape==(1,3,256,384) and context.shape==(3,256,384) and target.shape==(1,256,384),'fixed factual arrays')
    delta=pred[:,-1].astype(np.float64)-target.astype(np.float64)
    persist=context[-1:].astype(np.float64)-target.astype(np.float64)
    return dict(successor_MSE=float(np.mean(delta*delta,dtype=np.float64)),persistence_MSE=float(np.mean(persist*persist,dtype=np.float64)))


def pair_scores(rows,directory):
    result=[]
    for spec in protocol()['pairs']:
        lo,hi=(rows[spec[k]] for k in ('minus_index','plus_index'))
        row=dict(spec,missing=[r['index'] for r in (lo,hi) if r['score'] is None],plus_minus_successor_MSE=None,
            predicted_response_l2=None,observed_response_l2=None,response_cosine=None)
        if not row['missing']:
            p=[load_array(directory,r['prediction'])[:,-1].astype(np.float64) for r in (lo,hi)]
            o=[load_array(directory,r['label_patch']).astype(np.float64) for r in (lo,hi)]
            dp=(p[1]-p[0]).ravel();do=(o[1]-o[0]).ravel();pn=float(np.linalg.norm(dp));on=float(np.linalg.norm(do))
            row.update(plus_minus_successor_MSE=hi['score']['successor_MSE']-lo['score']['successor_MSE'],predicted_response_l2=pn,observed_response_l2=on,
                response_cosine=None if pn==0 or on==0 else float(np.dot(dp,do)/(pn*on)))
        result.append(row)
    return result


def empty_rows(q):
    return [dict(index=r['index'],root_slot=r['root_slot'],condition=r['condition'],task_id=r['task_id'],selection_digest=r['selection_digest'],
        prediction_status='not_started',label_status='not_started',prediction=None,context_patch=None,label_patch=None,score=None) for r in q['requests']]


def input_hashes(item):
    return {k:hashlib.sha256(np.ascontiguousarray(item[k]).tobytes()).hexdigest() for k in ('rgb','normalized_proprio','packets')}


def array_records(value):
    if not isinstance(value,dict):return []
    if '$array' in value:return [value['$array']]
    if '$dict' in value:return [r for v in value['$dict'].values() for r in array_records(v)]
    key='$tuple' if '$tuple' in value else '$list'
    return [r for v in value[key] for r in array_records(v)]


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'explicit MAIN host CUDA authority before work')
    store=Store(output);trace=Trace(store);rows=[];backend=None;manifest=None
    report=dict(schema=NAME+'.report',status='partial_or_failed',protocol=protocol(),errors=[],rows=rows,counts=trace.counts,
        calls=trace.calls,P1_admission=False,P5_admission=False,source_errors=[],prediction_seal=None)
    try:
        manifest=verify(prepared,expected);q=manifest['bridge'];rows.extend(empty_rows(q))
        binding=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected)
        report['execution_binding']=binding;store.put('execution_binding.json',binding);store.emit('run_started',execution_binding=binding,protocol=protocol())
        tick=time.monotonic();backend=CudaBackend(manifest,trace)
        report.update(model_before=backend.before,strict_loads=backend.strict,imported_sources=backend.sources,load_seconds=time.monotonic()-tick)
        tick=time.monotonic()
        for item,row in zip(bridge.host_inputs(q),rows):
            store.check();store.emit('source_history_ready',branch=row['index'],input_sha256=input_hashes(item))
            row['prediction_status']='attempted';row['prediction']=backend.predict(item)
            row['context_patch']=trace.calls[3*row['index']]['outputs']['$dict']['x_norm_patchtokens']['$array']
            row['prediction_status']='sealed';store.emit('prediction_saved',branch=row['index'],prediction=row['prediction'],context_patch=row['context_patch'])
        report['prediction_phase_seconds']=time.monotonic()-tick
        report['model_after_predictions']=backend.receipt();require(report['model_after_predictions']==backend.before,'model unchanged before label phase')
        store.emit('model_checked_before_labels',model_after_predictions=report['model_after_predictions'])
        report['prediction_seal']=predictions_seal(store,manifest,rows)
        # bridge checks every original full3 NPY before decoding any successor.
        tick=time.monotonic();store.emit('label_decode_intent',count=164,prediction_seal_sha256=report['prediction_seal']['sha256'])
        labels=bridge.labels_after_seal(q,store.path/'predictions_seal.json',report['prediction_seal']['sha256'])
        store.emit('labels_unlocked',count=len(labels),prediction_seal_sha256=report['prediction_seal']['sha256'])
        for item,row in zip(labels,rows):
            store.check();store.emit('label_pixels_ready',branch=row['index'],pixel_sha256=hashlib.sha256(item['rgb'].tobytes()).hexdigest())
            row['label_status']='attempted';backend.label(item['index'],item['rgb'])
            row['label_patch']=trace.calls[-1]['outputs']['$dict']['x_norm_patchtokens']['$array'];row['label_status']='sealed'
            row['score']=arithmetic(load_array(store.path,row['prediction']),load_array(store.path,row['context_patch']),load_array(store.path,row['label_patch']))
            store.emit('score_saved',branch=row['index'],label_patch=row['label_patch'],score=row['score'])
        report['label_and_score_seconds']=time.monotonic()-tick
        require(trace.counts['attempted']==trace.counts['returned']==protocol()['caps'] and len(trace.calls)==656,'all actual paid calls exact')
    except BaseException as exc:
        report['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        try:store.emit('failure',final=True,error=report['errors'][-1])
        except BaseException as extra:report['errors'].append(dict(type=type(extra).__name__,message='durable failure: '+str(extra)))
    finally:
        report['errors'].extend(dict(type='hook_cleanup',message=e) for e in trace.close())
        if not rows:
            rows.extend(dict(index=i,root_slot=bridge.factor.ORDER[i//41][0],condition=bridge.factor.ORDER[i//41][1],
                prediction_status='not_started',label_status='not_started',prediction=None,context_patch=None,label_patch=None,score=None) for i in range(164))
        if backend is not None:
            try:
                report['model_after']=backend.receipt();report['model_unchanged']=report['model_after']==backend.before
                require(report['model_unchanged'],'final unchanged model')
            except BaseException as exc:report['errors'].append(dict(type=type(exc).__name__,message='model final: '+str(exc)))
            try:report['memory']=backend.memory()
            except BaseException as exc:report['errors'].append(dict(type=type(exc).__name__,message='memory final: '+str(exc)))
        else:report['memory']=None  # failed initialization: no invented CUDA peaks
        if manifest is not None:
            for file,h in manifest['bindings'].items():
                try:require(sha(file)==h,'final source/asset SHA unchanged')
                except BaseException as exc:report['source_errors'].append(dict(path=file,error=str(exc)))
            report['source_rehash_bytes_per_pass']=sum(Path(f).stat().st_size for f in manifest['bindings'] if Path(f).is_file())
        try:report['pairs']=pair_scores(rows,store.path)
        except BaseException as exc:
            report['errors'].append(dict(type=type(exc).__name__,message='pair finalization: '+str(exc)))
            report['pairs']=[dict(s,missing=[s['minus_index'],s['plus_index']],plus_minus_successor_MSE=None,
                predicted_response_l2=None,observed_response_l2=None,response_cosine=None) for s in protocol()['pairs']]
        report['elapsed_seconds']=time.monotonic()-store.started
        if report['elapsed_seconds']>=WALL:report['errors'].append(dict(type='wall_cap',message='complete job including final checks exceeded900s'))
        if not report['errors'] and not report['source_errors'] and all(r['score'] is not None for r in rows):report['status']='fixed164_actual15_conditioned_factual_H1_recorded'
        try:store.emit('run_finished',final=True,status=report['status'],rows_digest=digest(rows),counts=trace.counts,errors=report['errors'])
        except BaseException as exc:
            report['status']='partial_or_failed';report['errors'].append(dict(type=type(exc).__name__,message='finish journal: '+str(exc)))
        report.update(artifacts=store.inventory(),journal_chain_tail=store.previous,output_bytes_before_report=store.bytes,
            actual_tensor_io_bytes=store.tensor_bytes,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        store.put('report.json',report,final=True)
    return dict(status=report['status'],path=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def encoder_reference(rgb):
    # Torch CUDA scalar division by255 uses float32 reciprocal multiplication.
    x=rgb.transpose(0,3,1,2).astype(np.float32)*np.float32(1./255.)
    mean=np.array([.485,.456,.406],np.float32)[None,:,None,None]
    std=np.array([.229,.224,.225],np.float32)[None,:,None,None]
    return (x-mean)/std


def consume(output,expected):
    directory=Path(output).absolute();r=bridge.read(directory/'report.json',expected,META)
    require(directory.parent==ROOT/'results' and directory.name.startswith(NAME+'_'),'own new run directory')
    require(r['schema']==NAME+'.report' and r['protocol']==protocol() and len(r['rows'])==164 and
        r['P1_admission'] is False and r['P5_admission'] is False,'exact new own report and full denominator/closed admission')
    inv=r['artifacts'];require(set(inv)|{'report.json'}=={p.name for p in directory.iterdir()},'complete exact directory inventory')
    require(sum(v['file_bytes'] for v in inv.values())==r['output_bytes_before_report'] and
        r['output_bytes_before_report']+(directory/'report.json').stat().st_size<=CAP,'actual output cap including report')
    for file,rec in inv.items():
        require(Path(file).name==file and not (directory/file).is_symlink() and sha(directory/file)==rec['sha256'] and
            (directory/file).stat().st_size==rec['file_bytes'],'all artifact SHA/bytes')
    if 'execution_binding' not in r:
        require(r['status']=='partial_or_failed' and r['errors'] and not r['calls'] and all(x['score'] is None for x in r['rows']),
            'invalid-prepare partial has no certified inference')
        if 'journal.jsonl' in inv:list(journal(directory/'journal.jsonl'))
        return dict(schema=NAME+'.consume',status='blocked_partial_capture',report_sha256=expected,requests=164,pairs=80,P5_admission=False,
            limitation='preflight failed before validated input binding; only saved artifact integrity certified')
    binding=r['execution_binding'];m=verify(binding['prepared'],binding['prepare_sha256']);q=m['bridge']
    require(bridge.read(directory/'execution_binding.json')==binding,'durable own execution binding')
    events=list(journal(directory/'journal.jsonl'));require(events and events[-1]['event_sha256']==r['journal_chain_tail'],'full journal tail')
    require(events[0]['kind']=='run_started' and events[0]['execution_binding']==binding and events[0]['protocol']==protocol(),'actual run start')
    require(events[-1]['kind']=='run_finished' and events[-1]['status']==r['status'] and events[-1]['rows_digest']==digest(r['rows']) and
        events[-1]['counts']==r['counts'] and events[-1]['errors']==r['errors'],'durable complete terminal summary')
    # All partial artifacts are checked, but no scientific negative/success inferred.
    if r['status']=='partial_or_failed':
        require(r['errors'] or r['source_errors'],'partial must retain actual error')
        return dict(schema=NAME+'.consume',status='blocked_partial_capture',report_sha256=expected,requests=164,pairs=80,P5_admission=False,
            limitation='artifact/hashchain verified; unfinished module trace/scoring not certified')
    require(r['status']=='fixed164_actual15_conditioned_factual_H1_recorded' and not r['errors'] and not r['source_errors'] and
        r['model_before']==r['model_after_predictions']==r['model_after'] and r['model_unchanged'] is True and r['elapsed_seconds']<WALL,'complete unmodified sources/models within wall')
    require(r['model_before']==bridge.read(bridge.PARENT,bridge.PARENT_SHA)['expected_model_receipts'],'original loader expected model state')
    memory=r['memory']
    require(isinstance(memory,dict) and set(memory)=={'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','cuda_free_bytes','cuda_total_bytes','host_peak_rss_bytes'} and
        all(type(v) is int and v>=0 for v in memory.values()) and
        memory['cuda_free_bytes']<=memory['cuda_total_bytes'] and memory['cuda_peak_allocated_bytes']<=memory['cuda_peak_reserved_bytes'],
        'actual complete CUDA and host memory receipt')
    require(all(v['exact_match'] and not any(v.get(k) for k in ('missing_keys','unexpected_keys','shape_mismatches','dtype_mismatches')) for v in r['strict_loads'].values()),'strict native loaded schema')
    require(set(r['strict_loads'])=={'predictor','proprio_encoder'} and
        {'app.plan_common.models.AdaLN_vit','app.plan_common.models.prop_embedding','dinov2.models.vision_transformer','app.plan_common.datasets.transforms'}<=set(r['imported_sources']),
        'nonempty exact model and preprocessing import closure')
    require(all(path in m['bindings'] and sha(path)==m['bindings'][path] for path in r['imported_sources'].values()),'actual imported source identities')
    calls=r['calls'];require(len(calls)==656 and [c['index'] for c in calls]==list(range(656)) and
        [(c['branch'],c['phase'],c['module']) for c in calls]==call_schedule(),'all fixed actual top module calls')
    io_records=[rec for c in calls for part in ('inputs','outputs') for rec in array_records(c[part])]
    require([rec['file'] for rec in io_records]==[f'tensor_{i:05d}.npy' for i in range(len(io_records))],'each module IO tensor has unique original durable slot; no replay/reuse receipts')
    counts=empty_counts();names={'execution_binding.json','journal.jsonl','predictions_seal.json'};tensor_sizes={}
    def load(rec):
        names.add(rec['file']);value=load_array(directory,rec);tensor_sizes[rec['file']]=value.nbytes;return value
    cursor=1
    def expect(kind):
        nonlocal cursor
        require(cursor<len(events) and events[cursor]['kind']==kind,'strict prediction/label chronology: '+kind)
        e=events[cursor];cursor+=1;return e
    inputs=iter(bridge.host_inputs(q))
    for i in range(164):
        req=q['requests'][i];row=r['rows'][i];item=next(inputs)
        e=expect('source_history_ready');require(e['branch']==i and e['input_sha256']==input_hashes(item),'actual source history input byte linkage')
        require(all(row[k]==req[k] for k in ('index','root_slot','condition','task_id','selection_digest')) and row['prediction_status']==row['label_status']=='sealed','full branch identity and endpoint completion')
        restored=[]
        for call in calls[3*i:3*i+3]:
            check_call(call,counts,expect,load);restored.append((restore(call['inputs'],load),restore(call['outputs'],load)))
        (encargs,enckw),encout=restored[0];(propargs,propkw),propout=restored[1];(predargs,predkw),predout=restored[2]
        require(enckw=={'is_training':True} and len(encargs)==1 and np.array_equal(encargs[0],encoder_reference(item['rgb'])),'actual encoder source RGB/preprocessing bitwise, no tolerance')
        require(propkw==predkw=={} and len(propargs)==1 and np.array_equal(propargs[0],item['normalized_proprio'][None]),'original pre-action normalized proprio')
        require(propout.shape==(1,3,1,16) and len(predargs)==3 and np.array_equal(predargs[0],encout['x_norm_patchtokens'].reshape(1,3,1,16,16,384)) and
            np.array_equal(predargs[1],item['packets'][None]) and np.array_equal(predargs[2],np.broadcast_to(propout,(1,3,256,16))),'actual full predictor input linkage; no future labels')
        require(row['prediction']==calls[3*i+2]['outputs']['$tuple'][0]['$array'] and row['context_patch']==calls[3*i]['outputs']['$dict']['x_norm_patchtokens']['$array'] and
            predout[0].shape==(1,3,256,384),'whole3 original prediction artifact association')
        e=expect('prediction_saved');require(e['branch']==i and e['prediction']==row['prediction'] and e['context_patch']==row['context_patch'],'durable row prediction')
    checked=expect('model_checked_before_labels');require(checked['model_after_predictions']==r['model_after_predictions'],'durable pre-label unchanged model barrier')
    sealed=expect('predictions_sealed');require(sealed['receipt']==r['prediction_seal'] and sealed['receipt']['sha256']==sha(directory/'predictions_seal.json'),'actual seal after all164 predictors')
    require(bridge.read(directory/'predictions_seal.json')==dict(schema=bridge.NAME+'.prediction_seal',prepared_semantic_sha256=digest(q),
        predictions=[dict(index=row['index'],path=str(directory/row['prediction']['file']),sha256=row['prediction']['sha256']) for row in r['rows']]),
        'seal exactly equals164 actual original module outputs')
    decode=expect('label_decode_intent');require(decode['count']==164 and decode['prediction_seal_sha256']==sealed['receipt']['sha256'],'durable all164 label-read dispatch only after seal')
    labels=bridge.labels_after_seal(q,directory/'predictions_seal.json',sealed['receipt']['sha256'])
    unlocked=expect('labels_unlocked');require(unlocked['count']==164 and unlocked['prediction_seal_sha256']==sealed['receipt']['sha256'],'full label phase unlocked by seal')
    for i,item in enumerate(labels):
        e=expect('label_pixels_ready');require(e['branch']==i and e['pixel_sha256']==hashlib.sha256(item['rgb'].tobytes()).hexdigest(),'all exact original factual label pixels')
        row=r['rows'][i];call=calls[492+i];check_call(call,counts,expect,load)
        args,kwargs=restore(call['inputs'],load);out=restore(call['outputs'],load)
        require(len(args)==1 and kwargs=={'is_training':True} and np.array_equal(args[0],encoder_reference(item['rgb'][None])) and
            row['label_patch']==call['outputs']['$dict']['x_norm_patchtokens']['$array'],'actual own factual label RGB encode only after seal')
        score=arithmetic(load(row['prediction']),load(row['context_patch']),out['x_norm_patchtokens'])
        require(score==row['score'],'independent exact float64 scoring, no tolerance')
        e=expect('score_saved');require(e['branch']==i and e['score']==score and e['label_patch']==row['label_patch'],'durable complete score receipt')
    require(cursor==len(events)-1 and counts==r['counts'] and counts['attempted']==counts['returned']==protocol()['caps'],'complete event census and API/batch/frame axes')
    require(names==set(inv),'all artifact roles accounted, no unexamined arrays')
    require(sum(tensor_sizes.values())==r['actual_tensor_io_bytes']<=protocol()['neural_tensor_io_upper_bound_bytes'],'actual complete module tensor byte census')
    require(pair_scores(r['rows'],directory)==r['pairs'] and len(r['pairs'])==80,'all80 independent differential recomputation')
    for file,h in m['bindings'].items():require(sha(file)==h,'sources stable after scoring consume')
    return dict(schema=NAME+'.consume',status='all164_factual_H1_and80_pairs_verified',report_sha256=expected,
        requests=164,pairs=80,artifacts=len(inv),counts=counts,neural_calls=0,physics_calls=0,P1_admission=False,P5_admission=False,
        diagnostic=protocol()['diagnostic'],matched_cross_model_performance=False,RGB_only=False)


def check_call(call,counts,expect,load):
    index=call['index'];require(call['status']=='returned','actual returned module')
    intent=expect('module_dispatch_intent');require(intent['call']=={k:call[k] for k in ('index','branch','phase','module','units')}|{'status':'attempted'},'exact actual pre-dispatch receipt')
    a=expect('module_inputs_saved');b=expect('module_returned');c=expect('module_outputs_saved')
    require(a['call_index']==b['call_index']==c['call_index']==index and a['inputs']==call['inputs'] and c['outputs']==call['outputs'],'complete ordered actual IO')
    args,kwargs=restore(call['inputs'],load);restore(call['outputs'],load)
    require(call['units']==units(call['module'],list(args[0].shape)),'independent batch/frame sample count')
    for phase in counts:add_counts(counts,phase,call['module'],call['units'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--bridge-prepared',required=True);p.add_argument('--bridge-prepare-sha',required=True);p.add_argument('--output',required=True)
    for name in ('verify','run'):
        p=sub.add_parser(name);p.add_argument('--prepared',required=True);p.add_argument('--prepare-sha',required=True)
        if name=='run':p.add_argument('--output',required=True);p.add_argument('--authorization',required=True)
    p=sub.add_parser('consume');p.add_argument('--output',required=True);p.add_argument('--report-sha',required=True);p.add_argument('--receipt',required=True)
    a=parser.parse_args()
    if a.command=='prepare':value=write_new(a.output,build(a.bridge_prepared,a.bridge_prepare_sha))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);value=dict(status='prepared_closure_verified',neural_calls=0)
    elif a.command=='run':value=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:value=write_new(a.receipt,consume(a.output,a.report_sha))
    print(json.dumps(value,sort_keys=True,allow_nan=False))
    return 2 if value.get('status') in ('partial_or_failed','blocked_partial_capture') else 0


if __name__=='__main__':raise SystemExit(main())
