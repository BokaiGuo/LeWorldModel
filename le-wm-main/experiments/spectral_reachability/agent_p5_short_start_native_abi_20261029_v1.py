"""Actual0 saved-IO resolver and pure T1/T2 guard fixtures, NOT a CUDA runner."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p5_roll_history_preconditions_20261029_v1 as pre

NAME='agent_p5_short_start_native_abi_20261029_v1'
ROOT=pre.ROOT
require,sha,read,raw,exact=pre.require,pre.sha,pre.read,pre.raw,pre.exact
PRE_SHA='0892bf49b229759907c80f96e7b63de98c1cf3b8a6e6c463a7fd53afb308c0b7'
H2='agent_p5_raw164_jepa_h2_cuda_20261029_v2'
REPORT_SHA='e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579'
STRICT_SHA='783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea'
PREPARE_SHA=pre.PINS['H2'][1]


def pointer(value,path):
    for token in path.strip('/').split('/'):
        token=token.replace('~1','/').replace('~0','~')
        value=value[int(token)] if isinstance(value,list) else value[token]
    return value


def resolve_actual0(report,manifest,strict,directory,index):
    require(type(index) is int and 0<=index<164,'literal original branch index')
    require(report['schema']==H2+'.report' and manifest['schema']==H2+'.prepare' and strict['schema']==H2+'.consume' and
        strict['report_sha256']==REPORT_SHA and strict['status']=='all164_open_loop_H2_and80_pairs_verified' and strict['requests']==164 and strict['pairs']==80,
        'explicit original schema/strict reference, NOT rebuilt here')
    require(report['status']=='fixed164_root10_open_loop_H2_recorded' and report['execution_binding']['prepare_sha256']==PREPARE_SHA and len(manifest['bridge']['requests'])==len(report['rows'])==164 and
        len(report['calls'])==820 and report['errors']==report['source_errors']==[] and report['model_before']==report['model_after_predictions']==report['model_after'],
        'complete original H2 metadata and same original model states')
    req=manifest['bridge']['requests'][index];row=report['rows'][index]
    require(req['index']==row['index']==index and all(type(req[k]) is type(row[k]) and req[k]==row[k] for k in ('root_slot','condition','task_id','selection_digest')) and
        (req['root_slot'],req['condition'])==((0,'off'),(0,'on'),(1,'off'),(1,'on'))[index//41], 'same original branch/root/task/selection')
    require(req['context_global_frames']==[0,5,10] and req['successor_global_frames']==[15,20] and
        len(req['images'])==3 and [(x['phase'],x['step']) for x in req['images']]==[('warmup',0),('warmup',5),('warmup',10)] and
        all(x['condition']==req['condition'] and x['overlay_index']==int(req['condition']=='on') for x in req['images']),
        'actual0 original pre-goal-bind history, not predicted15 or successor')
    refs={}
    def add(key,path,shape):
        rec=pointer(report,path)
        require(rec['shape']==list(shape) and rec['dtype']=='float32' and rec['nbytes']==4*int(np.prod(shape)) and
            Path(rec['file']).name==rec['file'] and report['artifacts'].get(rec['file'])=={k:rec[k] for k in ('sha256','file_bytes')},
            'actual input/encoder tensor pointer and original inventory SHA/bytes')
        refs[key]=dict(directory=str(Path(directory).absolute()),report_json_pointer=path,array=rec)
    c=4*index
    for j,key,unit in ((0,'official_encoder',dict(api=1,batch_samples=3,frame_samples=3)),
                       (1,'proprio_encoder',dict(api=1,batch_samples=1,frame_samples=3)),
                       (2,'predictor',dict(api=1,batch_samples=1,frame_samples=3))):
        call=report['calls'][c+j]
        require(call['index']==c+j and call['branch']==index and call['module']==key and call['phase']=='prediction' and
            call['status']=='returned' and call['units']==unit,'original native completed calls in fixed order')
    call=report['calls'][c+2];t=call['inputs']
    require(set(t)=={'$tuple'} and len(t['$tuple'])==2 and t['$tuple'][1]=={'$dict':{}} and
        set(t['$tuple'][0])=={'$tuple'} and len(t['$tuple'][0]['$tuple'])==3,'three positional native inputs, no kwargs')
    for j,(key,s) in enumerate(zip(('visual','action','proprio'),pre.SHAPES,strict=True)):
        add(key,f'/calls/{c+2}/inputs/$tuple/0/$tuple/{j}/$array',s)
    add('encoder_patch',f'/calls/{c}/outputs/$dict/x_norm_patchtokens/$array',(3,256,384))
    add('proprio_raw_input',f'/calls/{c+1}/inputs/$tuple/0/$tuple/0/$array',(1,3,4))
    add('proprio_encoded',f'/calls/{c+1}/outputs/$array',(1,3,1,16))
    require(row['context_patch']==refs['encoder_patch']['array'],'row/context/call original linkage')
    return dict(index=index,root_slot=req['root_slot'],condition=req['condition'],task_id=req['task_id'],selection_digest=req['selection_digest'],
        image0=req['images'][0],frames=[0],parent_frames=[0,5,10],native_parent_call=c+2,refs=refs,
        inherited_authority=dict(report_sha256=REPORT_SHA,prepare_sha256=PREPARE_SHA,strict_sha256=STRICT_SHA),
        slice='all three native inputs[:,0:1].copy(order=C), no encode or renormalize',
        source_normalization=dict(raw_proprio=req['raw_proprio'],commands=req['commands'],packets=req['packets'],
            normalized_proprio=req['normalized_proprio'],stats=manifest['bridge']['stats']),
        expected_model=report['model_before'],expected_strict_loads=report['strict_loads'],
        full_strict_rebuilt_here=False,native_IO_materialized_and_crosschecked=False,short_T_execution_certified=False)


def load_array(ref):
    rec=ref['array'];d=Path(ref['directory']);p=d/rec['file']
    require(d.is_absolute() and p.resolve()==p and not any(x.is_symlink() for x in (p,*p.parents)) and
        p.stat().st_size==rec['file_bytes']<=4*1024**2 and sha(p)==rec['sha256'],'whole original bounded NPY bytes/SHA')
    body=p.read_bytes();b=io.BytesIO(body);v=np.load(b,allow_pickle=False)
    require(b.tell()==len(body) and v.dtype==np.float32 and list(v.shape)==rec['shape'] and v.nbytes==rec['nbytes'] and np.isfinite(v).all(),
        'unaltered float32 finite original NPY, no trailing data')
    return v


def materialize_actual0(record):
    """Saved arrays only. Return in RAM; never write or execute predictors."""
    arrays={k:load_array(v) for k,v in record['refs'].items()};s=record['source_normalization']
    prop,actions=pre.normalized(s['raw_proprio'],s['commands'],s['stats'])
    require(exact(arrays['visual'],arrays['encoder_patch'].reshape(pre.SHAPES[0])) and
        exact(arrays['proprio'],np.broadcast_to(arrays['proprio_encoded'],pre.SHAPES[2])) and
        exact(arrays['proprio_raw_input'],prop[None]) and exact(prop,pre.f32(s['normalized_proprio'],(3,4))) and
        exact(arrays['action'],actions[None,:3]) and exact(actions,pre.f32(s['packets'],(4,10))),
        'actual full native encoder/prop/normalizer linkage, no recomputed feature substitute')
    result=tuple(arrays[k][:,0:1].copy(order='C') for k in ('visual','action','proprio'))
    return result,dict(index=record['index'],frames=[0],input_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in result],
        source_array_sha256={k:v['array']['sha256'] for k,v in record['refs'].items()},materialized_and_crosschecked=True,
        short_T_execution_certified=False,full_strict_rebuilt_here=False)


def design():
    return dict(schema=NAME+'.design',engineering_probe_only=True,fixed_branch=0,fixed_root_slot=0,fixed_condition='off',
        stages=[dict(T=1,frames=[0],predicts=5),dict(T=2,frames=[0,5],predicts=10)],
        second_stage='actual0 plus first NEW sealed predicted5 visual/proprio; original raw1..5 and6..10. No actual5 feedback.',
        caps=dict(predictor=dict(api=2,batch_samples=2,frame_samples=3),official_encoder=dict(api=0,batch_samples=0,frame_samples=0),
            proprio_encoder=dict(api=0,batch_samples=0,frame_samples=0),AD=0,physics=0),
        tensor_IO_effective_bytes=2457720,tensor_IO_cap_bytes=4*1024**2,output_cap_bytes=16*1024**2,
        wall_seconds=300,wall_includes='verify/load/dispatch/final model+source hashes',disk_floor_bytes=200*1024**2,
        initial_CUDA_free_floor_bytes=4*1024**3,loader='agent_p5_raw164_jepa_input_bridge_20261029_v1.load_host_cuda',
        loader_authorization='MAIN_P5_RAW164_JEPA_HOSTCUDA_ONLY',loader_source_sha256='d70352f2fc574ca8a226522fc3dda9f015305894c83a1d1826f05808c005da0f',
        loader_manifest='original bridge_prepared/bridge_prepare_sha256 from sealed H2 manifest',
        load_guard='actual loaded model receipts + strict schemas equal original H2, all actual imported source SHA bound; compare again after both calls',
        backend='HOSTCUDA frozen inference only, TF32off/MATH; no CPU fallback',
        new_native_hooks='NEW local guards accept only expected T1 then T2, no old full3 guard modification or global patch',
        durable_order='inputs/intents -> outputs/returns -> prediction seal -> final model/source checks; failures keep both declared slots',
        labels='none; no successor label or MSE in this engineering ABI check',
        production_ready=False,HOSTCUDA_executed=False,P1_admission=False,P5_admission=False)


class ShapeMeter:
    """Pure hook-contract model. Descriptors are NOT authenticated real tensors."""
    def __init__(self):
        self.attempted=dict(api=0,batch_samples=0,frame_samples=0);self.returned=dict(self.attempted);self.pending=None;self.device=None

    def before(self,module,shapes,devices,dtypes,requires_grad,inference,TF32off,MATH):
        require(self.pending is None and module=='predictor' and self.attempted['api']<2,'only two ordered predictor dispatches, no nested/encoder/retry')
        t=self.attempted['api']+1
        require(shapes==[(1,t,1,16,16,384),(1,t,10),(1,t,256,16)] and len(devices)==len(dtypes)==len(requires_grad)==3 and
            all(x=='cuda' or x.startswith('cuda:') for x in devices) and len(set(devices))==1 and
            (self.device is None or devices[0]==self.device) and
            dtypes==['float32']*3 and requires_grad==[False]*3 and inference is TF32off is MATH is True,
            'actual hook adapter must derive CUDA/shape/dtype/frozen/math descriptors from real tensors')
        self.pending=t;self.device=devices[0]
        for k,v in dict(api=1,batch_samples=1,frame_samples=t).items():self.attempted[k]+=v
        return dict(stage=t,units=dict(api=1,batch_samples=1,frame_samples=t))

    def after(self,shapes,devices,dtypes):
        t=self.pending;require(t is not None,'return must have durable attributable dispatch')
        require(shapes==[(1,t,256,384),None,(1,t,256,16)] and len(devices)==len(dtypes)==2 and
            devices==[self.device]*2 and dtypes==['float32']*2,'complete native visual/None/proprio output on original device')
        for k,v in dict(api=1,batch_samples=1,frame_samples=t).items():self.returned[k]+=v
        self.pending=None


def catalog(report,report_sha,prepared,prepare_sha,strict_receipt,strict_sha):
    require(sha(pre.__file__)==PRE_SHA and (report_sha,prepare_sha,strict_sha)==(REPORT_SHA,PREPARE_SHA,STRICT_SHA),'frozen helper and fixed actual external authorities')
    r=read(report,report_sha);p=read(prepared,prepare_sha);c=read(strict_receipt,strict_sha)
    require(r['execution_binding']['prepared']==str(Path(prepared).absolute()),'exact original prepared path binding')
    rows=[resolve_actual0(r,p,c,Path(report).absolute().parent,i) for i in range(164)]
    return dict(schema=NAME+'.catalog',status='all164_actual0_IO_pointers_resolved_not_executed',rows=rows,design=design(),
        authority=dict(report_sha256=report_sha,prepare_sha256=prepare_sha,strict_sha256=strict_sha),
        production_ready=False,arrays_decoded=False,upstream_full_strict_rebuilt=False,new_neural_calls=0,new_physics_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('report','report-sha','prepared','prepare-sha','strict-receipt','strict-sha','output'):p.add_argument('--'+k,required=True)
    q=vars(p.parse_args());output=q.pop('output')
    # Reuse only exclusive metadata codec/writer, not prior prepare or consumer.
    value=catalog(**q);target=Path(output).absolute()
    require(target.parent==ROOT/'results' and target.name.startswith(NAME+'_') and not target.exists(),'new own catalog only')
    body=raw(value);require(len(body)<=16*1024**2 and not any(x.is_symlink() for x in (target,*target.parents)),'bounded own metadata')
    import os
    import shutil
    require(shutil.disk_usage(target.parent).free>=200*1024**2+len(body),'200MiB disk floor')
    with target.open('xb') as f:require(f.write(body)==len(body),'complete catalog');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=value['status'],path=str(target),sha256=sha(target)),sort_keys=True))


if __name__=='__main__':main()
