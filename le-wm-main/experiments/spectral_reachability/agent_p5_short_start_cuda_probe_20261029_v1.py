"""MAIN-only branch0 HOSTCUDA T1->T2 ABI probe; no label/score/physics."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from experiments.spectral_reachability import agent_p5_short_start_native_abi_20261029_v1 as abi
from experiments.spectral_reachability import agent_p5_raw164_jepa_h2_cuda_20261029_v2 as codec

NAME='agent_p5_short_start_cuda_probe_20261029_v1'
ROOT=abi.ROOT
AUTH='MAIN_P5_SHORT_START_T1_T2_HOSTCUDA_ONLY'
CATALOG=ROOT/'results/agent_p5_short_start_native_abi_20261029_v1_catalog_MAIN01.json'
CATALOG_SHA='1c49b5eba91619a6798a9d387c524b6e7d8658c196a9ac75f1ea9a65226c2678'
ACTUAL0_SHA=['d6f4eae94431f1926df413adec5ecc34597dba54002a46dfc773f362f5efbf01',
 'cd767833f90a393ee0ecc5422cf72c71caee1d7c1f5667a03048dd7f6ecb1d84',
 '3058e78213d6cbfd85ec89ddae80045a1b53cf13aac71f071551d0ca1b9f3743']
CAP,TENSOR_CAP,FLOOR,RESERVE,WALL=16*1024**2,4*1024**2,200*1024**2,2*1024**2,300
PINS={abi.__name__:'f4749f6b7ec6b890523a02c95d1cb27754b8fd4cd78832159a1cbe70e67e90bd',
 abi.pre.__name__:abi.PRE_SHA,codec.__name__:'b661a7c27181ef29f8277e154006f9238ec1c94487c311c1065e4d42fcfc2241',
 codec.bridge.__name__:'d70352f2fc574ca8a226522fc3dda9f015305894c83a1d1826f05808c005da0f'}
require,sha,read,raw,exact=abi.require,abi.sha,abi.read,abi.raw,abi.exact
digest=codec.digest


def merge(bindings,extra):
    for path,h in extra.items():
        require(path not in bindings or bindings[path]==h,'conflict-free exact source/asset SHA')
        bindings[path]=h


def protocol():
    return dict(schema=NAME+'.protocol',branch=0,root_slot=0,condition='off',stages=[dict(T=1,frames=[0],target=5),dict(T=2,frames=[0,5],target=10)],
        caps=abi.design()['caps'],tensor_bytes=2457720,cap_bytes=CAP,tensor_cap_bytes=TENSOR_CAP,disk_floor_bytes=FLOOR,wall_seconds=WALL,
        backend='HOSTCUDA-only frozen inference, TF32off, requested MATH; reject observed fused attention kernels',
        upstream_full_strict_rebuilt=False,new_labels=0,new_scores=0,new_encoder=0,new_AD=0,new_physics=0,new_training=0,
        engineering_only=True,P1_admission=False,P5_admission=False)


def build():
    for module in (abi,abi.pre,codec,codec.bridge):require(sha(module.__file__)==PINS[module.__name__],'frozen IO/codec/loader source')
    paths=dict(report=ROOT/'results'/(abi.H2+'_cuda_MAIN01/report.json'),prepared=ROOT/'results'/(abi.H2+'_prepare_MAIN01.json'),
        strict=ROOT/'results'/(abi.H2+'_consume_MAIN01.json'))
    r,p,c=(read(paths[k],h) for k,h in (('report',abi.REPORT_SHA),('prepared',abi.PREPARE_SHA),('strict',abi.STRICT_SHA)))
    require(r['execution_binding']['prepared']==str(paths['prepared']),'actual prepared path')
    record=abi.resolve_actual0(r,p,c,paths['report'].parent,0);_,extraction=abi.materialize_actual0(record)
    catalog=read(CATALOG,CATALOG_SHA)
    require(catalog['schema']==abi.NAME+'.catalog' and catalog['status']=='all164_actual0_IO_pointers_resolved_not_executed' and
        len(catalog['rows'])==164 and catalog['rows'][0]==record and catalog['authority']==record['inherited_authority'] and
        catalog['design']==abi.design() and extraction['input_sha256']==ACTUAL0_SHA,'actual MAIN catalog plus exact three actual0 input-byte SHA')
    bindings=deepcopy(p['bindings'])
    for k,h in (('report',abi.REPORT_SHA),('prepared',abi.PREPARE_SHA),('strict',abi.STRICT_SHA)):merge(bindings,{str(paths[k]):h})
    for ref in record['refs'].values():merge(bindings,{str(Path(ref['directory'])/ref['array']['file']):ref['array']['sha256']})
    merge(bindings,{str(CATALOG):CATALOG_SHA})
    for module in (abi,abi.pre,codec,codec.bridge):merge(bindings,{str(Path(module.__file__).resolve()):PINS[module.__name__]})
    for path in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):merge(bindings,{str(path):sha(path)})
    for path,h in bindings.items():require(sha(path)==h,'complete inherited/source/model bytes unchanged, no upstream full strict')
    return dict(schema=NAME+'.prepare',status='prepared_no_neural_execution',protocol=protocol(),record=record,extraction=extraction,
        bridge_prepared=p['bridge_prepared'],bridge_prepare_sha256=p['bridge_prepare_sha256'],bindings=bindings,
        actual_catalog=dict(path=str(CATALOG),sha256=CATALOG_SHA))


def verify(prepared,expected):
    m=read(prepared,expected);require(m==build(),'exact rebuilt source/actual0 preparation, no neural calls');return m


def write_new(path,value):
    p=Path(path).absolute();body=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and
        not any(x.is_symlink() for x in (p,*p.parents)) and len(body)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'exclusive own bounded metadata')
    with p.open('xb') as f:require(f.write(body)==len(body),'complete metadata');f.flush();os.fsync(f.fileno())
    codec.bridge.physical.paired.fsync_dir(p.parent)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


class Store:
    def __init__(self,path,started=None):
        self.path=Path(path).absolute();self.started=time.monotonic() if started is None else started
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and not self.path.exists() and
            not any(x.is_symlink() for x in (self.path,*self.path.parents)) and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'exclusive new16MiB probe directory/floor')
        self.path.mkdir();self.artifacts={};self.total=0;self.tensor_bytes=0;self.seq=0;self.tail=None;self.array_index=0

    def check(self,size=0,final=False):
        require(self.total+size<=CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free>=FLOOR+size,'actual cap/floor/terminal reserve')
        require(final or time.monotonic()-self.started<WALL,'full job300s cap')

    def put(self,name,body,final=False,append=False):
        require(Path(name).name==name and name not in ('','.','..') and type(body) is bytes,'flat own bytes')
        self.check(len(body),final);p=self.path/name
        require(not p.is_symlink() and (not p.exists() or append and name=='journal.jsonl' and name in self.artifacts),'exclusive write/owned journal append')
        before=p.stat().st_size if p.exists() else 0
        try:
            with p.open('ab' if append else 'xb') as f:require(f.write(body)==len(body),'complete durable write');f.flush();os.fsync(f.fileno())
        finally:
            self.total+=(p.stat().st_size if p.exists() else 0)-before
            if p.exists():self.artifacts[name]=dict(sha256=sha(p),file_bytes=p.stat().st_size)
        rec=self.artifacts[name]
        codec.bridge.physical.paired.fsync_dir(self.path)
        return dict(file=name,**rec)

    def emit(self,kind,final=False,**fields):
        e=dict(sequence=self.seq,previous_sha256=self.tail,kind=kind,**deepcopy(fields));e['event_sha256']=digest(e)
        self.put('journal.jsonl',raw(e),final,append=True);self.tail=e['event_sha256'];self.seq+=1

    def array(self,a):
        require(a.dtype==np.float32 and np.isfinite(a).all() and self.tensor_bytes+a.nbytes<=TENSOR_CAP,'actual full finite float32 IO cap')
        b=io.BytesIO();np.save(b,np.ascontiguousarray(a),allow_pickle=False)
        rec=dict(self.put(f'tensor_{self.array_index:03d}.npy',b.getvalue()),shape=list(a.shape),dtype='float32',nbytes=a.nbytes)
        self.array_index+=1;self.tensor_bytes+=a.nbytes;return rec


def second_inputs(m,first_inputs,first_output):
    advanced=abi.pre.advance_history([0],first_inputs[0],first_inputs[2],first_output)
    # Source packets are saved actual native packets, not inferred padding.
    actions=abi.load_array(m['record']['refs']['action'])[:,:2].copy(order='C')
    require(exact(actions[:,:1],first_inputs[1]) and advanced['frames']==[0,5],'unchanged raw1..5 then original6..10')
    return advanced['visual'],actions,advanced['proprio']


class Trace:
    def __init__(self,store):
        self.store=store;self.guard=abi.ShapeMeter();self.calls=[];self.observed_returns=dict(api=0,batch_samples=0,frame_samples=0);self.handles=[];self.pending=None

    def install(self,models,torch):
        def save(x):
            require(torch.is_tensor(x) and x.is_cuda and x.dtype==torch.float32 and not x.requires_grad and torch.isfinite(x).all(),'full finite frozen CUDA tensor')
            return self.store.array(x.detach().contiguous().cpu().numpy())
        def pre(key):
            def hook(module,args,kwargs):
                self.store.check();require(not kwargs and len(args)==3,'native exact three args/no kwargs')
                require(all(torch.is_tensor(x) for x in args),'actual tensor args')
                witness=self.guard.before(key,[tuple(x.shape) for x in args],[str(x.device) for x in args],
                    [str(x.dtype).removeprefix('torch.') for x in args],[x.requires_grad for x in args],torch.is_inference_mode_enabled(),
                    not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,True)
                call=dict(index=len(self.calls),T=witness['stage'],units=witness['units'],status='attempted',
                    native_inputs=[dict(shape=list(x.shape),device=str(x.device),dtype=str(x.dtype).removeprefix('torch.'),requires_grad=x.requires_grad) for x in args],
                    inference=torch.is_inference_mode_enabled(),TF32off=not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                    requested_MATH=True)
                self.calls.append(call);self.pending=call;self.store.emit('dispatch_intent',call=deepcopy(call))
                call['inputs']=codec.tree((args,kwargs),save,torch.is_tensor);self.store.emit('inputs_saved',index=call['index'],inputs=call['inputs'])
            return hook
        def post(key):
            def hook(module,args,kwargs,out):
                call=self.pending;require(call is not None and key=='predictor','attributed actual native return')
                for k,v in call['units'].items():self.observed_returns[k]+=v
                call['status']='return_observed';self.store.emit('return_observed',index=call['index'])
                require(type(out) is tuple and len(out)==3 and out[1] is None and torch.is_tensor(out[0]) and torch.is_tensor(out[2]),'full native output tuple')
                xs=(out[0],out[2]);self.guard.after([tuple(out[0].shape),None,tuple(out[2].shape)],
                    [str(x.device) for x in xs],[str(x.dtype).removeprefix('torch.') for x in xs])
                call['native_outputs']=[dict(shape=list(x.shape),device=str(x.device),dtype=str(x.dtype).removeprefix('torch.'),requires_grad=x.requires_grad) for x in xs]
                call['outputs']=codec.tree(out,save,torch.is_tensor);call['status']='returned';self.store.emit('outputs_saved',index=call['index'],outputs=call['outputs'],native_outputs=call['native_outputs']);self.pending=None
            return hook
        for key,model in models.items():
            self.handles.append(model.register_forward_pre_hook(pre(key),with_kwargs=True))
            self.handles.append(model.register_forward_hook(post(key),with_kwargs=True))

    def close(self):
        errors=[]
        for handle in self.handles:
            try:handle.remove()
            except BaseException as e:errors.append(dict(type=type(e).__name__,message=str(e)))
        return errors


class CudaBackend:
    def __init__(self,m,trace):
        import torch
        require(torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=4*1024**3,'HOSTCUDA4GiB scratch, no CPU fallback')
        torch.cuda.reset_peak_memory_stats();self.torch=torch;self.trace=trace;self.attention_ops={}
        loaded=codec.bridge.load_host_cuda(m['bridge_prepared'],m['bridge_prepare_sha256'],'MAIN_P5_RAW164_JEPA_HOSTCUDA_ONLY')
        self.models=loaded['models'];self.before=loaded['model_before'];self.strict=loaded['strict'];self.sources=loaded['sources'];self.math=loaded['attention_context_factory']
        require(self.before==m['record']['expected_model'] and self.strict==m['record']['expected_strict_loads'],'same actual loaded state/schema as sealed H2')
        self.check_sources(m);trace.install(self.models,torch)

    def check_sources(self,m):
        for _,module in tuple(sys.modules.items()):
            file=getattr(module,'__file__',None)
            if file and module.__name__.split('.')[0] in ('app','src','dinov2'):
                path=str(Path(file).resolve());require(m['bindings'].get(path)==sha(path),'actual imported native source closure')
        require(all(m['bindings'].get(path)==sha(path) for path in self.sources.values()),'loader sources unchanged')

    def predict(self,args):
        t=self.torch
        from torch.utils._python_dispatch import TorchDispatchMode
        observed=self.attention_ops
        class AttentionGuard(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                name=str(func)
                require(not any(x in name.lower() for x in ('flash_attention','efficient_attention','cudnn_attention')),'observed fused attention rejected; do not bypass native guard')
                if 'attention' in name.lower():observed[name]=observed.get(name,0)+1
                return func(*args,**(kwargs or {}))
        with t.inference_mode(),self.math(),AttentionGuard():
            self.models['predictor'](*(t.from_numpy(x.copy()).to('cuda') for x in args))

    receipt=codec.CudaBackend.receipt
    memory=codec.CudaBackend.memory


def execute(m,prepared,expected,output,backend_factory=CudaBackend,started=None):
    store=Store(output,started);trace=Trace(store);backend=None;started=store.started
    binding=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected)
    r=dict(schema=NAME+'.report',status='partial_or_failed',protocol=protocol(),execution_binding=binding,errors=[],source_errors=[],
        stages=[dict(T=t,frames=f,prediction=None) for t,f in ((1,[0]),(2,[0,5]))],calls=trace.calls,
        model_before=None,model_after=None,P1_admission=False,P5_admission=False)
    try:
        store.put('execution_binding.json',raw(binding));store.emit('run_started',binding=binding,protocol=protocol())
        backend=backend_factory(m,trace);r.update(model_before=backend.before,strict_loads=backend.strict,imported_sources=backend.sources)
        args,extraction=abi.materialize_actual0(m['record']);require(extraction==m['extraction'],'same original actual0 extraction')
        r['extraction']=extraction
        for i in range(2):
            store.check();backend.predict(args)
            call=trace.calls[i];require(call['status']=='returned','actual full native output return required')
            r['stages'][i]['prediction']=deepcopy(call['outputs']);store.emit('stage_sealed',stage=i,outputs=call['outputs'])
            if i==0:
                out=codec.restore(call['outputs'],lambda rec:abi.load_array(dict(directory=str(store.path),array=rec)))
                args=second_inputs(m,args,out)
        require(trace.guard.attempted==trace.observed_returns==trace.guard.returned==dict(api=2,batch_samples=2,frame_samples=3) and
            store.tensor_bytes==2457720,'exact2 dispatch/return/validated calls and original effective IO bytes')
        seal=dict(schema=NAME+'.prediction_seal',prepare_sha256=expected,stages=r['stages'])
        r['prediction_seal']=store.put('prediction_seal.json',raw(seal));store.emit('all_predictions_sealed',seal=r['prediction_seal'])
        r['status']='branch0_T1_T2_native_ABI_recorded'
    except BaseException as e:r['errors'].append(dict(type=type(e).__name__,message=str(e)))
    finally:
        if backend is not None:
            try:
                r['model_after']=backend.receipt();require(r['model_after']==r['model_before'],'model state unchanged')
                backend.check_sources(m);r['memory']=backend.memory();r['attention_ops']=backend.attention_ops
            except BaseException as e:r['source_errors'].append(dict(type=type(e).__name__,message=str(e)))
        r['source_errors'].extend(trace.close())
        for path,h in m['bindings'].items():
            try:require(sha(path)==h,'final bound source/model/input SHA')
            except BaseException as e:r['source_errors'].append(dict(type=type(e).__name__,message=str(e),path=path))
        r.update(counts=dict(attempted=trace.guard.attempted,returned=trace.observed_returns,validated=trace.guard.returned),tensor_bytes=store.tensor_bytes,
            elapsed_seconds=time.monotonic()-started)
        if r['elapsed_seconds']>=WALL:r['errors'].append(dict(type='wall_cap',message='complete job exceeded300s'))
        if r['errors'] or r['source_errors']:r['status']='partial_or_failed'
        store.emit('run_finished',final=True,status=r['status'],calls_digest=digest(r['calls']),counts=r['counts'])
        r.update(artifacts=deepcopy(store.artifacts),journal_chain_tail=store.tail,output_bytes_before_report=store.total)
        store.put('report.json',raw(r),final=True)
    return dict(status=r['status'],report=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH and not os.environ.get('PYTHONPATH'),'explicit MAIN-only host authorization/no overlay')
    started=time.monotonic();m=verify(prepared,expected)
    # verify is part of the wall budget, not an unmetered warmup.
    return execute(m,prepared,expected,output,started=started)


def consume(output,expected):
    d=Path(output).absolute();r=read(d/'report.json',expected)
    require(d.parent==ROOT/'results' and d.name.startswith(NAME+'_') and not any(x.is_symlink() for x in (d,*d.parents)),'own original output')
    require(r['schema']==NAME+'.report' and r['protocol']==protocol() and len(r['stages'])==2,'both fixed stage denominators')
    require(set(r['artifacts'])|{'report.json'}=={p.name for p in d.iterdir()},'exact entire output inventory')
    for name,rec in r['artifacts'].items():
        p=d/name;require(Path(name).name==name and not p.is_symlink() and sha(p)==rec['sha256'] and p.stat().st_size==rec['file_bytes'],'whole artifact bytes/SHA')
    events=list(codec.journal(d/'journal.jsonl'));require(events[-1]['event_sha256']==r['journal_chain_tail'],'journal hashchain/tail')
    if r['status']=='partial_or_failed':
        require(r['errors'] or r['source_errors'],'actual retained failure')
        return dict(schema=NAME+'.consume',status='blocked_partial_probe',report_sha256=expected,stages=2,P1_admission=False,P5_admission=False)
    m=verify(r['execution_binding']['prepared'],r['execution_binding']['prepare_sha256'])
    require(r['status']=='branch0_T1_T2_native_ABI_recorded' and not r['errors'] and not r['source_errors'] and r['elapsed_seconds']<WALL and
        r['model_before']==r['model_after']==m['record']['expected_model'] and r['strict_loads']==m['record']['expected_strict_loads'],'complete unchanged original loaded state')
    args,extraction=abi.materialize_actual0(m['record']);require(extraction==r['extraction']==m['extraction'],'same exact actual0')
    require(len(r['calls'])==2 and r['counts']==dict(attempted=dict(api=2,batch_samples=2,frame_samples=3),returned=dict(api=2,batch_samples=2,frame_samples=3),validated=dict(api=2,batch_samples=2,frame_samples=3)), 'exact paid/returned/validated2 no retries')
    guard=abi.ShapeMeter();tensor_bytes=0;array_index=0
    names={'execution_binding.json','journal.jsonl','prediction_seal.json'}
    def load(rec):
        nonlocal tensor_bytes,array_index
        require(rec['file']==f'tensor_{array_index:03d}.npy' and r['artifacts'].get(rec['file'])=={k:rec[k] for k in ('sha256','file_bytes')},'all native arrays unique/ordered/inventoried')
        array_index+=1;names.add(rec['file']);tensor_bytes+=rec['nbytes'];return abi.load_array(dict(directory=str(d),array=rec))
    for i,c in enumerate(r['calls']):
        actual,kwargs=codec.restore(c['inputs'],load);out=codec.restore(c['outputs'],load);t=i+1
        require(c['index']==i and c['T']==t and c['status']=='returned' and kwargs=={} and all(exact(x,y) for x,y in zip(args,actual,strict=True)), 'full saved actual native input causal linkage')
        specs=c['native_inputs'];require(len(specs)==3 and all(s['shape']==list(x.shape) and s['dtype']==str(x.dtype) for s,x in zip(specs,actual,strict=True)), 'actual CUDA input metadata and saved NPY agree')
        guard.before('predictor',[x.shape for x in actual],[x['device'] for x in specs],[x['dtype'] for x in specs],
            [x['requires_grad'] for x in specs],c['inference'],c['TF32off'],c['requested_MATH'])
        require(type(out) is tuple and len(out)==3 and out[1] is None,'complete native result')
        specs=c['native_outputs'];require(len(specs)==2 and all(s['shape']==list(x.shape) and s['dtype']==str(x.dtype) and s['requires_grad'] is False for s,x in zip(specs,(out[0],out[2]),strict=True)), 'full output CUDA metadata and NPY agree')
        guard.after([out[0].shape,None,out[2].shape],[x['device'] for x in specs],[x['dtype'] for x in specs])
        require(r['stages'][i]==dict(T=t,frames=[0] if i==0 else [0,5],prediction=c['outputs']),'exact two stage output pointers')
        if i==0:args=second_inputs(m,args,out)
    require(names==set(r['artifacts']) and tensor_bytes==r['tensor_bytes']==2457720 and sum(x['file_bytes'] for x in r['artifacts'].values())==r['output_bytes_before_report'] and
        r['output_bytes_before_report']+(d/'report.json').stat().st_size<=CAP,'complete effective tensor bytes/output cap')
    require(read(d/'prediction_seal.json',r['prediction_seal']['sha256'])==dict(schema=NAME+'.prediction_seal',prepare_sha256=r['execution_binding']['prepare_sha256'],stages=r['stages']),'both actual predictions sealed')
    require([e['kind'] for e in events]==['run_started']+['dispatch_intent','inputs_saved','return_observed','outputs_saved','stage_sealed']*2+['all_predictions_sealed','run_finished'],'exact no-label/no-retry durable event order')
    require(read(d/'execution_binding.json',r['artifacts']['execution_binding.json']['sha256'])==r['execution_binding'] and
        events[0]['binding']==r['execution_binding'] and events[0]['protocol']==protocol(),'durable original execution binding')
    for i,c in enumerate(r['calls']):
        intent,inp,ret,out,stage=events[1+5*i:6+5*i]
        require(intent['call']=={k:c[k] for k in ('index','T','units','native_inputs','inference','TF32off','requested_MATH')}|{'status':'attempted'} and
            inp['index']==ret['index']==out['index']==i and inp['inputs']==c['inputs'] and out['outputs']==c['outputs'] and out['native_outputs']==c['native_outputs'] and
            stage['stage']==i and stage['outputs']==c['outputs'],'exact dispatch/input/return/output/seal chronology, not kinds-only')
    require(events[-2]['seal']==r['prediction_seal'] and events[-1]['status']==r['status'] and
        events[-1]['calls_digest']==digest(r['calls']) and events[-1]['counts']==r['counts'],'full final/seal receipts')
    require(all(m['bindings'].get(path)==sha(path) for path in r['imported_sources'].values()) and
        all(not any(k in op.lower() for k in ('flash_attention','efficient_attention','cudnn_attention')) for op in r['attention_ops']),
        'actual loader-source receipts and no observed forbidden fused attention')
    return dict(schema=NAME+'.consume',status='branch0_T1_T2_saved_native_ABI_verified',report_sha256=expected,stages=2,
        upstream_full_strict_rebuilt=False,new_neural_calls=0,new_physics_calls=0,P1_admission=False,P5_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--output',required=True)
    for cmd in ('verify','run'):
        a=s.add_parser(cmd);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd=='run':a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=s.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha',required=True);a.add_argument('--receipt',required=True)
    q=p.parse_args()
    if q.command=='prepare':v=write_new(q.output,build())
    elif q.command=='verify':verify(q.prepared,q.prepare_sha);v=dict(status='probe_prepare_verified',new_neural_calls=0)
    elif q.command=='run':v=run(q.prepared,q.prepare_sha,q.output,q.authorization)
    else:v=write_new(q.receipt,consume(q.output,q.report_sha))
    print(json.dumps(v,sort_keys=True,allow_nan=False))
    if v['status'] in ('partial_or_failed','blocked_partial_probe'):raise SystemExit(2)


if __name__=='__main__':main()
