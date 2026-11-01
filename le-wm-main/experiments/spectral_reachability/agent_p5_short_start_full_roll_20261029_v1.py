"""Fixed164 short-start causal pred5/10/15/20; MAIN HOSTCUDA only.

All656 full native predictions sealed before common H2 labels. No training,
encoder, physics, scientific admission or cross-model performance ranking.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p5_short_start_cuda_probe_20261029_v1 as probe

NAME='agent_p5_short_start_full_roll_20261029_v1'
ROOT=probe.ROOT
N=164
AUTH='MAIN_P5_SHORT_START_FULL_ROLL_HOSTCUDA_ONLY'
CAP,TENSOR_CAP,FLOOR,RESERVE,META,ARRAY_CAP,WALL=3*1024**3,2*1024**3,200*1024**2,64*1024**2,32*1024**2,4*1024**2,900
FRAMES=([0],[0,5],[0,5,10],[5,10,15])
PACKETS=([0],[0,1],[0,1,2],[1,2,3])
TARGETS=(5,10,15,20)
PROBE_SOURCE='56e4e9919b0e5b21fbeca11230758c999854739482b91214f688cb91066fcfe6'
PROTOCOL_PATH=ROOT/'innerdocs/agent_p5_short_start_full_roll_protocol_20261029_v1.md'
PROTOCOL_SHA='d042320fb3270c21bd818c5d27b6b84df4b5417fa8c51a85e161c99672a711b0'
PROBE_SHAS=dict(prepared='1fe5f72e9234cfa2360d5a837a68dfbc56b1a94b37fb82775f0be90f63fd8e2c',
    report='2f1f60a2f34bf329dcfcb7e5ea378dd3dbf22283be15aa640301ea1ed62974d8',
    strict='1e376b1f06ca3b04ad71822b76178308dc4ad3d700641c97726b1c7f94c5d346')
abi,codec=probe.abi,probe.codec
require,sha,read,raw,exact,digest,merge=probe.require,probe.sha,probe.read,probe.raw,probe.exact,probe.digest,probe.merge
pair_schedule=abi.pre.pair_schedule


def units():return dict(api=4*N,batch_samples=4*N,frame_samples=9*N)


def protocol():
    return dict(schema=NAME+'.protocol',branches=N,pairs=pair_schedule(),frames=[list(x) for x in FRAMES],targets=list(TARGETS),
        packet_indices=[list(x) for x in PACKETS],predictor=units(),new_encoder=0,new_proprio_encoder=0,new_AD=0,new_physics=0,new_training=0,
        native_IO_bytes=7373160*N,bank_tensor_bytes=409800*N,label_tensor_bytes=1966080*N,output_cap=CAP,tensor_cap=TENSOR_CAP,
        array_cap=ARRAY_CAP,metadata_cap=META,floor=FLOOR,final_reserve=RESERVE,wall_seconds=WALL,
        source_protocol_sha256=PROTOCOL_SHA,labels='saved H2 batch3 actual5/10; saved H2 batch2 actual15/20; only after all predictions sealed',
        extra_actual_agent_position_velocity=True,RGB_only=False,fresh=False,known_roots=2,
        requested_MATH_is_not_kernel_certificate=True,observed_fused_attention_rejected=True,
        upstream_fullstrict_rebuilt=False,P1_admission=False,P5_admission=False)


def authority():
    """Metadata/source hashing only; no tensor labels decoded, no fullstrict."""
    require(sha(probe.__file__)==PROBE_SOURCE and sha(PROTOCOL_PATH)==PROTOCOL_SHA,'frozen probe and approved protocol')
    for module in (abi,abi.pre,codec,codec.bridge):require(sha(module.__file__)==probe.PINS[module.__name__],'frozen resolver/loader/codec')
    paths={k:ROOT/'results'/v for k,v in dict(prepared=abi.H2+'_prepare_MAIN01.json',report=abi.H2+'_cuda_MAIN01/report.json',strict=abi.H2+'_consume_MAIN01.json').items()}
    p,r,c=(read(paths[k],h) for k,h in (('prepared',abi.PREPARE_SHA),('report',abi.REPORT_SHA),('strict',abi.STRICT_SHA)))
    pp={k:ROOT/'results'/v for k,v in dict(prepared=probe.NAME+'_prepare_MAIN01.json',report=probe.NAME+'_cuda_MAIN01/report.json',strict=probe.NAME+'_consume_MAIN01.json').items()}
    prep,rep,strict=(read(pp[k],PROBE_SHAS[k]) for k in ('prepared','report','strict'))
    require(strict['schema']==probe.NAME+'.consume' and strict['status']=='branch0_T1_T2_saved_native_ABI_verified' and
        strict['report_sha256']==PROBE_SHAS['report'] and rep['status']=='branch0_T1_T2_native_ABI_recorded' and
        rep['execution_binding']==dict(prepared=str(pp['prepared']),prepare_sha256=PROBE_SHAS['prepared']) and
        rep['errors']==rep['source_errors']==[] and rep['model_before']==rep['model_after']==r['model_before'] and
        prep['protocol']==probe.protocol() and rep['protocol']==probe.protocol(),'actual completed short-start native engineering evidence')
    require(r['execution_binding']['prepared']==str(paths['prepared']),'original H2 exact prepare path')
    catalog=read(probe.CATALOG,probe.CATALOG_SHA)
    require(catalog['schema']==abi.NAME+'.catalog' and catalog['status']=='all164_actual0_IO_pointers_resolved_not_executed' and
        len(catalog['rows'])==N and catalog['design']==abi.design(),'all164 fixed catalog, not selected subset')
    bindings=deepcopy(p['bindings']);merge(bindings,prep['bindings'])
    rows=[]
    for i in range(N):
        record=abi.resolve_actual0(r,p,c,paths['report'].parent,i)
        require(record==catalog['rows'][i] and catalog['authority']==record['inherited_authority'],'exact per-branch catalog crosslink')
        q=r['rows'][i]['label_patch'];call=r['calls'][4*N+i]
        require(q==call['outputs']['$dict']['x_norm_patchtokens']['$array'] and call['module']=='official_encoder' and
            call['phase']=='label' and call['branch']==i and call['status']=='returned' and q['shape']==[2,256,384] and
            q['dtype']=='float32' and q['nbytes']==2*256*384*4 and Path(q['file']).name==q['file'] and
            r['artifacts'][q['file']]=={k:q[k] for k in ('sha256','file_bytes')},'same H2 common label native call/inventory')
        labels=[record['refs']['encoder_patch'],dict(directory=str(paths['report'].parent),array=q,report_json_pointer=f'/rows/{i}/label_patch')]
        for ref in (*record['refs'].values(),*labels):merge(bindings,{str(Path(ref['directory'])/ref['array']['file']):ref['array']['sha256']})
        rows.append(dict(index=i,record=record,labels=labels))
    require(prep['record']==rows[0]['record'] and rep['extraction']==prep['extraction'] and
        prep['extraction']['input_sha256']==probe.ACTUAL0_SHA,'probe/catalog branch0 exact inputs')
    for k,h in (('prepared',abi.PREPARE_SHA),('report',abi.REPORT_SHA),('strict',abi.STRICT_SHA)):merge(bindings,{str(paths[k]):h})
    for k,h in PROBE_SHAS.items():merge(bindings,{str(pp[k]):h})
    for path,h in ((probe.CATALOG,probe.CATALOG_SHA),(Path(probe.__file__).resolve(),PROBE_SOURCE),(PROTOCOL_PATH,PROTOCOL_SHA)):
        merge(bindings,{str(path):h})
    for path in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):merge(bindings,{str(path):sha(path)})
    for path,h in bindings.items():require(sha(path)==h,'all inherited actual file bytes/source hashes unchanged')
    return dict(schema=NAME+'.authority',rows=rows,bindings=bindings,protocol=protocol(),
        bridge_prepared=p['bridge_prepared'],bridge_prepare_sha256=p['bridge_prepare_sha256'])


def own(path):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not any(x.is_symlink() for x in (p,*p.parents)),'own flat results namespace/no links')
    return p


def write_new(path,value):
    p=own(path);b=raw(value)
    require(not p.exists() and len(b)<=META and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive bounded metadata/floor')
    with p.open('xb') as f:require(f.write(b)==len(b),'complete metadata');f.flush();os.fsync(f.fileno())
    codec.bridge.physical.paired.fsync_dir(p.parent)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


class Store:
    """Streaming hashes, no quadratic rehash of the growing journal."""
    def __init__(self,path,started=None,external_bytes=0):
        self.path=own(path);require(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'exclusive output and full cap+floor')
        self.path.mkdir();self.started=time.monotonic() if started is None else started
        self.artifacts={};self.hashes={};self.total=0;self.external_bytes=external_bytes;self.tensor_bytes=0;self.seq=0;self.tail=None;self.array_index=0
    def check(self,size=0,final=False):
        require(self.total+self.external_bytes+size<=CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free>=FLOOR+size,'actual aggregate output cap/floor/reserve')
        require(final or time.monotonic()-self.started<WALL,'whole job900s cap')
    def put(self,name,body,final=False,append=False):
        require(Path(name).name==name and name not in ('','.','..') and type(body) is bytes,'flat complete own bytes')
        require(append and name=='journal.jsonl' or len(body)<=max(META,ARRAY_CAP),'bounded single artifact')
        self.check(len(body),final);p=self.path/name
        require(not p.is_symlink() and (not p.exists() or append and name=='journal.jsonl' and name in self.artifacts),'exclusive file or owned journal append')
        before=p.stat().st_size if p.exists() else 0
        h=self.hashes.get(name,hashlib.sha256())
        try:
            with p.open('ab' if append else 'xb') as f:
                wrote=f.write(body);require(wrote==len(body),'complete durable bytes');f.flush();os.fsync(f.fileno())
        finally:
            if p.exists():
                size=p.stat().st_size;delta=size-before;self.total+=delta
                h.update(body[:delta]);self.hashes[name]=h
                self.artifacts[name]=dict(sha256=h.hexdigest(),file_bytes=size)
        codec.bridge.physical.paired.fsync_dir(self.path)
        return dict(file=name,**self.artifacts[name])
    def emit(self,kind,final=False,**fields):
        e=dict(sequence=self.seq,previous_sha256=self.tail,kind=kind,**deepcopy(fields));e['event_sha256']=digest(e)
        self.put('journal.jsonl',raw(e),final,append=True);self.tail=e['event_sha256'];self.seq+=1
    def array(self,a):
        require(a.dtype==np.float32 and np.isfinite(a).all() and a.nbytes<=ARRAY_CAP-256 and self.tensor_bytes+a.nbytes<=TENSOR_CAP,'full finite float32 tensor caps')
        b=io.BytesIO();np.save(b,np.ascontiguousarray(a),allow_pickle=False)
        rec=dict(self.put(f'tensor_{self.array_index:05d}.npy',b.getvalue()),shape=list(a.shape),dtype='float32',nbytes=a.nbytes)
        self.array_index+=1;self.tensor_bytes+=a.nbytes;return rec


def packets(record):
    s=record['source_normalization'];_,a=abi.pre.normalized(s['raw_proprio'],s['commands'],s['stats'])
    require(a.shape==(4,10) and exact(a,abi.pre.f32(s['packets'],(4,10))),'unchanged four original action packets')
    return a[None].copy(order='C')


def first_slice(ref):
    """Hash full original bytes, decode only time0 via mmap, never future slots."""
    rec=ref['array'];p=Path(ref['directory'])/rec['file']
    require(p.resolve()==p and not any(x.is_symlink() for x in (p,*p.parents)) and p.stat().st_size==rec['file_bytes']<=ARRAY_CAP and sha(p)==rec['sha256'],'original slice source bytes')
    a=np.load(p,allow_pickle=False,mmap_mode='r')
    require(a.dtype==np.float32 and list(a.shape)==rec['shape'] and a.nbytes==rec['nbytes'] and a.flags.c_contiguous and a.offset+a.nbytes==p.stat().st_size,'bounded contiguous single NPY without trailer')
    v=a[:,0:1].copy(order='C');require(np.isfinite(v).all(),'actual0 finite');return v


def prepare(output):
    target=own(output);require(not target.exists(),'exclusive prepare')
    a=authority();s=Store(target.with_name(target.stem+'_inputs'));rows=[];errors=[]
    try:
        for item in a['rows']:
            args,ex=abi.materialize_actual0(item['record'])
            rows.append(dict(index=item['index'],inputs=[s.array(x) for x in args],packets=s.array(packets(item['record'])),extraction=ex))
        require(s.tensor_bytes==409800*N,'complete actual0/action bank bytes')
    except BaseException as e:errors.append(dict(type=type(e).__name__,message=str(e)))
    bank=dict(schema=NAME+'.bank',status='partial_bank' if errors else 'complete_actual0_action_bank',authority_digest=digest(a),rows=rows,errors=errors,
        artifacts=deepcopy(s.artifacts),tensor_bytes=s.tensor_bytes)
    bankref=s.put('bank.json',raw(bank),final=True)
    require(not errors,'partial input bank retained; no prepare emitted')
    m=dict(schema=NAME+'.prepare',status='prepared_no_neural_execution',authority=a,bank=dict(directory=str(s.path),manifest=bankref),
        record=a['rows'][0]['record'],bridge_prepared=a['bridge_prepared'],bridge_prepare_sha256=a['bridge_prepare_sha256'],bindings=deepcopy(a['bindings']))
    for name,rec in s.artifacts.items():merge(m['bindings'],{str(s.path/name):rec['sha256']})
    return write_new(target,m)


def bank_read(m):
    d=own(m['bank']['directory']);rec=m['bank']['manifest'];b=read(d/'bank.json',rec['sha256'])
    require(rec['file']=='bank.json' and rec['file_bytes']==(d/'bank.json').stat().st_size and b['schema']==NAME+'.bank' and
        b['status']=='complete_actual0_action_bank' and not b['errors'] and len(b['rows'])==N and b['authority_digest']==digest(m['authority']) and
        b['tensor_bytes']==409800*N,'complete exact input bank authority')
    require(set(b['artifacts'])|{'bank.json'}=={p.name for p in d.iterdir()},'whole bank inventory/no extras')
    for name,ref in b['artifacts'].items():
        p=d/name;require(Path(name).name==name and not p.is_symlink() and p.stat().st_size==ref['file_bytes'] and sha(p)==ref['sha256'],'exact bank bytes')
    return b


def bank_inputs(m,b,i):
    row=b['rows'][i];require(row['index']==i,'fixed original branch order')
    load=lambda rec:abi.load_array(dict(directory=m['bank']['directory'],array=rec))
    args=tuple(load(x) for x in row['inputs']);acts=load(row['packets'])
    require([x.shape for x in args]==[(1,1,1,16,16,384),(1,1,10),(1,1,256,16)] and acts.shape==(1,4,10) and
        exact(args[1],acts[:,:1]) and row['extraction']['input_sha256']==[hashlib.sha256(x.tobytes()).hexdigest() for x in args],'unaltered actual0 bank and action linkage')
    return args,acts


def verify(prepared,expected):
    m=read(prepared,expected);a=authority()
    require(m['schema']==NAME+'.prepare' and m['status']=='prepared_no_neural_execution' and m['authority']==a and
        m['record']==a['rows'][0]['record'] and m['bridge_prepared']==a['bridge_prepared'] and m['bridge_prepare_sha256']==a['bridge_prepare_sha256'],'reconstructed metadata/source authority')
    b=bank_read(m);d=Path(m['bank']['directory']);bindings=deepcopy(a['bindings'])
    for name,rec in b['artifacts'].items():merge(bindings,{str(d/name):rec['sha256']})
    merge(bindings,{str(d/'bank.json'):m['bank']['manifest']['sha256']});require(m['bindings']==bindings,'entire input bank/source hash closure')
    for i,item in enumerate(a['rows']):
        args,acts=bank_inputs(m,b,i);record=item['record']
        for j,k in enumerate(('visual','action','proprio')):require(exact(args[j],first_slice(record['refs'][k])),'source time0 only, no future feature decoding')
        require(exact(acts,packets(record)),'all original literal actions and original normalizer')
    return m


class Meter:
    def __init__(self):
        self.attempted=dict(api=0,batch_samples=0,frame_samples=0);self.returned=dict(self.attempted);self.pending=None;self.device=None
    def before(self,module,shapes,devices,dtypes,requires_grad,inference,TF32off,MATH):
        require(self.pending is None and module=='predictor' and self.attempted['api']<4*N,'only fixed predictor dispatches, no encoder/nested/retry')
        t=(1,2,3,3)[self.attempted['api']%4]
        require(shapes==[(1,t,1,16,16,384),(1,t,10),(1,t,256,16)] and len(devices)==3 and len(set(devices))==1 and
            all(x=='cuda' or x.startswith('cuda:') for x in devices) and (self.device is None or self.device==devices[0]) and
            dtypes==['float32']*3 and requires_grad==[False]*3 and inference is TF32off is MATH is True,'actual short-start CUDA/frozen/input descriptor guard')
        self.pending=t;self.device=devices[0];u=dict(api=1,batch_samples=1,frame_samples=t)
        for k,v in u.items():self.attempted[k]+=v
        return dict(stage=t,units=u)
    def after(self,shapes,devices,dtypes):
        t=self.pending;require(t is not None and shapes==[(1,t,256,384),None,(1,t,256,16)] and
            devices==[self.device]*2 and dtypes==['float32']*2,'attributable full native output CUDA/shape/dtype')
        for k,v in dict(api=1,batch_samples=1,frame_samples=t).items():self.returned[k]+=v
        self.pending=None


class Trace(probe.Trace):
    def __init__(self,store):
        super().__init__(store);self.guard=Meter()


class CudaBackend(probe.CudaBackend):
    """Unchanged loader/math/observed-op guards, NEW four-stage trace only."""


def next_inputs(stage,args,out,acts):
    require(stage in (0,1,2),'no fifth call')
    advanced=abi.pre.advance_history(FRAMES[stage],args[0],args[2],out)
    require(advanced['frames']==FRAMES[stage+1],'exact causal time advancement')
    return advanced['visual'],acts[:,PACKETS[stage+1]].copy(order='C'),advanced['proprio']


def empty_rows(m):
    return [dict(index=i,identity={k:item['record'][k] for k in ('root_slot','condition','task_id','selection_digest')},
        stages=[dict(target=t,frames=list(f),call=4*i+j,prediction=None) for j,(t,f) in enumerate(zip(TARGETS,FRAMES,strict=True))],labels=None,score=None)
        for i,item in enumerate(m['authority']['rows'])]


def seal_value(rows,expected):return dict(schema=NAME+'.prediction_seal',prepare_sha256=expected,predictions=[dict(index=r['index'],stages=r['stages']) for r in rows])


def labels_after_seal(m,rows,store,ref,expected):
    require(len(rows)==N and all(len(r['stages'])==4 and all(x['prediction'] is not None for x in r['stages']) for r in rows) and
        read(store.path/ref['file'],ref['sha256'])==seal_value(rows,expected),'all656 complete full outputs sealed before any label numerical read')
    for item in m['authority']['rows']:
        store.check();i=item['index'];store.emit('label_read_intent',branch=i,refs=item['labels'])
        arrays=[abi.load_array(ref) for ref in item['labels']]
        require([a.shape for a in arrays]==[(3,256,384),(2,256,384)],'exact H2 common label batches3 and2')
        yield i,arrays


def observations(labels):return [labels[0][j:j+1] for j in (1,2)]+[labels[1][j:j+1] for j in (0,1)]


def scores(predictions,labels,actual0):
    mse=lambda a,b:float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2,dtype=np.float64))
    return dict(MSE=[mse(p[0][:,-1],o) for p,o in zip(predictions,observations(labels),strict=True)],
        actual0_persistence_MSE=[mse(actual0.reshape(1,256,384),o) for o in observations(labels)])


def prefix_checks(m,b,rows,load):
    """All164, nominal of each ORIGINAL41 group; no outcome-driven selection."""
    result=[];cache={}
    for i,row in enumerate(rows):
        nominal=41*(i//41)
        if nominal not in cache:
            args,acts=bank_inputs(m,b,nominal)
            outs=[codec.restore(rows[nominal]['stages'][j]['prediction'],load) for j in (0,1)]
            cache[nominal]=(args,acts,outs)
        ref,ref_acts,ref_outs=cache[nominal];args,acts=bank_inputs(m,b,i)
        outs=[codec.restore(row['stages'][j]['prediction'],load) for j in (0,1)]
        result.append(dict(index=i,reference_index=nominal,actual0_equal=[exact(x,y) for x,y in zip(args,ref,strict=True)],
            warmup_packets_equal=exact(acts[:,:2],ref_acts[:,:2]),
            pred5_10_equal=[dict(visual=exact(out[0],r[0]),proprio=exact(out[2],r[2])) for out,r in zip(outs,ref_outs,strict=True)]))
    return result


def pairs(rows,load):
    result=[]
    for spec in pair_schedule():
        lo,hi=(rows[spec[k]] for k in ('minus_index','plus_index'));missing=[r['index'] for r in (lo,hi) if r['score'] is None]
        row=dict(spec,missing=missing,times=None)
        if not missing:
            obs=[observations([load(x) for x in r['labels']]) for r in (lo,hi)];times=[]
            for j,t in enumerate(TARGETS):
                ps=[load(r['stages'][j]['prediction']['$tuple'][0]['$array'])[:,-1].astype(np.float64) for r in (lo,hi)]
                dp=(ps[1]-ps[0]).ravel();do=(obs[1][j].astype(np.float64)-obs[0][j].astype(np.float64)).ravel()
                pn=float(np.linalg.norm(dp));on=float(np.linalg.norm(do));dot=float(np.dot(dp,do));den=2*spec['h']
                times.append(dict(target=t,predicted_response_l2=pn,observed_response_l2=on,dot=dot,cosine=None if pn==0 or on==0 else dot/(pn*on),
                    predicted_central_difference_l2=float(np.linalg.norm(dp/den)),observed_central_difference_l2=float(np.linalg.norm(do/den)),
                    plus_minus_MSE=hi['score']['MSE'][j]-lo['score']['MSE'][j]))
            row['times']=times
        result.append(row)
    return result


def execute(m,prepared,expected,output,backend_factory=CudaBackend,started=None):
    b=bank_read(m);bankbytes=sum(p.stat().st_size for p in Path(m['bank']['directory']).iterdir())+Path(prepared).stat().st_size
    store=Store(output,started,bankbytes);trace=Trace(store);backend=None
    binding=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected)
    r=dict(schema=NAME+'.report',status='partial_or_failed',protocol=protocol(),execution_binding=binding,rows=empty_rows(m),calls=trace.calls,
        pairs=[dict(x,missing=[x['minus_index'],x['plus_index']],times=None) for x in pair_schedule()],errors=[],source_errors=[],model_before=None,model_after=None,
        prefix_checks=[dict(index=i,reference_index=41*(i//41),actual0_equal=None,warmup_packets_equal=None,pred5_10_equal=None) for i in range(N)],
        inherited_cost_source=dict(report_sha256=abi.REPORT_SHA,strict_sha256=abi.STRICT_SHA),P1_admission=False,P5_admission=False)
    load=lambda rec:abi.load_array(dict(directory=str(store.path),array=rec))
    try:
        store.put('execution_binding.json',raw(binding));store.emit('run_started',binding=binding,protocol=protocol())
        backend=backend_factory(m,trace);r.update(model_before=backend.before,strict_loads=backend.strict,imported_sources=backend.sources)
        for i,row in enumerate(r['rows']):
            args,acts=bank_inputs(m,b,i)
            for j in range(4):
                store.check();backend.predict(args);call=trace.calls[4*i+j]
                require(call['status']=='returned','full native return before causal advancement')
                row['stages'][j]['prediction']=deepcopy(call['outputs']);store.emit('stage_sealed',branch=i,stage=j,outputs=call['outputs'])
                if j<3:args=next_inputs(j,args,codec.restore(call['outputs'],load),acts)
        require(trace.guard.attempted==trace.observed_returns==trace.guard.returned==units() and store.tensor_bytes==7373160*N,'all656 actual paid/returned/validated native predictions')
        r['prediction_seal']=store.put('predictions_seal.json',raw(seal_value(r['rows'],expected)));store.emit('all_predictions_sealed',seal=r['prediction_seal'])
        r['prefix_checks']=prefix_checks(m,b,r['rows'],load);store.emit('warmup_identity_saved',checks_digest=digest(r['prefix_checks']))
        r['model_after_predictions']=backend.receipt();require(r['model_after_predictions']==r['model_before'],'model unchanged at prediction seal')
        for i,label_arrays in labels_after_seal(m,r['rows'],store,r['prediction_seal'],expected):
            row=r['rows'][i];row['labels']=[store.array(x) for x in label_arrays];store.emit('labels_saved',branch=i,labels=row['labels'])
            predictions=[codec.restore(x['prediction'],load) for x in row['stages']]
            args,_=bank_inputs(m,b,i);row['score']=scores(predictions,label_arrays,args[0]);store.emit('score_saved',branch=i,score=row['score'])
        r['pairs']=pairs(r['rows'],load);store.emit('pairs_saved',pairs_digest=digest(r['pairs']))
        require(store.tensor_bytes==(7373160+1966080)*N,'complete native plus common-label saved bytes')
        r['status']='all164_short_start_causal_roll_recorded'
    except BaseException as e:r['errors'].append(dict(type=type(e).__name__,message=str(e)))
    finally:
        if backend is not None:
            try:
                r['model_after']=backend.receipt();require(r['model_before']==r['model_after'],'model unchanged after scoring')
                backend.check_sources(m);r['memory']=backend.memory();r['attention_ops']=backend.attention_ops
            except BaseException as e:r['source_errors'].append(dict(type=type(e).__name__,message=str(e)))
        r['source_errors'].extend(trace.close())
        for path,h in m['bindings'].items():
            try:require(sha(path)==h,'final complete source/model/input bank hashes')
            except BaseException as e:r['source_errors'].append(dict(type=type(e).__name__,message=str(e),path=path))
        r.update(counts=dict(attempted=trace.guard.attempted,returned=trace.observed_returns,validated=trace.guard.returned),tensor_bytes=store.tensor_bytes,
            external_prepare_bank_bytes=bankbytes,elapsed_seconds=time.monotonic()-store.started)
        if r['elapsed_seconds']>=WALL:r['errors'].append(dict(type='wall_cap',message='verify/load/dispatch/final hashes exceeded900s'))
        if r['errors'] or r['source_errors']:r['status']='partial_or_failed'
        store.emit('run_finished',final=True,status=r['status'],calls_digest=digest(r['calls']),counts=r['counts'])
        r.update(artifacts=deepcopy(store.artifacts),journal_chain_tail=store.tail,output_bytes_before_report=store.total)
        store.put('report.json',raw(r),final=True)
    return dict(status=r['status'],report=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH and not os.environ.get('PYTHONPATH'),'explicit MAIN HOSTCUDA authorization/no source overlay')
    started=time.monotonic();m=verify(prepared,expected)
    return execute(m,prepared,expected,output,started=started)


def consume(output,expected):
    d=own(output);r=read(d/'report.json',expected)
    require(r['schema']==NAME+'.report' and r['protocol']==protocol() and len(r['rows'])==N and len(r['pairs'])==len(pair_schedule()),'full branch/pair denominators')
    require(set(r['artifacts'])|{'report.json'}=={p.name for p in d.iterdir()},'whole output exact inventory')
    for name,ref in r['artifacts'].items():
        p=d/name;require(Path(name).name==name and not p.is_symlink() and p.stat().st_size==ref['file_bytes'] and sha(p)==ref['sha256'],'all output file bytes/SHA')
    events=list(codec.journal(d/'journal.jsonl'));require(events[-1]['event_sha256']==r['journal_chain_tail'],'complete canonical journal chain')
    if r['status']=='partial_or_failed':
        require(r['errors'] or r['source_errors'],'actual retained engineering failure')
        return dict(schema=NAME+'.consume',status='blocked_partial_roll',report_sha256=expected,branches=N,pairs=len(pair_schedule()),P1_admission=False,P5_admission=False)
    m=verify(r['execution_binding']['prepared'],r['execution_binding']['prepare_sha256']);b=bank_read(m)
    require(r['status']=='all164_short_start_causal_roll_recorded' and not r['errors'] and not r['source_errors'] and r['elapsed_seconds']<WALL and
        r['model_before']==r['model_after_predictions']==r['model_after']==m['record']['expected_model'] and r['strict_loads']==m['record']['expected_strict_loads'] and
        r['counts']==dict(attempted=units(),returned=units(),validated=units()) and len(r['calls'])==4*N,'complete actual native counts/load schema/unchanged weights')
    pos=0
    def event(kind,**fields):
        nonlocal pos
        e=events[pos];pos+=1
        require(e==dict(sequence=e['sequence'],previous_sha256=e['previous_sha256'],event_sha256=e['event_sha256'],kind=kind,**fields),'exact full durable event fields/order')
    event('run_started',binding=r['execution_binding'],protocol=protocol())
    require(read(d/'execution_binding.json',r['artifacts']['execution_binding.json']['sha256'])==r['execution_binding'],'durable execution binding')
    index=0;tensor_bytes=0;names={'journal.jsonl','execution_binding.json','predictions_seal.json'}
    def load(rec):
        nonlocal index,tensor_bytes
        require(rec['file']==f'tensor_{index:05d}.npy' and r['artifacts'][rec['file']]=={k:rec[k] for k in ('sha256','file_bytes')},'unique full IO/label arrays ordered exactly once')
        index+=1;tensor_bytes+=rec['nbytes'];names.add(rec['file']);return abi.load_array(dict(directory=str(d),array=rec))
    reread=lambda rec:abi.load_array(dict(directory=str(d),array=rec))
    guard=Meter();fresh=empty_rows(m)
    for i,row in enumerate(r['rows']):
        args,acts=bank_inputs(m,b,i)
        require(row['index']==i and row['identity']==fresh[i]['identity'],'literal original branch/root/condition/task/selection')
        for j,stage in enumerate(row['stages']):
            c=r['calls'][4*i+j];actual,kw=codec.restore(c['inputs'],load);out=codec.restore(c['outputs'],load)
            require(c['index']==4*i+j and c['T']==len(FRAMES[j]) and c['status']=='returned' and kw=={} and
                all(exact(x,y) for x,y in zip(actual,args,strict=True)),'all actual native inputs causal actual0/newpred/action linkage')
            specs=c['native_inputs'];require(len(specs)==3 and all(s['shape']==list(x.shape) and s['dtype']==str(x.dtype) for s,x in zip(specs,actual,strict=True)),'input metadata matches NPY')
            witness=guard.before('predictor',[x.shape for x in actual],[s['device'] for s in specs],[s['dtype'] for s in specs],
                [s['requires_grad'] for s in specs],c['inference'],c['TF32off'],c['requested_MATH']);require(c['units']==witness['units'],'actual module sample/token units')
            require(type(out) is tuple and len(out)==3 and out[1] is None,'full native output tuple')
            specs=c['native_outputs'];require(len(specs)==2 and all(s['shape']==list(x.shape) and s['dtype']==str(x.dtype) and s['requires_grad'] is False for s,x in zip(specs,(out[0],out[2]),strict=True)),'output metadata matches NPY')
            guard.after([out[0].shape,None,out[2].shape],[s['device'] for s in specs],[s['dtype'] for s in specs])
            require(stage==fresh[i]['stages'][j]|{'prediction':c['outputs']},'exact target/frame/native-output pointers')
            intent={k:c[k] for k in ('index','T','units','native_inputs','inference','TF32off','requested_MATH')}|{'status':'attempted'}
            event('dispatch_intent',call=intent);event('inputs_saved',index=c['index'],inputs=c['inputs']);event('return_observed',index=c['index'])
            event('outputs_saved',index=c['index'],outputs=c['outputs'],native_outputs=c['native_outputs']);event('stage_sealed',branch=i,stage=j,outputs=c['outputs'])
            if j<3:args=next_inputs(j,args,out,acts)
    require(tensor_bytes==7373160*N and guard.returned==units(),'all complete native outputs before labels')
    ref=r['prediction_seal'];require(ref==dict(file='predictions_seal.json',**r['artifacts']['predictions_seal.json']) and
        read(d/ref['file'],ref['sha256'])==seal_value(r['rows'],r['execution_binding']['prepare_sha256']),'all656 full-output prediction seal')
    event('all_predictions_sealed',seal=ref)
    require(r['prefix_checks']==prefix_checks(m,b,r['rows'],reread),'all164 original-group warmup input/output byte identity checks')
    event('warmup_identity_saved',checks_digest=digest(r['prefix_checks']))
    for i,row in enumerate(r['rows']):
        event('label_read_intent',branch=i,refs=m['authority']['rows'][i]['labels'])
        originals=[abi.load_array(x) for x in m['authority']['rows'][i]['labels']];labels=[load(x) for x in row['labels']]
        require(all(exact(x,y) for x,y in zip(originals,labels,strict=True)),'same H2 full common-label batches bytes, no H1 batch substitution')
        event('labels_saved',branch=i,labels=row['labels'])
        predictions=[codec.restore(x['prediction'],reread) for x in row['stages']];args,_=bank_inputs(m,b,i)
        require(row['score']==scores(predictions,labels,args[0]),'independent per-endpoint exact float64 MSE arithmetic')
        event('score_saved',branch=i,score=row['score'])
    require(r['pairs']==pairs(r['rows'],reread),'all80 pair arithmetic/zero-null rules exact')
    event('pairs_saved',pairs_digest=digest(r['pairs']));event('run_finished',status=r['status'],calls_digest=digest(r['calls']),counts=r['counts'])
    bankbytes=sum(p.stat().st_size for p in Path(m['bank']['directory']).iterdir())+Path(r['execution_binding']['prepared']).stat().st_size
    require(pos==len(events) and names==set(r['artifacts']) and tensor_bytes==r['tensor_bytes']==(7373160+1966080)*N and
        tensor_bytes+409800*N<=TENSOR_CAP and sum(x['file_bytes'] for x in r['artifacts'].values())==r['output_bytes_before_report'] and
        r['external_prepare_bank_bytes']==bankbytes and r['output_bytes_before_report']+(d/'report.json').stat().st_size+bankbytes<=CAP,'complete IO/label/bank tensors and aggregate storage caps')
    require(all(m['bindings'].get(p)==sha(p) for p in r['imported_sources'].values()) and
        all(not any(k in op.lower() for k in ('flash_attention','efficient_attention','cudnn_attention')) for op in r['attention_ops']),'original source receipts and no observed forbidden fused attention')
    return dict(schema=NAME+'.consume',status='all164_short_start_causal_roll_saved_verified',report_sha256=expected,branches=N,pairs=len(pair_schedule()),
        predictor=units(),new_encoder=0,new_physics=0,upstream_fullstrict_rebuilt=False,P1_admission=False,P5_admission=False)


def main():
    require(N==164,'production denominator fixed164')
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('prepare');q.add_argument('--output',required=True)
    for cmd in ('verify','run'):
        q=s.add_parser(cmd);q.add_argument('--prepared',required=True);q.add_argument('--prepare-sha',required=True)
        if cmd=='run':q.add_argument('--output',required=True);q.add_argument('--authorization',required=True)
    q=s.add_parser('consume');q.add_argument('--output',required=True);q.add_argument('--report-sha',required=True);q.add_argument('--receipt',required=True)
    q=p.parse_args()
    if q.command=='prepare':v=prepare(q.output)
    elif q.command=='verify':verify(q.prepared,q.prepare_sha);v=dict(status='full_roll_prepare_verified')
    elif q.command=='run':v=run(q.prepared,q.prepare_sha,q.output,q.authorization)
    else:v=write_new(q.receipt,consume(q.output,q.report_sha))
    print(json.dumps(v,sort_keys=True,allow_nan=False))
    if v['status'] in ('partial_or_failed','blocked_partial_roll'):raise SystemExit(2)


if __name__=='__main__':main()
