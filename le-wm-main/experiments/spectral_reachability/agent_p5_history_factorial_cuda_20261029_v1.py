"""Fixed164 full-history visual x proprio internal intervention; MAIN HOST CUDA only."""
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

from experiments.spectral_reachability import agent_p5_raw164_jepa_cuda_20261029_v1 as h1
from experiments.spectral_reachability import agent_p5_raw164_jepa_h2_cuda_20261029_v2 as h2

bridge=h2.bridge
ROOT=bridge.ROOT
NAME='agent_p5_history_factorial_cuda_20261029_v1'
AUTH='MAIN_P5_HISTORY_FACTORIAL_HOSTCUDA_ONLY'
CAP,FLOOR,FINAL=2*1024**3,200*1024**2,64*1024**2
WALL,META,TENSOR_CAP=900,32*1024**2,1024**3
require,sha,raw,digest=bridge.require,bridge.sha,bridge.raw,bridge.digest
ARMS=('observed_visual_rolled_proprio','rolled_visual_observed_proprio')
CELLS=('AA','AP','PA','PP')  # visual first, proprio second; A=actual, P=rolled
PINS={
 'H1':('agent_p5_raw164_jepa_cuda_20261029_v1_cuda_MAIN01/report.json','e7ac4ef9eb7491f619de25b04c55f5e56560e2833a840dd5305f3db6207b3a5d'),
 'H2':('agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_MAIN01/report.json','e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579'),
 'H1_strict':('agent_p5_raw164_jepa_cuda_20261029_v1_consume_MAIN01.json','4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759'),
 'H2_strict':('agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_MAIN01.json','783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea')}
SOURCE_PINS={h1.__name__:'610523392e1923e4371556606109def7f882a4a90741e2e14b63af68880e2180',
 h2.__name__:'b661a7c27181ef29f8277e154006f9238ec1c94487c311c1065e4d42fcfc2241',
 bridge.__name__:'d70352f2fc574ca8a226522fc3dda9f015305894c83a1d1826f05808c005da0f'}
SHAPES=((1,3,1,16,16,384),(1,3,10),(1,3,256,16))


def protocol():
    return dict(schema=NAME+'.protocol',branches=164,pairs=bridge.factor.schedule(),arms=list(ARMS),cells=list(CELLS),
        alignment=dict(history_global_frames=[5,10,15],successor_global_frame=20,action_packets_global=[[6,10],[11,15],[16,20]]),
        intervention='entire aligned visual/proprio embedding blocks, not single15 or pure horizon',
        caps={k:dict(api=328 if k=='predictor' else 0,batch_samples=328 if k=='predictor' else 0,
            frame_samples=984 if k=='predictor' else 0) for k in h2.KEYS},
        order='branch-major AP then PA; save actual native IO; seal all328; only then decode common H1 labels and diagonals',
        common_label='original H1 actual20 patch embedding, same-condition factual successor for every cell',
        inherited_diagonals='AA=H1; PP=H2 second call; no rerun; inherited original calls/cost separately disclosed',
        label_batch_seam='H2 terminal label may differ in floating bytes from H1; report bitwise/maxabs, never mix labels in factorial',
        scoring='float64 C-order MSE; visual/proprio main effects and interaction, all80 response cosines; no tuned tolerance',
        full_history_numeric_seam='whole saved blocks include original batch/layout numerical differences; not isolated actual15 content',
        known_roots=2,conditions=['off','on'],RGB_only=False,extra_actual_agent_position_velocity=True,
        physically_reachable_mixed_context=False,physical_counterfactual_certified=False,causal_control_benefit=False,
        cross_model_performance_ranking=False,exploratory_post_outcome=True,
        P1_admission=False,P5_admission=False,new_encoder=0,new_AD=0,new_physics=0,new_training=0,
        cap_bytes=CAP,tensor_io_cap_bytes=TENSOR_CAP,metadata_cap_bytes=META,array_cap_bytes=h2.ARRAY_CAP,
        disk_floor_bytes=FLOOR,terminal_reserve_bytes=FINAL,wall_seconds=WALL,initial_cuda_free_floor_bytes=4*1024**3,
        wall_scope='verify/load/dispatch/score/final source and model checks; bounded failure fsync cleanup may follow',
        backend='HOST CUDA only, frozen eval/inference, TF32off/MATH; no CPU neural fallback')


def schedule():return [(i,a) for i in range(164) for a in ARMS]


def merge(bindings,other):
    for path,value in other.items():
        require(path not in bindings or bindings[path]==value,'no conflicting source/asset authority')
        bindings[path]=value


def ref(directory,rec):return dict(directory=str(Path(directory).absolute()),array=deepcopy(rec))


def load_array(directory,rec):
    a=h2.load_array(directory,rec)
    body=(Path(directory)/rec['file']).read_bytes();buf=io.BytesIO(body);np.load(buf,allow_pickle=False)
    require(buf.tell()==len(body),'single whole NPY, no trailing bytes or second array');return a


def load_ref(r):return load_array(r['directory'],r['array'])


def predictor_refs(call,directory):
    require(call['module']=='predictor' and call['phase']=='prediction' and call['status']=='returned' and
        call['units']==dict(api=1,batch_samples=1,frame_samples=3),'actual completed native predictor call')
    t=call['inputs'];require(set(t)=={'$tuple'} and len(t['$tuple'])==2,'native args kwargs tree')
    a,k=t['$tuple'];require(k=={'$dict':{}} and set(a)=={'$tuple'} and len(a['$tuple'])==3,'exact three positional inputs')
    out=[]
    for x,shape in zip(a['$tuple'],SHAPES,strict=True):
        require(set(x)=={'$array'} and x['$array']['shape']==list(shape) and x['$array']['dtype']=='float32','complete aligned native input shape')
        out.append(ref(directory,x['$array']))
    return out


def aligned_request(i,a,b,ra,rb,ca,cb,d1,d2):
    for key in ('index','root_slot','condition','task_id','selection_digest'):
        require(a[key]==b[key]==ra[key]==rb[key] and (key!='index' or a[key]==i),'same branch/root/condition/task/selection')
    require(a['context_global_frames']==[5,10,15] and a['successor_global_frame']==20 and
        b['context_global_frames']==[0,5,10] and b['successor_global_frames']==[15,20],'reject shifted source chronology')
    require(a['label']==b['labels'][1] and a['label']['phase']=='plan' and a['label']['step']==10 and
        a['label']['condition']==a['condition'] and a['label']['overlay_index']==int(a['condition']=='on'),'same factual successor PNG identity')
    require(ca['branch']==cb['branch']==i,'exact source call attribution')
    aa,pp=predictor_refs(ca,d1),predictor_refs(cb,d2)
    u1,u2=load_ref(aa[1]),load_ref(pp[1])
    require(h2.exact_array(u1,u2) and h2.exact_array(u1,np.asarray(a['packets'],np.float32)[None]) and
        h2.exact_array(u2,np.asarray(b['packets'],np.float32)[None,1:]) and
        h2.exact_array(np.asarray(a['commands'],np.float32),np.asarray(b['commands'],np.float32)[5:]),
        'all3 actual normalized action packets and raw6..20 identical, never cross actions')
    require(ra['prediction']==ca['outputs']['$tuple'][0]['$array'] and rb['prediction']==cb['outputs']['$tuple'][0]['$array'],
        'inherited diagonal is original whole3 output')
    return dict(index=i,root_slot=a['root_slot'],condition=a['condition'],task_id=a['task_id'],selection_digest=a['selection_digest'],
        aligned_history_global_frames=[5,10,15],raw_outgoing16_to20=a['commands'][-5:],
        actual_inputs=aa,rolled_inputs=pp,AA=ref(d1,ra['prediction']),PP=ref(d2,rb['prediction']),
        common_label=ref(d1,ra['label_patch']),H2_label=ref(d2,rb['label_patch']),label_png=deepcopy(a['label']))


def upstream(kind,bindings):
    module=h1 if kind=='H1' else h2
    report_path=ROOT/'results'/PINS[kind][0];receipt_path=ROOT/'results'/PINS[kind+'_strict'][0]
    r=bridge.read(report_path,PINS[kind][1],META);c=bridge.read(receipt_path,PINS[kind+'_strict'][1],META)
    require(r['schema']==module.NAME+'.report' and c['schema']==module.NAME+'.consume' and c['report_sha256']==PINS[kind][1] and
        c['status']==('all164_factual_H1_and80_pairs_verified' if kind=='H1' else 'all164_open_loop_H2_and80_pairs_verified') and
        r['status']==('fixed164_actual15_conditioned_factual_H1_recorded' if kind=='H1' else 'fixed164_root10_open_loop_H2_recorded') and
        len(r['rows'])==164 and len(r['pairs'])==80 and c['requests']==164 and c['pairs']==80 and
        r['errors']==r['source_errors']==[] and r['model_before']==r['model_after_predictions']==r['model_after'] and
        r['model_unchanged'] is True and c['counts']==r['counts'] and
        r['counts']['attempted']==r['counts']['returned']==module.protocol()['caps'],'complete authoritative prior run/strict receipt')
    require(r['P1_admission'] is r['P5_admission'] is c['P1_admission'] is c['P5_admission'] is False,'prior admission closed')
    require([(x['branch'],x['phase'],x['module']) for x in r['calls']]==module.call_schedule() and
        [x['index'] for x in r['calls']]==list(range(len(r['calls']))) and
        all(x['missing']==[] for x in r['pairs']),'full ordered native calls/pairs')
    binding=r['execution_binding'];p=module.verify(binding['prepared'],binding['prepare_sha256'])
    merge(bindings,p['bindings']);merge(bindings,{binding['prepared']:binding['prepare_sha256'],str(report_path):PINS[kind][1],
        str(receipt_path):PINS[kind+'_strict'][1]})
    directory=report_path.parent
    require(set(r['artifacts'])|{'report.json'}=={f.name for f in directory.iterdir()},'exact original directory census')
    for name,rec in r['artifacts'].items():
        path=directory/name
        require(Path(name).name==name and not path.is_symlink() and path.stat().st_size==rec['file_bytes'] and sha(path)==rec['sha256'],'every original artifact SHA/bytes, no feature decoding')
        merge(bindings,{str(path):rec['sha256']})
    tail=None
    for e in h2.journal(directory/'journal.jsonl'):tail=e
    require(tail is not None and tail['event_sha256']==r['journal_chain_tail'] and tail['kind']=='run_finished' and
        tail['rows_digest']==digest(r['rows']) and tail['status']==r['status'],'original full journal chain terminal binding')
    return r,p,directory


def build():
    bindings={}
    for module in (h1,h2,bridge):require(sha(module.__file__)==SOURCE_PINS[module.__name__],'sealed upstream source')
    r1,p1,d1=upstream('H1',bindings);r2,p2,d2=upstream('H2',bindings)
    require(p1['bridge']['stats']==p2['bridge']['stats'] and r1['model_before']==r2['model_before'] and
        r1['strict_loads']==r2['strict_loads'],'identical model/normalizer/strict loaded schemas')
    require(r1['protocol']['pairs']==r2['protocol']['pairs']==protocol()['pairs'],'same complete80 schedule')
    requests=[aligned_request(i,a,b,r1['rows'][i],r2['rows'][i],r1['calls'][3*i+2],r2['calls'][4*i+3],d1,d2)
        for i,(a,b) in enumerate(zip(p1['bridge']['requests'],p2['bridge']['requests'],strict=True))]
    require(len(requests)==164 and [(r['root_slot'],r['condition']) for r in requests]==[bridge.factor.ORDER[i//41] for i in range(164)],'no cohort selection')
    for path in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):merge(bindings,{str(path):sha(path)})
    return dict(schema=NAME+'.prepare',status='prepared_no_neural_execution',protocol=protocol(),requests=requests,bindings=bindings,
        bridge_prepared=p1['bridge_prepared'],bridge_prepare_sha256=p1['bridge_prepare_sha256'],expected_model=r1['model_before'],
        expected_strict_loads=r1['strict_loads'],inherited_costs={k:dict(counts=r['counts'],elapsed_seconds=r['elapsed_seconds'],
            report_sha256=PINS[k][1]) for k,r in (('H1',r1),('H2',r2))})


def verify(path,expected):
    m=bridge.read(path,expected,META);require(m==build(),'exact rebuilt whole prepared source/artifact/model closure');return m


def write_new(path,value):
    path=Path(path).absolute();body=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and
        not any(p.is_symlink() for p in (path,*path.parents)) and len(body)<=META and shutil.disk_usage(path.parent).free>=FLOOR+len(body),'exclusive own bounded metadata')
    with path.open('xb') as f:require(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())
    bridge.physical.paired.fsync_dir(path.parent)
    return dict(path=str(path),sha256=sha(path),status=value['status'])


class Store(h2.Store):
    # Pure codec/tree reuse only; new ownership, resource checks and tensor cap.
    def __init__(self,path):
        self.started=time.monotonic();self.path=Path(path).absolute()
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and not self.path.exists() and
            not any(p.is_symlink() for p in (self.path,*self.path.parents)) and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'NEW2GiB+floor')
        self.path.mkdir();bridge.physical.paired.fsync_dir(self.path.parent)
        self.bytes=0;self.hashes={};self.sizes={};self.seq=0;self.previous=None;self.array_index=0;self.tensor_bytes=0

    def check(self,size=0,final=False):
        require(type(size) is int and size>=0 and self.bytes+size<=CAP-(0 if final else FINAL) and
            shutil.disk_usage(self.path).free>=FLOOR+size,'actual byte cap/floor/terminal reserve')
        require(final or time.monotonic()-self.started<WALL,'full job900s before dispatch/write')

    def array(self,value):
        require(isinstance(value,np.ndarray) and value.dtype==np.float32 and np.isfinite(value).all() and
            self.tensor_bytes+value.nbytes<=TENSOR_CAP,'full finite float32 tensor IO1GiB cap')
        buf=io.BytesIO();np.save(buf,np.ascontiguousarray(value),allow_pickle=False);body=buf.getvalue()
        require(len(body)<=h2.ARRAY_CAP,'whole native tensor, no truncation')
        name=f'tensor_{self.array_index:05d}.npy';self.array_index+=1
        rec=dict(self.write(name,body),shape=list(value.shape),dtype='float32',nbytes=value.nbytes)
        self.tensor_bytes+=value.nbytes;return rec


def mixed_inputs(req,arm):
    require(arm in ARMS and req['aligned_history_global_frames']==[5,10,15],'fixed mixed arm/time')
    a,p=req['actual_inputs'],req['rolled_inputs']
    v,u,prop=(a[0],a[1],p[2]) if arm==ARMS[0] else (p[0],a[1],a[2])
    args=tuple(load_ref(x) for x in (v,u,prop))
    require(all(x.shape==s and x.dtype==np.float32 for x,s in zip(args,SHAPES,strict=True)) and
        h2.exact_array(args[1],load_ref(p[1])),'original complete histories and exactly common actions')
    return args


class Trace:
    def __init__(self,store):
        self.store=store;self.calls=[];self.counts=h2.empty_counts();self.where=None;self.pending=None;self.handles=[]

    def install(self,models,torch):
        def save(t):
            require(t.is_cuda and t.dtype==torch.float32 and not t.requires_grad and torch.isfinite(t).all(),'actual finite frozen CUDA IO')
            return self.store.array(t.detach().contiguous().cpu().numpy())
        def pre(key):
            def hook(module,args,kwargs):
                require(key=='predictor' and self.pending is None and self.where is not None and not torch.is_grad_enabled() and
                    not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'no encoder/AD/TF32 or nested calls')
                n=len(self.calls);require(n<len(schedule()) and self.where==schedule()[n] and len(args)==3 and not kwargs and
                    all(tuple(x.shape)==s for x,s in zip(args,SHAPES,strict=True)),'fixed actual call order/shape/cap before dispatch')
                self.store.check();unit=h2.units(key,list(args[0].shape));h2.add_counts(self.counts,'attempted',key,unit)
                require(all(self.counts['attempted'][key][a]<=v for a,v in protocol()['caps'][key].items()),'actual pre-dispatch resource cap')
                c=dict(index=n,branch=self.where[0],arm=self.where[1],module=key,units=unit,status='attempted')
                self.calls.append(c);self.pending=c;self.store.emit('module_dispatch_intent',call=deepcopy(c))
                c['inputs']=h2.tree((args,kwargs),save,torch.is_tensor);self.store.emit('module_inputs_saved',call_index=n,inputs=c['inputs'])
            return hook
        def post(key):
            def hook(module,args,kwargs,out):
                c=self.pending;require(c is not None and c['module']==key,'attributed native return')
                h2.add_counts(self.counts,'returned',key,c['units']);c['status']='returned';self.store.emit('module_returned',call_index=c['index'])
                c['outputs']=h2.tree(out,save,torch.is_tensor);self.store.emit('module_outputs_saved',call_index=c['index'],outputs=c['outputs']);self.pending=None
            return hook
        for key in h2.KEYS:
            self.handles.append(models[key].register_forward_pre_hook(pre(key),with_kwargs=True))
            self.handles.append(models[key].register_forward_hook(post(key),with_kwargs=True))

    close=h2.Trace.close


class CudaBackend:
    def __init__(self,m,trace):
        import torch
        require(torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=protocol()['initial_cuda_free_floor_bytes'],'HOST CUDA4GiB scratch, no fallback')
        torch.cuda.reset_peak_memory_stats()
        loaded=bridge.load_host_cuda(m['bridge_prepared'],m['bridge_prepare_sha256'],'MAIN_P5_RAW164_JEPA_HOSTCUDA_ONLY')
        self.torch=torch;self.models=loaded['models'];self.before=loaded['model_before'];self.strict=loaded['strict'];self.sources=loaded['sources']
        self.math=loaded['attention_context_factory'];self.trace=trace
        require(self.before==m['expected_model'] and self.strict==m['expected_strict_loads'],'same actual loaded bytes and schema, not metadata-only model identity')
        require(all(p in m['bindings'] and sha(p)==m['bindings'][p] for p in self.sources.values()),'loader imported source binding')
        trace.install(self.models,torch)

    def predict(self,index,arm,args):
        self.trace.where=(index,arm);t=self.torch
        with t.inference_mode(),self.math():
            out=self.models['predictor'](*(t.from_numpy(x.copy()).to('cuda') for x in args))
            require(isinstance(out,tuple) and len(out)==3 and out[1] is None and tuple(out[0].shape)==(1,3,256,384) and
                tuple(out[2].shape)==(1,3,256,16),'whole native visual AND proprio outputs')
        return self.trace.calls[-1]['outputs']['$tuple'][0]['$array']

    receipt=h2.CudaBackend.receipt
    memory=h2.CudaBackend.memory


def empty_rows():
    return [dict(index=i,root_slot=bridge.factor.ORDER[i//41][0],condition=bridge.factor.ORDER[i//41][1],
        predictions={a:None for a in ARMS},score=None) for i in range(164)]


def seal_value(m,rows):
    require(len(rows)==164 and [r['index'] for r in rows]==list(range(164)) and
        all(r['predictions'][a] is not None for r in rows for a in ARMS),'all328 actual outputs before label unlock')
    return dict(schema=NAME+'.prediction_seal',prepared_semantic_sha256=digest(m),
        predictions=[dict(index=r['index'],arm=a,array=r['predictions'][a]) for r in rows for a in ARMS])


def unlock(m,rows,directory,rec):
    seal=bridge.read(Path(directory)/rec['file'],rec['sha256'],META)
    require(seal==seal_value(m,rows),'exact all328 prediction seal')
    for r in rows:
        for a in ARMS:require(load_array(directory,r['predictions'][a]).shape==(1,3,256,384),'full sealed prediction before labels')


def mse(a,b):return float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2,dtype=np.float64))


def factorial(values):
    require(set(values)==set(CELLS),'all four cells, no selected corner')
    aa,ap,pa,pp=(np.asarray(values[c],np.float64) for c in CELLS)
    require(aa.shape==ap.shape==pa.shape==pp.shape and all(np.isfinite(x).all() for x in (aa,ap,pa,pp)),'same finite factorial support')
    return dict(visual=((pa-aa)+(pp-ap))/2,proprio=((ap-aa)+(pp-pa))/2,interaction=pp-pa-ap+aa)


def features(req,row,directory):
    cells={'AA':load_ref(req['AA'])[0,-1].copy(),'PP':load_ref(req['PP'])[0,-1].copy()}
    for arm,cell in zip(ARMS,('AP','PA'),strict=True):cells[cell]=load_array(directory,row['predictions'][arm])[0,-1].copy()
    label=load_ref(req['common_label']);other=load_ref(req['H2_label'])
    require(label.shape==(1,256,384) and other.shape==(2,256,384),'original actual20 label batch shapes')
    return cells,label[0],dict(bitwise_equal=h2.exact_array(label[0],other[1]),maxabs=float(np.max(np.abs(label[0].astype(float)-other[1].astype(float)))))


def row_score(cells,label,seam):
    scores={c:mse(v,label) for c,v in cells.items()};effects=factorial(cells)
    return dict(common_label_cell_MSE=scores,MSE_effects={k:float(v) for k,v in factorial(scores).items()},
        feature_effect_L2={k:float(np.linalg.norm(v)) for k,v in effects.items()},label_H1_H2_numeric_seam=seam)


def score_all(m,rows,directory,emit):
    for req,row in zip(m['requests'],rows,strict=True):
        cells,label,seam=features(req,row,directory);row['score']=row_score(cells,label,seam)
        emit('score_saved',branch=row['index'],score=row['score'])
    result=[]
    for s in protocol()['pairs']:
        lo,hi=s['minus_index'],s['plus_index']
        a,la,_=features(m['requests'][lo],rows[lo],directory);b,lb,_=features(m['requests'][hi],rows[hi],directory)
        observed=lb.astype(float)-la.astype(float);on=float(np.linalg.norm(observed));responses={c:b[c].astype(float)-a[c].astype(float) for c in CELLS}
        item=dict(s,observed_response_L2=on,cells={},response_effect_L2={k:float(np.linalg.norm(v)) for k,v in factorial(responses).items()})
        for c,d in responses.items():
            pn=float(np.linalg.norm(d));item['cells'][c]=dict(predicted_response_L2=pn,
                response_cosine=None if pn==0 or on==0 else float(np.dot(d.ravel(),observed.ravel())/(pn*on)),
                plus_minus_MSE=rows[hi]['score']['common_label_cell_MSE'][c]-rows[lo]['score']['common_label_cell_MSE'][c])
        result.append(item)
    return result


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'explicit MAIN authority before any model or output creation')
    store=Store(output);trace=Trace(store);rows=empty_rows();backend=None;m=None
    r=dict(schema=NAME+'.report',status='partial_or_failed',protocol=protocol(),errors=[],source_errors=[],rows=rows,
        calls=trace.calls,counts=trace.counts,prediction_seal=None,pairs=[dict(s,missing=True) for s in protocol()['pairs']],P1_admission=False,P5_admission=False)
    try:
        m=verify(prepared,expected);r['execution_binding']=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected)
        store.put('execution_binding.json',r['execution_binding']);store.emit('run_started',execution_binding=r['execution_binding'],protocol=protocol())
        backend=CudaBackend(m,trace);r.update(model_before=backend.before,strict_loads=backend.strict,imported_sources=backend.sources,inherited_costs=m['inherited_costs'])
        for req,row in zip(m['requests'],rows,strict=True):
            for arm in ARMS:
                store.check();args=mixed_inputs(req,arm)
                store.emit('history_ready',branch=req['index'],arm=arm,input_bytes_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in args])
                row['predictions'][arm]=backend.predict(req['index'],arm,args)
                store.emit('prediction_saved',branch=req['index'],arm=arm,prediction=row['predictions'][arm])
        require(trace.counts['attempted']==trace.counts['returned']==protocol()['caps'] and len(trace.calls)==328,'complete actual paid328 only')
        r['model_after_predictions']=backend.receipt();require(r['model_after_predictions']==backend.before,'model unchanged before label unlock')
        store.emit('model_checked_before_labels',model_after_predictions=r['model_after_predictions'])
        r['prediction_seal']=store.put('predictions_seal.json',seal_value(m,rows));store.emit('predictions_sealed',receipt=r['prediction_seal'])
        unlock(m,rows,store.path,r['prediction_seal']);store.emit('labels_unlocked',prediction_seal_sha256=r['prediction_seal']['sha256'])
        r['pairs']=score_all(m,rows,store.path,store.emit)
    except BaseException as exc:
        r['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        try:store.emit('failure',final=True,error=r['errors'][-1])
        except BaseException as extra:r['errors'].append(dict(type=type(extra).__name__,message='failure journal: '+str(extra)))
    finally:
        r['errors'].extend(dict(type='hook_cleanup',message=e) for e in trace.close())
        if backend is not None:
            try:
                r['model_after']=backend.receipt();r['model_unchanged']=r['model_after']==backend.before
                require(r['model_unchanged'],'final model unchanged');r['memory']=backend.memory()
            except BaseException as exc:r['errors'].append(dict(type=type(exc).__name__,message='model/memory final: '+str(exc)))
        if m is not None:
            for path,h in m['bindings'].items():
                try:require(sha(path)==h,'final source/asset bytes unchanged')
                except BaseException as exc:r['source_errors'].append(dict(path=path,error=str(exc)))
        r['elapsed_seconds']=time.monotonic()-store.started
        if r['elapsed_seconds']>=WALL:r['errors'].append(dict(type='wall_cap',message='full job exceeded900s'))
        if not r['errors'] and not r['source_errors'] and all(row['score'] is not None for row in rows):r['status']='all164_history_factorial_recorded'
        try:store.emit('run_finished',final=True,status=r['status'],rows_digest=digest(rows),counts=trace.counts,errors=r['errors'])
        except BaseException as exc:
            r['status']='partial_or_failed';r['errors'].append(dict(type=type(exc).__name__,message='terminal journal: '+str(exc)))
        r.update(artifacts=store.inventory(),journal_chain_tail=store.previous,output_bytes_before_report=store.bytes,actual_tensor_io_bytes=store.tensor_bytes,
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        store.put('report.json',r,final=True)
    return dict(status=r['status'],path=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def consume(output,expected):
    directory=Path(output).absolute();r=bridge.read(directory/'report.json',expected,META)
    require(directory.parent==ROOT/'results' and directory.name.startswith(NAME+'_') and not any(p.is_symlink() for p in (directory,*directory.parents)),'own nonsymlink output')
    require(r['schema']==NAME+'.report' and r['protocol']==protocol() and len(r['rows'])==164 and len(r['pairs'])==80 and
        r['P1_admission'] is r['P5_admission'] is False,'whole164/80 and closed gates')
    inv=r['artifacts'];require(set(inv)|{'report.json'}=={p.name for p in directory.iterdir()},'exact complete output inventory')
    require(sum(v['file_bytes'] for v in inv.values())==r['output_bytes_before_report'] and
        r['output_bytes_before_report']+(directory/'report.json').stat().st_size<=CAP,'actual full output cap')
    for name,rec in inv.items():
        p=directory/name;require(Path(name).name==name and not p.is_symlink() and p.stat().st_size==rec['file_bytes'] and sha(p)==rec['sha256'],'every saved artifact bytes/SHA')
    events=list(h2.journal(directory/'journal.jsonl')) if 'journal.jsonl' in inv else []
    require(not events or events[-1]['event_sha256']==r['journal_chain_tail'],'complete journal tail')
    if r['status']=='partial_or_failed':
        require(r['errors'] or r['source_errors'],'retain actual partial cause')
        return dict(schema=NAME+'.consume',status='blocked_partial_capture',report_sha256=expected,branches=164,pairs=80,P1_admission=False,P5_admission=False,
            limitation='artifact/hashchain integrity only; partial not a scientific negative or complete IO certification')
    require(r['status']=='all164_history_factorial_recorded' and not r['errors'] and not r['source_errors'] and r['elapsed_seconds']<WALL,'complete within wall')
    binding=r['execution_binding'];m=verify(binding['prepared'],binding['prepare_sha256'])
    require(bridge.read(directory/'execution_binding.json')==binding and r['model_before']==r['model_after_predictions']==r['model_after']==m['expected_model'] and
        r['model_unchanged'] is True and r['strict_loads']==m['expected_strict_loads'] and r['inherited_costs']==m['inherited_costs'],'actual same loaded model, inherited cost separated')
    require(all(p in m['bindings'] and sha(p)==m['bindings'][p] for p in r['imported_sources'].values()) and
        {'app.plan_common.models.AdaLN_vit','app.plan_common.models.prop_embedding','dinov2.models.vision_transformer'}<=set(r['imported_sources']),'actual imported sources')
    mem=r['memory'];require(set(mem)=={'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','cuda_free_bytes','cuda_total_bytes','host_peak_rss_bytes'} and
        all(type(v) is int and v>=0 for v in mem.values()) and mem['cuda_peak_allocated_bytes']<=mem['cuda_peak_reserved_bytes'] and mem['cuda_free_bytes']<=mem['cuda_total_bytes'],'actual memory receipt')
    calls=r['calls'];require(len(calls)==328 and [(c['branch'],c['arm']) for c in calls]==schedule() and
        [c['index'] for c in calls]==list(range(328)),'no omission/retry/padding/call reorder')
    cursor=0;counts=h2.empty_counts();names={'execution_binding.json','journal.jsonl','predictions_seal.json'};tensor_bytes=0;array_index=0
    def expect(kind):
        nonlocal cursor
        require(cursor<len(events) and events[cursor]['kind']==kind,'exact durable event ordering: '+kind)
        e=events[cursor];cursor+=1;return e
    def load(rec):
        nonlocal tensor_bytes,array_index
        require(rec['file']==f'tensor_{array_index:05d}.npy' and inv.get(rec['file'])=={k:rec[k] for k in ('sha256','file_bytes')},'every actual IO tensor unique/ordered/inventory linked')
        array_index+=1;names.add(rec['file']);tensor_bytes+=rec['nbytes'];return load_array(directory,rec)
    start=expect('run_started');require(start['execution_binding']==binding and start['protocol']==protocol(),'durable actual start')
    for req,row in zip(m['requests'],r['rows'],strict=True):
        require({k:row[k] for k in ('index','root_slot','condition')}=={k:req[k] for k in ('index','root_slot','condition')} and
            set(row['predictions'])==set(ARMS),'exact full branch identity and exactly two new prediction slots')
        for j,arm in enumerate(ARMS):
            args=mixed_inputs(req,arm);ready=expect('history_ready')
            require(ready['branch']==req['index'] and ready['arm']==arm and ready['input_bytes_sha256']==[hashlib.sha256(x.tobytes()).hexdigest() for x in args],'unaltered whole input blocks ready')
            c=calls[2*req['index']+j]
            require(c['module']=='predictor' and c['status']=='returned' and c['units']==dict(api=1,batch_samples=1,frame_samples=3),'native actual predictor paid count')
            intent=expect('module_dispatch_intent');require(intent['call']=={k:c[k] for k in ('index','branch','arm','module','units')}|{'status':'attempted'},'actual pre-dispatch receipt')
            inp=expect('module_inputs_saved');ret=expect('module_returned');out=expect('module_outputs_saved')
            require(inp['call_index']==ret['call_index']==out['call_index']==c['index'] and inp['inputs']==c['inputs'] and out['outputs']==c['outputs'],'complete actual IO journal')
            (actual,kwargs)=h2.restore(c['inputs'],load);native=h2.restore(c['outputs'],load)
            require(kwargs=={} and isinstance(actual,tuple) and len(actual)==3 and all(h2.exact_array(x,y) for x,y in zip(actual,args,strict=True)),'bitwise exact factorial native input linkage, no labels')
            require(isinstance(native,tuple) and len(native)==3 and native[1] is None and native[0].shape==(1,3,256,384) and native[2].shape==(1,3,256,16) and
                row['predictions'][arm]==c['outputs']['$tuple'][0]['$array'],'complete native outputs and row attribution')
            for phase in counts:h2.add_counts(counts,phase,'predictor',c['units'])
            saved=expect('prediction_saved');require(saved['branch']==req['index'] and saved['arm']==arm and saved['prediction']==row['predictions'][arm],'durable original whole3 output')
    checked=expect('model_checked_before_labels');require(checked['model_after_predictions']==r['model_after_predictions'],'unchanged model barrier')
    sealed=expect('predictions_sealed');require(sealed['receipt']==r['prediction_seal'],'durable seal association');unlock(m,r['rows'],directory,r['prediction_seal'])
    unlocked=expect('labels_unlocked');require(unlocked['prediction_seal_sha256']==r['prediction_seal']['sha256'],'all328 seal before labels/scores')
    rows=deepcopy(r['rows'])
    def score_event(kind,**fields):
        e=expect(kind);require(all(e[k]==v for k,v in fields.items()),'exact independent score event')
    pairs=score_all(m,rows,directory,score_event)
    require(rows==r['rows'] and pairs==r['pairs'],'independent all164 four-cell scores/factorial and80 pairs')
    finish=expect('run_finished');require(finish['status']==r['status'] and finish['rows_digest']==digest(rows) and finish['counts']==counts and finish['errors']==[] and
        cursor==len(events) and counts==r['counts'] and counts['attempted']==counts['returned']==protocol()['caps'],'full event/call census and exact caps')
    require(names==set(inv) and tensor_bytes==r['actual_tensor_io_bytes']<=TENSOR_CAP,'every artifact role/actual tensor byte census')
    for path,value in m['bindings'].items():require(sha(path)==value,'final unchanged source/model/artifact closure')
    return dict(schema=NAME+'.consume',status='all164_history_factorial_and80_pairs_verified',report_sha256=expected,branches=164,pairs=80,
        new_predictor_calls=328,new_encoder=0,new_physics=0,counts=counts,artifacts=len(inv),P1_admission=False,P5_admission=False,
        physical_counterfactual_certified=False,exploratory_post_outcome=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    for cmd in ('verify','run'):
        a=sub.add_parser(cmd);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd=='run':a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=sub.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha',required=True);a.add_argument('--receipt',required=True)
    args=p.parse_args()
    if args.command=='prepare':value=write_new(args.output,build())
    elif args.command=='verify':verify(args.prepared,args.prepare_sha);value=dict(status='prepared_closure_verified',new_neural_calls=0)
    elif args.command=='run':value=run(args.prepared,args.prepare_sha,args.output,args.authorization)
    else:value=write_new(args.receipt,consume(args.output,args.report_sha))
    print(json.dumps(value,sort_keys=True,allow_nan=False));return 2 if value['status'] in ('partial_or_failed','blocked_partial_capture') else 0


if __name__=='__main__':raise SystemExit(main())
