"""MAIN HOSTCUDA: all164 early5/10 visual x proprio, fixed short pred15.

Four NEW T3 calls per branch, including bitwise old RR native IO replay.
Prior fullstrict is inherited from exact persisted receipts, never rerun here.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import time
import shutil

import numpy as np

from experiments.spectral_reachability import agent_p5_short_start_comparison_20261029_v1 as evidence

short=evidence.short
codec=short.codec
ROOT=short.ROOT
NAME='agent_p5_early_history_factorial_20261029_v1'
AUTH='MAIN_P5_EARLY_HISTORY_FACTORIAL_HOSTCUDA_ONLY'
N=164
CELLS=('AA','AR','RA','RR')
SHAPES=((1,3,1,16,16,384),(1,3,10),(1,3,256,16))
CAP,TENSOR_CAP,FLOOR,RESERVE,META,WALL=3*1024**3,2*1024**3,200*1024**2,64*1024**2,32*1024**2,900
SHORT_REPORT='1b014fc5c2e9868ee5e08e0dc05b602b5e9708a342c9c842698183f3430297c4'
SHORT_STRICT='3a20eb780028ddd40a88d11c59b8ada9dcbc0c7147bf1df9aa9acde332aa26fd'
EVIDENCE_SOURCE='45c1b957f60b0e9be1b731c88b38c30e60ba5c55502e62969ff205fafc770f1e'
PROTOCOL_SHA='5e80815ceceb1599ee88226dc58dd80965f59d6952ff7d1cef7c62f7fe903f0c'
require,sha,raw,digest,exact=short.require,short.sha,short.raw,short.digest,short.exact
pair_schedule=short.pair_schedule


def units():return dict(api=4*N,batch_samples=4*N,frame_samples=12*N)


def read(path,expected):
    used={};p=evidence.checked(path,expected,used)
    require(p.stat().st_size<=META,'own bounded32MiB metadata before decoding')
    return evidence.read(p,expected,used)


def protocol():
    return dict(schema=NAME+'.protocol',branches=N,pairs=pair_schedule(),cells=list(CELLS),budget_mode='all_four_NEW_656',
        frames=[5,10,15],crossed_frames=[5,10],locked_frame=15,action_packet_indices=[1,2,3],
        action_raw_ranges=[[6,10],[11,15],[16,20]],target=20,predictor=units(),
        order='branch-major AA AR RA RR, all full native IO; all656 seal before commonH2 label values',
        A='H2 second native input actual5/10',R='short causal pred5/10',latest='short causal pred15 in every cell',
        RR='NEW full native three inputs and two full3 outputs must equal old short last call bitwise; no tolerance',
        new_encoder=0,new_proprio_encoder=0,new_AD=0,new_physics=0,new_training=0,
        native_IO_bytes=9830880*N,label_tensor_bytes=786432*N,total_tensor_bytes=10617312*N,
        output_cap=CAP,tensor_cap=TENSOR_CAP,metadata_cap=META,array_cap=short.ARRAY_CAP,
        floor=FLOOR,final_reserve=RESERVE,wall_seconds=WALL,initial_cuda_free_floor=4*1024**3,
        scoring='float64 C-order all256patch*384 MSE; all80 response differences /2h; exact zero -> null',
        effects='visual=((RA-AA)+(RR-AR))/2; proprio=((AR-AA)+(RR-RA))/2; interaction=RR-RA-AR+AA; joint=RR-AA',
        prior_fullstrict_rebuilt=False,prior_receipts='exact pinned complete receipts plus field/SHA/inventory/journal/source joins; not a new prior semantic fullstrict',
        requested_MATH_is_not_kernel_certificate=True,observed_fused_attention_rejected=True,
        exploratory_after_prior_outcomes=True,known_roots=2,fresh=False,RGB_only=False,extra_actual_agent_position_velocity=True,
        mixed_context_physical_counterfactual=False,scientific_repair_claim=False,equal_FLOPs=False,P1_admission=False,P5_admission=False)


def checked_ref(bundle,rec,pointer,used):
    d=bundle['directory'];r=bundle['report']
    require(Path(rec['file']).name==rec['file'] and r['artifacts'].get(rec['file'])=={k:rec[k] for k in ('sha256','file_bytes')},'original descriptor inventory')
    evidence.checked(d/rec['file'],rec['sha256'],used)
    return dict(directory=str(d),array=deepcopy(rec),report_json_pointer=pointer)


def load_ref(ref):return short.abi.load_array(ref)


def source_inputs(req):
    a=tuple(load_ref(x) for x in req['actual_inputs']);r=tuple(load_ref(x) for x in req['rolled_inputs'])
    for args in (a,r):
        require(len(args)==3 and all(x.dtype==np.float32 and x.shape==s and np.isfinite(x).all() for x,s in zip(args,SHAPES,strict=True)),
            'all original finite float32 T3 inputs')
    require(exact(a[1],r[1]),'same full original normalized actions, not a factor')
    return a,r


def splice(a,r,cell):
    require(cell in CELLS,'only four declared cells')
    for block in (a,r):
        require(len(block)==3 and all(x.shape==s and x.dtype==np.float32 and np.isfinite(x).all() for x,s in zip(block,SHAPES,strict=True)),
            'full native shape/dtype/support')
    require(exact(a[1],r[1]),'unaltered original action buffer')
    v=a[0] if cell[0]=='A' else r[0];p=a[2] if cell[1]=='A' else r[2]
    return np.concatenate([v[:,:2],r[0][:,2:]],axis=1),r[1].copy(),np.concatenate([p[:,:2],r[2][:,2:]],axis=1)


def witness(req,cell,args):
    a,r=source_inputs(req);expected=splice(a,r,cell)
    require(len(args)==3 and all(exact(x,y) for x,y in zip(args,expected,strict=True)),'all native input bytes match chosen early history; latest/action fixed')
    return dict(cell=cell,frames=[5,10,15],early_sources=[cell[0],cell[1]],
        raw_input_sha256=[hashlib.sha256(x.tobytes(order='C')).hexdigest() for x in args],
        latest_visual_sha256=hashlib.sha256(args[0][:,2:].tobytes()).hexdigest(),
        latest_proprio_sha256=hashlib.sha256(args[2][:,2:].tobytes()).hexdigest())


def causal_check(req):
    a,r=source_inputs(req)
    for j,stage in enumerate(req['causal_outputs']):
        v,p=(load_ref(stage[k]) for k in ('visual','proprio'))
        t=(1,2,3)[j]
        require(v.shape==(1,t,256,384) and p.shape==(1,t,256,16) and
            exact(r[0][:,j:j+1].reshape(1,1,256,384),v[:,-1:]) and exact(r[2][:,j:j+1],p[:,-1:]),
            'rolled5/10/15 use only genuine short stage last tokens')
    return dict(index=req['index'],visual_early_equal=exact(a[0][:,:2],r[0][:,:2]),proprio_early_equal=exact(a[2][:,:2],r[2][:,:2]))


def bundles(used):
    bs=evidence.bundle('short',ROOT/'results'/(short.NAME+'_cuda_MAIN01'),SHORT_REPORT,
        ROOT/'results'/(short.NAME+'_consume_MAIN01.json'),SHORT_STRICT,used)
    f=evidence.FIXED['H2'];bh=evidence.bundle('H2',ROOT/'results'/(f['stem']+'_cuda_MAIN01'),f['report_sha'],
        ROOT/'results'/(f['stem']+'_consume_MAIN01.json'),f['receipt_sha'],used)
    return bs,bh


def build():
    used={};evidence.checked(Path(evidence.__file__).resolve(),EVIDENCE_SOURCE,used)
    doc=ROOT/'innerdocs/agent_p5_next_identifiability_protocol_20261029_v1.md';evidence.checked(doc,PROTOCOL_SHA,used)
    bs,bh=bundles(used);f=evidence.FIXED['H2']
    rs,rh,ps,ph=bs['report'],bh['report'],bs['prepared'],bh['prepared']
    require(rs['model_before']==rh['model_before'] and rs['strict_loads']==rh['strict_loads'],'same actual inherited model/strict loaded state')
    require(len(rs['rows'])==len(rh['rows'])==len(ps['authority']['rows'])==len(ph['bridge']['requests'])==N and
        rs['protocol']['pairs']==rh['protocol']['pairs']==pair_schedule(),'all original164/80, no subset')
    reqs=[]
    for i,(q,s,h) in enumerate(zip(ph['bridge']['requests'],rs['rows'],rh['rows'],strict=True)):
        rec=ps['authority']['rows'][i]['record'];identity={k:q[k] for k in ('index','root_slot','condition','task_id','selection_digest')}
        require(q['index']==i and (q['root_slot'],q['condition'])==((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41] and
            s['index']==h['index']==rec['index']==i and s['identity']=={k:identity[k] for k in s['identity']} and
            all(rec[k]==h[k]==identity[k] for k in identity),'exact full branch/root/condition/task/selection identity')
        norm=rec['source_normalization']
        require(q['context_global_frames']==rec['parent_frames']==[0,5,10] and q['successor_global_frames']==[15,20] and
            q['commands']==norm['commands'] and q['packets']==norm['packets'] and q['raw_proprio']==norm['raw_proprio'] and
            ph['bridge']['stats']==norm['stats'],'same true history/normalizer/raw20/actions, no source remapping')
        ac,rc=rh['calls'][4*i+3],rs['calls'][4*i+3]
        require(ac['index']==rc['index']==4*i+3 and ac['branch']==i and ac['module']=='predictor' and ac['phase']=='prediction' and
            ac['status']==rc['status']=='returned' and ac['units']==rc['units']==dict(api=1,batch_samples=1,frame_samples=3),
            'actual full T3 source calls')
        native=lambda c:c['inputs']['$tuple'][0]['$tuple']
        require(ac['inputs']['$tuple'][1]==rc['inputs']['$tuple'][1]=={'$dict':{}} and len(native(ac))==len(native(rc))==3,'literal native positional ABI')
        refs=lambda b,c:[checked_ref(b,x['$array'],f'/calls/{4*i+3}/inputs/$tuple/0/$tuple/{j}/$array',used) for j,x in enumerate(native(c))]
        label=rh['rows'][i]['label_patch']
        require(label==rh['calls'][4*N+i]['outputs']['$dict']['x_norm_patchtokens']['$array'] and label['shape']==[2,256,384] and
            ps['authority']['rows'][i]['labels'][1]['array']==label and s['labels'][1]['sha256']==label['sha256'],
            'same original H2 batch2 successor20 labels; no numerical label read')
        out=[]
        for j in range(3):
            c=rs['calls'][4*i+j];stage=s['stages'][j]
            require(stage==dict(target=(5,10,15)[j],frames=list(short.FRAMES[j]),call=4*i+j,prediction=c['outputs']),
                'original causal stage identity, not full3 early predictions')
            out.append({k:checked_ref(bs,c['outputs']['$tuple'][slot]['$array'],f'/calls/{4*i+j}/outputs/$tuple/{slot}/$array',used)
                for k,slot in (('visual',0),('proprio',2))})
        require(s['stages'][3]==dict(target=20,frames=[5,10,15],call=4*i+3,prediction=rc['outputs']), 'old RR final full3 attribution')
        req=dict(identity,actual_inputs=refs(bh,ac),rolled_inputs=refs(bs,rc),causal_outputs=out,
            RR_outputs={k:checked_ref(bs,rc['outputs']['$tuple'][slot]['$array'],f'/calls/{4*i+3}/outputs/$tuple/{slot}/$array',used)
                for k,slot in (('visual',0),('proprio',2))},
            label=checked_ref(bh,label,f'/rows/{i}/label_patch',used),label_png=q['labels'][1])
        require(exact(source_inputs(req)[1][1],np.asarray(q['packets'],np.float32)[None,1:]),'native buffer equals original three packet indices1/2/3')
        req['input_census']=causal_check(req);reqs.append(req)
    require(any(not x['input_census']['visual_early_equal'] for x in reqs) and any(not x['input_census']['proprio_early_equal'] for x in reqs),
        'each early factor must differ somewhere in full164; equal branches remain, no selection')
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):evidence.checked(p,sha(p),used)
    return dict(schema=NAME+'.prepare',status='prepared_no_neural_execution',protocol=protocol(),requests=reqs,bindings=used,
        record=deepcopy(ps['record']),bridge_prepared=ps['bridge_prepared'],bridge_prepare_sha256=ps['bridge_prepare_sha256'],
        inherited=dict(short_report_sha=SHORT_REPORT,short_strict_sha=SHORT_STRICT,H2_report_sha=f['report_sha'],H2_strict_sha=f['receipt_sha'],
            short_resources=bs['resources'],H2_resources=bh['resources'],prior_fullstrict_rebuilt=False))


def verify(prepared,expected):
    m=read(prepared,expected)
    require(m==build(),'entire deterministic prepare/source closure; no nested prior fullstrict');return m


def own(path):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not any(x.is_symlink() for x in (p,*p.parents)), 'own flat nonsymlink result')
    return p


def write_new(path,value):
    p=own(path);body=raw(value)
    require(not p.exists() and len(body)<=META and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'exclusive metadata/cap/floor')
    with p.open('xb') as f:require(f.write(body)==len(body),'full durable write');f.flush();os.fsync(f.fileno())
    codec.bridge.physical.paired.fsync_dir(p.parent)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


class Store(short.Store):
    def __init__(self,path,started=None,external_bytes=0):
        self.path=own(path);self.started=time.monotonic() if started is None else started;self.external_bytes=external_bytes
        require(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new3GiB+floor')
        self.path.mkdir();codec.bridge.physical.paired.fsync_dir(self.path)
        self.artifacts={};self.hashes={};self.total=0;self.tensor_bytes=0;self.seq=0;self.tail=None;self.array_index=0

    def check(self,size=0,final=False):
        require(type(size) is int and size>=0 and self.external_bytes+self.total+size<=CAP-(0 if final else RESERVE) and
            shutil.disk_usage(self.path).free>=FLOOR+size,'complete actual cap/floor/reserve including prepare')
        require(final or time.monotonic()-self.started<WALL,'verify/load/run/final900s cap')

    def array(self,a):
        require(isinstance(a,np.ndarray) and a.dtype==np.float32 and np.isfinite(a).all() and self.tensor_bytes+a.nbytes<=TENSOR_CAP,'whole finite float32 tensor2GiB')
        b=io.BytesIO();np.save(b,np.ascontiguousarray(a),allow_pickle=False);body=b.getvalue()
        require(len(body)<=short.ARRAY_CAP,'single complete4MiB array')
        rec=dict(self.put(f'tensor_{self.array_index:05d}.npy',body),shape=list(a.shape),dtype='float32',nbytes=a.nbytes)
        self.array_index+=1;self.tensor_bytes+=a.nbytes;return rec


class Meter(short.Meter):
    def before(self,module,shapes,devices,dtypes,requires_grad,inference,TF32off,MATH):
        require(self.pending is None and module=='predictor' and self.attempted['api']<4*N,'only four declared T3 predictor calls/branch')
        require(shapes==list(SHAPES) and len(devices)==3 and len(set(devices))==1 and
            all(x=='cuda' or x.startswith('cuda:') for x in devices) and (self.device is None or self.device==devices[0]) and
            dtypes==['float32']*3 and requires_grad==[False]*3 and inference is TF32off is MATH is True,'actual CUDA float32 frozen T3 ABI')
        self.pending=3;self.device=devices[0];u=dict(api=1,batch_samples=1,frame_samples=3)
        for k,v in u.items():self.attempted[k]+=v
        return dict(stage=3,units=u)


class Trace(short.probe.Trace):
    def __init__(self,store):
        super().__init__(store);self.guard=Meter()


# Inherited backend installs hooks on ALL loaded models; observed fused guard
# remains active. No old module global or old full3 guard is altered.
CudaBackend=short.CudaBackend


def empty_rows(m):
    return [dict(index=q['index'],root_slot=q['root_slot'],condition=q['condition'],task_id=q['task_id'],selection_digest=q['selection_digest'],
        predictions={c:None for c in CELLS},RR_verified=None,label=None,score=None) for q in m['requests']]


def rr_check(req,args,out):
    old=tuple(load_ref(x) for x in req['rolled_inputs'])
    expected=(load_ref(req['RR_outputs']['visual']),None,load_ref(req['RR_outputs']['proprio']))
    flags=dict(inputs=[exact(x,y) for x,y in zip(args,old,strict=True)],
        full3_outputs=[exact(out[0],expected[0]),out[1] is expected[1] is None,exact(out[2],expected[2])])
    require(all(flags['inputs']) and all(flags['full3_outputs']),'NEW RR all native IO bitwise old short, never tolerance')
    return flags


def seal_value(rows,expected):
    require(len(rows)==N and [r['index'] for r in rows]==list(range(N)) and all(r['RR_verified'] is not None and
        all(r['predictions'][c] is not None for c in CELLS) for r in rows),'all656 full output trees and RR checks before labels')
    return dict(schema=NAME+'.prediction_seal',prepare_sha256=expected,
        predictions=[dict(index=r['index'],outputs=r['predictions'],RR_verified=r['RR_verified']) for r in rows])


def labels_after_seal(m,rows,store,seal,expected):
    value=seal_value(rows,expected)  # fail before opening even a seal on missing call
    require(read(store.path/seal['file'],seal['sha256'])==value,'exact durable all656 prediction seal')
    for q in m['requests']:
        store.check();store.emit('label_read_intent',branch=q['index'],source=q['label'])
        label=load_ref(q['label']);require(label.shape==(2,256,384),'same H2 native batch2 label')
        yield q['index'],label


def effects(v):
    require(set(v)==set(CELLS),'all four cells')
    a,b,c,d=(np.asarray(v[k],np.float64) for k in CELLS)
    require(a.shape==b.shape==c.shape==d.shape and all(np.isfinite(x).all() for x in (a,b,c,d)),'same finite factorial support')
    return dict(visual=((c-a)+(d-b))/2,proprio=((b-a)+(d-c))/2,interaction=d-c-b+a,joint=d-a)


def row_score(cells,label):
    require(label.shape==(1,256,384) and all(x.shape==label.shape for x in cells.values()),'same final20 allpatch support')
    ms={k:float(np.mean((x.astype(np.float64)-label.astype(np.float64))**2,dtype=np.float64)) for k,x in cells.items()}
    return dict(MSE=ms,MSE_effects={k:float(x) for k,x in effects(ms).items()},
        feature_effect_L2={k:float(np.linalg.norm(x)) for k,x in effects(cells).items()})


def features(row,load):
    cells={c:load(row['predictions'][c]['$tuple'][0]['$array'])[:,-1] for c in CELLS}
    return cells,load(row['label'])[1:2]


def pair(spec,minus,plus):
    a,la=minus;b,lb=plus
    cells={c:evidence.response(a[c],b[c],la,lb) for c in CELLS}
    for c,v in cells.items():
        v['cosine_sign']=evidence.sign(v['cosine'])
        v['central_predicted_L2']=float(np.linalg.norm((b[c].astype(np.float64)-a[c].astype(np.float64)).ravel()/(2*spec['h'])))
        v['central_observed_L2']=float(np.linalg.norm((lb.astype(np.float64)-la.astype(np.float64)).ravel()/(2*spec['h'])))
    diff={c:b[c].astype(np.float64)-a[c].astype(np.float64) for c in CELLS}
    return dict(spec,missing=[],cells=cells,response_effect_L2={k:float(np.linalg.norm(x)) for k,x in effects(diff).items()})


def pairs(rows,load):
    return [pair(s,features(rows[s['minus_index']],load),features(rows[s['plus_index']],load)) for s in pair_schedule()]


def summaries(rows,pair_rows):
    groups=[(None,None)]+[(r,c) for r,c in ((0,'off'),(0,'on'),(1,'off'),(1,'on'))];result=[]
    signs=('positive','negative','zero','null')
    for root,cond in groups:
        accept=lambda x:root is None or (x['root_slot'],x['condition'])==(root,cond)
        rs=[x for x in rows if accept(x)];ps=[x for x in pair_rows if accept(x)]
        mean=lambda xs:None if not xs else float(np.mean(xs,dtype=np.float64))
        result.append(dict(root_slot=root,condition=cond,branches=len(rs),pairs=len(ps),
            mean_MSE={c:mean([x['score']['MSE'][c] for x in rs]) for c in CELLS},
            mean_MSE_effects={k:mean([x['score']['MSE_effects'][k] for x in rs]) for k in ('visual','proprio','interaction','joint')},
            responses={c:dict(observed_zero=sum(x['cells'][c]['observed_zero'] for x in ps),predicted_zero=sum(x['cells'][c]['predicted_zero'] for x in ps),
                signs={s:sum(x['cells'][c]['cosine_sign']==s for x in ps) for s in signs}) for c in CELLS},
            sign_transitions={a+'__to__'+b:{s+'__to__'+t:sum(x['cells'][a]['cosine_sign']==s and x['cells'][b]['cosine_sign']==t for x in ps)
                for s in signs for t in signs} for j,a in enumerate(CELLS) for b in CELLS[j+1:]}))
    return result


def execute(m,prepared,expected,output,backend_factory=CudaBackend,started=None):
    store=Store(output,started,Path(prepared).stat().st_size);trace=Trace(store);backend=None;rows=empty_rows(m)
    binding=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected)
    r=dict(schema=NAME+'.report',status='partial_or_failed',protocol=protocol(),execution_binding=binding,rows=rows,calls=trace.calls,
        pairs=[dict(s,missing=[s['minus_index'],s['plus_index']],cells=None) for s in pair_schedule()],summaries=None,prediction_seal=None,
        errors=[],source_errors=[],model_before=None,model_after=None,P1_admission=False,P5_admission=False,inherited=m['inherited'])
    load=lambda rec:short.abi.load_array(dict(directory=str(store.path),array=rec))
    try:
        store.put('execution_binding.json',raw(binding));store.emit('run_started',binding=binding,protocol=protocol())
        backend=backend_factory(m,trace);r.update(model_before=backend.before,strict_loads=backend.strict,imported_sources=backend.sources)
        for q,row in zip(m['requests'],rows,strict=True):
            a,p=source_inputs(q)
            for cell in CELLS:
                args=splice(a,p,cell);store.check();store.emit('history_ready',branch=q['index'],cell=cell,witness=witness(q,cell,args))
                backend.predict(args);call=trace.calls[-1];require(call['status']=='returned','complete validated full3 native return')
                row['predictions'][cell]=deepcopy(call['outputs'])
                if cell=='RR':row['RR_verified']=rr_check(q,args,codec.restore(call['outputs'],load))
                store.emit('cell_saved',branch=q['index'],cell=cell,outputs=call['outputs'],RR_verified=row['RR_verified'] if cell=='RR' else None)
        require(len(trace.calls)==4*N and trace.guard.attempted==trace.observed_returns==trace.guard.returned==units() and
            store.tensor_bytes==protocol()['native_IO_bytes'],'all656 paid/returned/validated, full IO no padding')
        r['model_after_predictions']=backend.receipt();require(r['model_after_predictions']==backend.before,'frozen model before labels')
        store.emit('model_checked_before_labels',model=r['model_after_predictions'])
        r['prediction_seal']=store.put('predictions_seal.json',raw(seal_value(rows,expected)));store.emit('all_predictions_sealed',seal=r['prediction_seal'])
        for i,label in labels_after_seal(m,rows,store,r['prediction_seal'],expected):
            row=rows[i];row['label']=store.array(label);store.emit('label_saved',branch=i,label=row['label'])
            cells,obs=features(row,load);row['score']=row_score(cells,obs);store.emit('score_saved',branch=i,score=row['score'])
        r['pairs']=pairs(rows,load);r['summaries']=summaries(rows,r['pairs']);store.emit('all_scores_saved',pairs_digest=digest(r['pairs']),summaries_digest=digest(r['summaries']))
        require(store.tensor_bytes==protocol()['total_tensor_bytes'],'all native plus copied label bytes')
        r['status']='all164_early_history_factorial_recorded'
    except BaseException as e:r['errors'].append(dict(type=type(e).__name__,message=str(e)))
    finally:
        if backend is not None:
            try:
                r['model_after']=backend.receipt();require(r['model_after']==backend.before,'model unchanged after score')
                backend.check_sources(m);r['memory']=backend.memory();r['attention_ops']=backend.attention_ops
            except BaseException as e:r['source_errors'].append(dict(type=type(e).__name__,message=str(e)))
        r['source_errors'].extend(trace.close())
        for path,h in m['bindings'].items():
            try:require(sha(path)==h,'final unchanged source/model/asset closure')
            except BaseException as e:r['source_errors'].append(dict(path=path,type=type(e).__name__,message=str(e)))
        r.update(counts=dict(attempted=trace.guard.attempted,returned=trace.observed_returns,validated=trace.guard.returned),
            tensor_bytes=store.tensor_bytes,elapsed_seconds=time.monotonic()-store.started,external_prepare_bytes=store.external_bytes)
        if r['elapsed_seconds']>=WALL:r['errors'].append(dict(type='wall_cap',message='verify/load/dispatch/final exceeded900s'))
        if r['errors'] or r['source_errors']:r['status']='partial_or_failed'
        store.emit('run_finished',final=True,status=r['status'],calls_digest=digest(r['calls']),counts=r['counts'],errors=r['errors'],source_errors=r['source_errors'])
        r.update(artifacts=deepcopy(store.artifacts),journal_chain_tail=store.tail,output_bytes_before_report=store.total)
        store.put('report.json',raw(r),final=True)
    return dict(status=r['status'],path=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH and not os.environ.get('PYTHONPATH'),'explicit MAIN HOSTCUDA authority/no overlay')
    started=time.monotonic();m=verify(prepared,expected)
    return execute(m,prepared,expected,output,started=started)


def consume(output,expected):
    d=own(output);r=read(d/'report.json',expected)
    require(r['schema']==NAME+'.report' and r['protocol']==protocol() and len(r['rows'])==N and len(r['pairs'])==len(pair_schedule()) and
        r['P1_admission'] is r['P5_admission'] is False,'entire164/80 closed-gate report')
    inv=r['artifacts'];require(set(inv)|{'report.json'}=={p.name for p in d.iterdir()},'exact entire output inventory')
    for name,ref in inv.items():
        p=d/name;require(Path(name).name==name and not p.is_symlink() and p.stat().st_size==ref['file_bytes'] and sha(p)==ref['sha256'],'all saved bytes/SHA')
    require(sum(x['file_bytes'] for x in inv.values())==r['output_bytes_before_report'] and
        r['external_prepare_bytes']+r['output_bytes_before_report']+(d/'report.json').stat().st_size<=CAP,'aggregate output/prepare cap')
    events=list(codec.journal(d/'journal.jsonl'));require(events and events[-1]['event_sha256']==r['journal_chain_tail'],'canonical complete journal chain')
    if r['status']=='partial_or_failed':
        require(r['errors'] or r['source_errors'],'partial must retain actual error')
        return dict(schema=NAME+'.consume',status='blocked_partial_factorial',report_sha256=expected,branches=N,pairs=len(pair_schedule()),
            limitation='inventory/hashchain only, not complete semantic certification or scientific negative',P1_admission=False,P5_admission=False)
    b=r['execution_binding'];m=verify(b['prepared'],b['prepare_sha256'])
    require(r['status']=='all164_early_history_factorial_recorded' and not r['errors'] and not r['source_errors'] and r['elapsed_seconds']<WALL and
        r['external_prepare_bytes']==Path(b['prepared']).stat().st_size and r['inherited']==m['inherited'] and
        r['model_before']==r['model_after_predictions']==r['model_after']==m['record']['expected_model'] and r['strict_loads']==m['record']['expected_strict_loads'],
        'complete actual loaded model/state/schema and engineering receipt')
    require(all(m['bindings'].get(p)==sha(p) for p in r['imported_sources'].values()) and
        all(not any(k in op.lower() for k in ('flash_attention','efficient_attention','cudnn_attention')) for op in r['attention_ops']),'imported source closure/observed attention guard')
    pos=0;idx=0;tensor_bytes=0;names={'execution_binding.json','journal.jsonl','predictions_seal.json'}
    def event(kind,**fields):
        nonlocal pos
        require(pos<len(events),'missing durable event');e=events[pos];pos+=1
        require(e==dict(sequence=e['sequence'],previous_sha256=e['previous_sha256'],event_sha256=e['event_sha256'],kind=kind,**fields),'exact full durable event fields/order: '+kind)
    def load(rec):
        nonlocal idx,tensor_bytes
        require(rec['file']==f'tensor_{idx:05d}.npy' and inv.get(rec['file'])=={k:rec[k] for k in ('sha256','file_bytes')},'every IO/label ordered unique and inventoried')
        idx+=1;tensor_bytes+=rec['nbytes'];names.add(rec['file']);return short.abi.load_array(dict(directory=str(d),array=rec))
    reread=lambda rec:short.abi.load_array(dict(directory=str(d),array=rec))
    require(read(d/'execution_binding.json',inv['execution_binding.json']['sha256'])==b,'durable actual execution binding')
    event('run_started',binding=b,protocol=protocol());guard=Meter();fresh=empty_rows(m)
    require(len(r['calls'])==4*N,'exact656 module calls, no retry/omission')
    for q,row,base in zip(m['requests'],r['rows'],fresh,strict=True):
        require(all(row[k]==base[k] for k in ('index','root_slot','condition','task_id','selection_digest')) and set(row['predictions'])==set(CELLS),'all original branch identities/four slots')
        a,p=source_inputs(q)
        for j,cell in enumerate(CELLS):
            c=r['calls'][4*q['index']+j];args=splice(a,p,cell);event('history_ready',branch=q['index'],cell=cell,witness=witness(q,cell,args))
            actual,kw=codec.restore(c['inputs'],load);out=codec.restore(c['outputs'],load)
            require(c['index']==4*q['index']+j and c['T']==3 and c['status']=='returned' and kw=={} and
                len(actual)==3 and all(exact(x,y) for x,y in zip(actual,args,strict=True)),'all native chosen early history bytes, latest/action locked')
            ni,no=c['native_inputs'],c['native_outputs']
            require(len(ni)==3 and all(s['shape']==list(x.shape) and s['dtype']==str(x.dtype) for s,x in zip(ni,actual,strict=True)),'input metadata vs NPY')
            w=guard.before('predictor',[x.shape for x in actual],[x['device'] for x in ni],[x['dtype'] for x in ni],[x['requires_grad'] for x in ni],c['inference'],c['TF32off'],c['requested_MATH'])
            require(c['units']==w['units'] and type(out) is tuple and len(out)==3 and out[1] is None and len(no)==2 and
                all(s['shape']==list(x.shape) and s['dtype']==str(x.dtype) and s['requires_grad'] is False for s,x in zip(no,(out[0],out[2]),strict=True)),'full native visual/proprio output ABI')
            guard.after([out[0].shape,None,out[2].shape],[x['device'] for x in no],[x['dtype'] for x in no])
            intent={k:c[k] for k in ('index','T','units','native_inputs','inference','TF32off','requested_MATH')}|{'status':'attempted'}
            event('dispatch_intent',call=intent);event('inputs_saved',index=c['index'],inputs=c['inputs']);event('return_observed',index=c['index'])
            event('outputs_saved',index=c['index'],outputs=c['outputs'],native_outputs=no)
            require(row['predictions'][cell]==c['outputs'],'full3 both output pointers, not final-only')
            rr=rr_check(q,actual,out) if cell=='RR' else None
            if cell=='RR':require(row['RR_verified']==rr,'all RR native IO exact to old short')
            event('cell_saved',branch=q['index'],cell=cell,outputs=c['outputs'],RR_verified=rr)
    require(tensor_bytes==protocol()['native_IO_bytes'],'all656 native bytes before labels')
    event('model_checked_before_labels',model=r['model_after_predictions'])
    seal=r['prediction_seal'];require(seal==dict(file='predictions_seal.json',**inv['predictions_seal.json']) and
        read(d/seal['file'],seal['sha256'])==seal_value(r['rows'],b['prepare_sha256']),'all656 full-tree seal verified')
    event('all_predictions_sealed',seal=seal)
    for q,row in zip(m['requests'],r['rows'],strict=True):
        event('label_read_intent',branch=q['index'],source=q['label']);saved=load(row['label']);original=load_ref(q['label'])
        require(exact(saved,original) and saved.shape==(2,256,384),'same H2 labelbank, no batch change')
        event('label_saved',branch=q['index'],label=row['label']);cells,obs=features(row,reread)
        require(row['score']==row_score(cells,obs),'independent all four MSE/effects');event('score_saved',branch=q['index'],score=row['score'])
    require(r['pairs']==pairs(r['rows'],reread) and r['summaries']==summaries(r['rows'],r['pairs']),'all80 pairs and all/fourstrata independently recomputed')
    event('all_scores_saved',pairs_digest=digest(r['pairs']),summaries_digest=digest(r['summaries']))
    counts=dict(attempted=units(),returned=units(),validated=units())
    require(r['counts']==counts and guard.returned==units(),'actual full fourcell counters')
    event('run_finished',status=r['status'],calls_digest=digest(r['calls']),counts=counts,errors=[],source_errors=[])
    require(pos==len(events) and names==set(inv) and tensor_bytes==r['tensor_bytes']==protocol()['total_tensor_bytes']<=TENSOR_CAP,'every event/artifact/tensor accounted')
    for path,h in m['bindings'].items():require(sha(path)==h,'final bound source/model/asset bytes unchanged')
    return dict(schema=NAME+'.consume',status='all164_early_history_four_NEW_cells_and80_pairs_verified',report_sha256=expected,
        branches=N,pairs=len(pair_schedule()),predictor=units(),RR_all_native_IO_bitwise_verified=True,prior_fullstrict_rebuilt=False,
        new_encoder=0,new_physics=0,P1_admission=False,P5_admission=False)


def main():
    require(N==164,'production fixed164, no denominator switch')
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--output',required=True)
    for cmd in ('verify','run'):
        a=s.add_parser(cmd);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd=='run':a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=s.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha',required=True);a.add_argument('--receipt',required=True)
    q=p.parse_args()
    if q.command=='prepare':v=write_new(q.output,build())
    elif q.command=='verify':verify(q.prepared,q.prepare_sha);v=dict(status='early_history_prepare_verified',prior_fullstrict_rebuilt=False)
    elif q.command=='run':v=run(q.prepared,q.prepare_sha,q.output,q.authorization)
    else:v=write_new(q.receipt,consume(q.output,q.report_sha))
    print(json.dumps(v,sort_keys=True,allow_nan=False))
    if v['status'] in ('partial_or_failed','blocked_partial_factorial'):raise SystemExit(2)


if __name__=='__main__':main()
