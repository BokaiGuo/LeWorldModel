"""MAIN HOSTCUDA temporal5x10: 328 new corners +164 bitwise RR bridges.

All492 full native outputs sealed before labels; no prior fullstrict rerun.
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

from experiments.spectral_reachability import agent_p5_early_history_factorial_20261029_v1 as early

short,codec,evidence=early.short,early.codec,early.evidence
ROOT=early.ROOT
NAME='agent_p5_temporal_history_factorial_20261029_v1'
AUTH='MAIN_P5_TEMPORAL_HISTORY_FACTORIAL_HOSTCUDA_ONLY'
N=164
ARMS=('T_AR','T_RA','T_RR')
CELLS=('T_AA','T_AR','T_RA','T_RR')
SHAPES=early.SHAPES
CAP,TENSOR_CAP,FLOOR,RESERVE,META,WALL=3*1024**3,2*1024**3,200*1024**2,64*1024**2,32*1024**2,900
REPORT_SHA='4fc86a7fdc69fbe95003e417ee38e3a4d430360a8eb5bd0c89f73ee845926d0f'
RECEIPT_SHA='d9461f7a21fa13454eca1f4371488431cf833a08ee758990c2dcb999b342600e'
PREPARE_SHA='c8376a96a8320d4f147bf5e526954fbe8e31d37db565428ad82916dcc9d0f9c0'
SCALAR_SHA='ba881dc05522683fe44491a04e028c27dd66e20ae1bb4cd65be9a8cb22ee212e'
EARLY_SOURCE='39dbb6d61121ea9dbd6763be9b0104c45a8b3dbcf20ad3b090f21bed5a5645b7'
PROTOCOL_SHA='8726514b653d85f4736102e06cc81b8652e46f9ebcab6c0875df2fe0141aee7d'
require,sha,raw,digest,exact=early.require,early.sha,early.raw,early.digest,early.exact
load_ref=early.load_ref
pair_schedule=early.pair_schedule


def units():return dict(api=3*N,batch_samples=3*N,frame_samples=9*N)


def protocol():
    return dict(schema=NAME+'.protocol',branches=N,pairs=pair_schedule(),arms=list(ARMS),cells=list(CELLS),budget_mode='predeclared_328_plus164_RR_492',
        frames=[5,10,15],crossed_slots=[0,1],locked_slot=2,target=20,action_packet_indices=[1,2,3],
        A='actual visual+proprio at one time slot',R='causal short predicted visual+proprio at one time slot',
        latest='short pred15 visual and proprio locked in every cell',predictor=units(),
        order='branch-major T_AR T_RA T_RR; all492 full native IO seal before any future label decode',
        inherited_T_AA='old early AA full3 output, not newly predicted',T_RR='NEW all native IO bitwise prior early RR; old AA/RR sources retained',
        native_IO_bytes=7373160*N,label_tensor_bytes=786432*N,total_tensor_bytes=8159592*N,
        output_cap=CAP,tensor_cap=TENSOR_CAP,metadata_cap=META,array_cap=short.ARRAY_CAP,floor=FLOOR,final_reserve=RESERVE,wall_seconds=WALL,
        new_encoder=0,new_proprio_encoder=0,new_AD=0,new_physics=0,new_training=0,
        effects='slot5=((T_RA-T_AA)+(T_RR-T_AR))/2; slot10=((T_AR-T_AA)+(T_RR-T_RA))/2; interaction=T_RR-T_RA-T_AR+T_AA; joint=T_RR-T_AA',
        prior_fullstrict_rebuilt=False,independent_scalar_evidence='pinned complete saved analysis, no nested analyze/verify',
        scoring='common H2 label20 batch2 slot1; float64 C-order all256x384; all80 including zero; exactzero -> null',
        cross_job_limit='NEW RR bridge certifies this input full3 reproduction only, not arbitrary inputs or old AA global determinism',
        requested_MATH_is_not_kernel_certificate=True,observed_fused_attention_rejected=True,known_roots=2,fresh=False,RGB_only=False,
        extra_actual_agent_position_velocity=True,mixed_context_physical_counterfactual=False,exploratory_after_prior_outcomes=True,
        independent_confirmation=False,equal_FLOPs=False,P1_admission=False,P5_admission=False)


def read(path,expected):
    p=evidence.checked(path,expected,{})
    require(p.stat().st_size<=META,'bounded32MiB metadata');return evidence.read(p,expected,{})


def prior(used):
    d=ROOT/'results'/(early.NAME+'_cuda_MAIN01');rp=d/'report.json';cp=ROOT/'results'/(early.NAME+'_consume_MAIN01.json')
    r=evidence.read(rp,REPORT_SHA,used);c=evidence.read(cp,RECEIPT_SHA,used)
    require(r['schema']==early.NAME+'.report' and c['schema']==early.NAME+'.consume' and
        r['status']=='all164_early_history_factorial_recorded' and c['status']=='all164_early_history_four_NEW_cells_and80_pairs_verified' and
        c['report_sha256']==REPORT_SHA and c['branches']==N and c['pairs']==len(pair_schedule()) and c['predictor']==early.units() and
        c['RR_all_native_IO_bitwise_verified'] is True and c['prior_fullstrict_rebuilt'] is False,'complete pinned prior MAIN receipt')
    require(r['protocol']==early.protocol() and r['errors']==r['source_errors']==[] and len(r['rows'])==N and len(r['calls'])==4*N and
        len(r['pairs'])==len(pair_schedule()) and r['counts']==dict(attempted=early.units(),returned=early.units(),validated=early.units()) and
        r['model_before']==r['model_after_predictions']==r['model_after'] and
        all(x['P1_admission'] is x['P5_admission'] is False for x in (r,c)),'all original branches/cells/counts/model/closed gates')
    b=r['execution_binding'];require(b['prepare_sha256']==PREPARE_SHA,'actual prepared SHA')
    p=evidence.read(b['prepared'],PREPARE_SHA,used)
    require(p['schema']==early.NAME+'.prepare' and p['status']=='prepared_no_neural_execution' and p['protocol']==r['protocol'] and
        len(p['requests'])==N and p['record']['expected_model']==r['model_before'] and p['record']['expected_strict_loads']==r['strict_loads'] and
        p['inherited']==r['inherited'],'original prepared closure, no upstream build/verify/consume')
    for path,h in p['bindings'].items():evidence.checked(path,h,used)
    evidence.inventory(d,r,used)
    ap=ROOT/'results/agent_p5_factorial_independent_scalars_20261029_v1_MAIN01.json';a=evidence.read(ap,SCALAR_SHA,used)
    require(a['schema']=='agent_p5_factorial_independent_scalars_20261029_v1.analysis' and
        a['status']=='all164_fourcell_and80_pair_independent_scalars_verified' and a['branches']==N and a['pair_count']==len(pair_schedule()) and
        a['arguments']==dict(report=str(rp),report_sha=REPORT_SHA,receipt=str(cp),receipt_sha=RECEIPT_SHA) and
        a['verification_boundary']['scalar_arithmetic_independently_recomputed'] is True and
        a['verification_boundary']['prior_fullstrict_rebuilt'] is False and a['P1_admission'] is a['P5_admission'] is False,
        'exact completed independent scalar identity; no recomputation or outcome selection')
    for path,h in a['bindings'].items():evidence.checked(path,h,used)
    return d,r,p


def source_inputs(req):
    a=tuple(load_ref(x) for x in req['AA_inputs']);r=tuple(load_ref(x) for x in req['RR_inputs'])
    for block in (a,r):
        require(len(block)==3 and all(x.shape==s and x.dtype==np.float32 and np.isfinite(x).all() for x,s in zip(block,SHAPES,strict=True)), 'full finite float32 native T3')
    require(exact(a[1],r[1]) and exact(a[0][:,2:],r[0][:,2:]) and exact(a[2][:,2:],r[2][:,2:]),'old AA/RR same actions and latest15')
    return a,r


def splice(a,r,cell):
    require(cell in CELLS,'only temporal cells, not old modality names')
    for block in (a,r):require(len(block)==3 and all(x.shape==s and x.dtype==np.float32 and np.isfinite(x).all() for x,s in zip(block,SHAPES,strict=True)),'complete native source shapes/dtypes')
    require(exact(a[1],r[1]) and exact(a[0][:,2:],r[0][:,2:]) and exact(a[2][:,2:],r[2][:,2:]),'fixed action and latest input')
    choices=cell[2:];blocks=[a if x=='A' else r for x in choices]
    return (np.concatenate([blocks[0][0][:,:1],blocks[1][0][:,1:2],r[0][:,2:]],axis=1),r[1].copy(),
        np.concatenate([blocks[0][2][:,:1],blocks[1][2][:,1:2],r[2][:,2:]],axis=1))


def census(req):
    a,r=source_inputs(req)
    flags={c:{old:all(exact(x,y) for x,y in zip(splice(a,r,c),endpoint,strict=True)) for old,endpoint in (('T_AA',a),('T_RR',r))} for c in ARMS[:2]}
    return dict(index=req['index'],new_corner_equals_old=flags,
        slot_equal=[dict(visual=exact(a[0][:,j:j+1],r[0][:,j:j+1]),proprio=exact(a[2][:,j:j+1],r[2][:,j:j+1])) for j in (0,1)])


def build():
    used={};evidence.checked(Path(early.__file__).resolve(),EARLY_SOURCE,used)
    evidence.checked(ROOT/'innerdocs/agent_p5_factorial_next_identifiability_20261029_v1.md',PROTOCOL_SHA,used)
    d,r,p=prior(used);bundle=dict(directory=d,report=r);requests=[]
    for i,(row,q) in enumerate(zip(r['rows'],p['requests'],strict=True)):
        keys=('index','root_slot','condition','task_id','selection_digest')
        require(row['index']==i and (row['root_slot'],row['condition'])==((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41] and
            all(row[k]==q[k] for k in keys) and set(row['predictions'])==set(early.CELLS) and
            row['RR_verified']==dict(inputs=[True]*3,full3_outputs=[True]*3),'original full identity and completed old RR')
        req={k:row[k] for k in keys}
        for j,name in ((0,'AA'),(3,'RR')):
            c=r['calls'][4*i+j];inp=c['inputs']['$tuple']
            require(c['index']==4*i+j and c['T']==3 and c['status']=='returned' and c['units']==dict(api=1,batch_samples=1,frame_samples=3) and
                inp[1]=={'$dict':{}} and len(inp[0]['$tuple'])==3 and row['predictions'][name]==c['outputs'],'old AA/RR native call attribution')
            req[name+'_inputs']=[early.checked_ref(bundle,x['$array'],f'/calls/{4*i+j}/inputs/$tuple/0/$tuple/{k}/$array',used) for k,x in enumerate(inp[0]['$tuple'])]
            req[name+'_outputs']={k:early.checked_ref(bundle,c['outputs']['$tuple'][slot]['$array'],f'/calls/{4*i+j}/outputs/$tuple/{slot}/$array',used)
                for k,slot in (('visual',0),('proprio',2))}
            require(c['outputs']['$tuple'][1] is None and all(req[name+'_outputs'][k]['array']['shape']==shape and
                req[name+'_outputs'][k]['array']['dtype']=='float32' for k,shape in (('visual',[1,3,256,384]),('proprio',[1,3,256,16]))), 'both inherited endpoints are complete original full3 outputs')
        req['label']=early.checked_ref(bundle,row['label'],f'/rows/{i}/label',used)
        require(req['label']['array']['shape']==[2,256,384] and req['label']['array']['sha256']==q['label']['array']['sha256'],'same saved common H2 label metadata, not decode')
        a,rr=source_inputs(req);old_a,old_r=early.source_inputs(q)
        require(all(exact(x,y) for x,y in zip(a,early.splice(old_a,old_r,'AA'),strict=True)) and
            all(exact(x,y) for x,y in zip(rr,early.splice(old_a,old_r,'RR'),strict=True)), 'old native history lineage exact to original sources')
        req['input_census']=census(req);requests.append(req)
    for corner in ARMS[:2]:
        for endpoint in ('T_AA','T_RR'):
            require(any(not q['input_census']['new_corner_equals_old'][corner][endpoint] for q in requests), 'new corner cannot be all164 duplicate old endpoint')
    require(any(not all(exact(x,y) for x,y in zip(splice(*source_inputs(q),'T_AR'),splice(*source_inputs(q),'T_RA'),strict=True)) for q in requests), 'new two corners must not be globally identical')
    for path in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):evidence.checked(path,sha(path),used)
    return dict(schema=NAME+'.prepare',status='prepared_no_neural_execution',protocol=protocol(),requests=requests,bindings=used,
        record=p['record'],bridge_prepared=p['bridge_prepared'],bridge_prepare_sha256=p['bridge_prepare_sha256'],
        inherited=dict(report_sha256=REPORT_SHA,receipt_sha256=RECEIPT_SHA,prepare_sha256=PREPARE_SHA,independent_scalar_sha256=SCALAR_SHA,
            counts=r['counts'],elapsed_seconds=r['elapsed_seconds'],cost_sources=r['inherited'],prior_fullstrict_rebuilt=False))


def verify(path,expected):
    m=read(path,expected);require(m==build(),'deterministic new prepare closure; no nested fullstrict');return m


def own(path):
    p=Path(path).absolute();require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not any(x.is_symlink() for x in (p,*p.parents)),'own nonsymlink flat result');return p


def write_new(path,value):
    p=own(path);body=raw(value)
    require(not p.exists() and len(body)<=META and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'exclusive metadata/cap/floor')
    with p.open('xb') as f:require(f.write(body)==len(body),'durable complete metadata');f.flush();os.fsync(f.fileno())
    codec.bridge.physical.paired.fsync_dir(p.parent);return dict(status=value['status'],path=str(p),sha256=sha(p))


class Store(early.Store):
    def __init__(self,path,started=None,external_bytes=0):
        self.path=own(path);self.started=time.monotonic() if started is None else started;self.external_bytes=external_bytes
        require(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'exclusive NEW3GiB+floor')
        self.path.mkdir();codec.bridge.physical.paired.fsync_dir(self.path)
        self.artifacts={};self.hashes={};self.total=0;self.tensor_bytes=0;self.seq=0;self.tail=None;self.array_index=0

    def check(self,size=0,final=False):
        require(type(size) is int and size>=0 and self.external_bytes+self.total+size<=CAP-(0 if final else RESERVE) and
            shutil.disk_usage(self.path).free>=FLOOR+size,'aggregate output/prepare cap/floor/reserve')
        require(final or time.monotonic()-self.started<WALL,'verify/load/run/final900s cap')


class Meter(early.Meter):
    def before(self,module,shapes,devices,dtypes,requires_grad,inference,TF32off,MATH):
        require(self.pending is None and module=='predictor' and self.attempted['api']<3*N,'only predeclared three T3 calls/branch')
        require(shapes==list(SHAPES) and len(devices)==3 and len(set(devices))==1 and all(x=='cuda' or x.startswith('cuda:') for x in devices) and
            (self.device is None or self.device==devices[0]) and dtypes==['float32']*3 and requires_grad==[False]*3 and
            inference is TF32off is MATH is True,'actual frozen CUDA-only aligned T3 descriptor')
        self.pending=3;self.device=devices[0];u=dict(api=1,batch_samples=1,frame_samples=3)
        for k,v in u.items():self.attempted[k]+=v
        return dict(stage=3,units=u)


class Trace(early.Trace):
    def __init__(self,store):
        super().__init__(store);self.guard=Meter()


CudaBackend=early.CudaBackend


def witness(req,cell,args):
    require(all(exact(x,y) for x,y in zip(args,splice(*source_inputs(req),cell),strict=True)),'exact selected temporal slots, paired modalities and fixed action/latest')
    return dict(cell=cell,slot5_source=cell[2],slot10_source=cell[3],latest_source='short_pred15',
        input_sha256=[hashlib.sha256(x.tobytes(order='C')).hexdigest() for x in args])


def rr_check(req,args,out):
    expected=(load_ref(req['RR_outputs']['visual']),None,load_ref(req['RR_outputs']['proprio']))
    flags=dict(inputs=[exact(x,y) for x,y in zip(args,(load_ref(x) for x in req['RR_inputs']),strict=True)],
        full3_outputs=[exact(out[0],expected[0]),out[1] is None,exact(out[2],expected[2])])
    require(all(flags['inputs']) and all(flags['full3_outputs']),'new RR full native IO bitwise prior RR; no tolerance')
    return flags


def empty_rows(m):
    return [{k:q[k] for k in ('index','root_slot','condition','task_id','selection_digest')}|dict(predictions={c:None for c in ARMS},
        inherited_T_AA=q['AA_outputs'],RR_verified=None,label=None,score=None) for q in m['requests']]


def seal_value(rows,expected):
    require(len(rows)==N and [r['index'] for r in rows]==list(range(N)) and all(r['RR_verified'] is not None and all(r['predictions'][c] is not None for c in ARMS) for r in rows), 'all492 new full output trees before labels')
    return dict(schema=NAME+'.prediction_seal',prepare_sha256=expected,predictions=[dict(index=r['index'],outputs=r['predictions'],inherited_T_AA=r['inherited_T_AA'],RR_verified=r['RR_verified']) for r in rows])


def labels_after_seal(m,rows,store,seal,expected):
    v=seal_value(rows,expected);require(read(store.path/seal['file'],seal['sha256'])==v,'complete durable492 seal')
    for q in m['requests']:
        store.check();store.emit('label_read_intent',branch=q['index'],source=q['label'])
        a=load_ref(q['label']);require(a.shape==(2,256,384),'original common batch2 label');yield q['index'],a


def effects(v):
    require(set(v)==set(CELLS),'all four temporal cells')
    converted=early.effects({old:v[new] for old,new in zip(early.CELLS,CELLS,strict=True)})
    return {('slot5' if k=='visual' else 'slot10' if k=='proprio' else k):x for k,x in converted.items()}


def row_score(cells,label):
    require(label.shape==(1,256,384) and all(x.shape==label.shape for x in cells.values()),'same final20 support')
    v={c:float(np.mean((x.astype(np.float64)-label.astype(np.float64))**2,dtype=np.float64)) for c,x in cells.items()}
    return dict(MSE=v,MSE_effects={k:float(x) for k,x in effects(v).items()},feature_effect_L2={k:float(np.linalg.norm(x)) for k,x in effects(cells).items()})


def features(req,row,load):
    cells={'T_AA':load_ref(req['AA_outputs']['visual'])[:,-1]}
    require(load_ref(req['AA_outputs']['proprio']).shape==(1,3,256,16),'complete inherited AA proprio output role')
    for c in ARMS:cells[c]=load(row['predictions'][c]['$tuple'][0]['$array'])[:,-1]
    return cells,load(row['label'])[1:2]


def pair(spec,a,b):
    renamed=lambda x:({old:x[0][new] for old,new in zip(early.CELLS,CELLS,strict=True)},x[1])
    p=early.pair(spec,renamed(a),renamed(b));p['cells']={new:p['cells'][old] for old,new in zip(early.CELLS,CELLS,strict=True)}
    p['response_effect_L2']={('slot5' if k=='visual' else 'slot10' if k=='proprio' else k):v for k,v in p['response_effect_L2'].items()};return p


def pairs(m,rows,load):
    return [pair(s,features(m['requests'][s['minus_index']],rows[s['minus_index']],load),features(m['requests'][s['plus_index']],rows[s['plus_index']],load)) for s in pair_schedule()]


def summaries(rows,ps):
    result=[];signs=('positive','negative','zero','null')
    for root,cond in [(None,None),(0,'off'),(0,'on'),(1,'off'),(1,'on')]:
        keep=lambda x:root is None or (x['root_slot'],x['condition'])==(root,cond)
        rs=[x for x in rows if keep(x)];xs=[x for x in ps if keep(x)];mean=lambda v:None if not v else float(np.mean(v,dtype=np.float64))
        result.append(dict(root_slot=root,condition=cond,branches=len(rs),pairs=len(xs),mean_MSE={c:mean([r['score']['MSE'][c] for r in rs]) for c in CELLS},
            mean_MSE_effects={k:mean([r['score']['MSE_effects'][k] for r in rs]) for k in ('slot5','slot10','interaction','joint')},
            responses={c:dict(observed_zero=sum(x['cells'][c]['observed_zero'] for x in xs),predicted_zero=sum(x['cells'][c]['predicted_zero'] for x in xs),
                signs={s:sum(x['cells'][c]['cosine_sign']==s for x in xs) for s in signs}) for c in CELLS},
            sign_transitions={a+'__to__'+b:{s+'__to__'+t:sum(x['cells'][a]['cosine_sign']==s and x['cells'][b]['cosine_sign']==t for x in xs)
                for s in signs for t in signs} for j,a in enumerate(CELLS) for b in CELLS[j+1:]}))
    return result


def execute(m,prepared,expected,output,backend_factory=CudaBackend,started=None):
    store=Store(output,started,Path(prepared).stat().st_size);trace=Trace(store);backend=None;rows=empty_rows(m)
    b=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected)
    r=dict(schema=NAME+'.report',status='partial_or_failed',protocol=protocol(),execution_binding=b,rows=rows,calls=trace.calls,errors=[],source_errors=[],
        pairs=[dict(s,missing=[s['minus_index'],s['plus_index']],cells=None) for s in pair_schedule()],summaries=None,prediction_seal=None,
        model_before=None,model_after=None,inherited=m['inherited'],P1_admission=False,P5_admission=False)
    load=lambda rec:short.abi.load_array(dict(directory=str(store.path),array=rec))
    try:
        store.put('execution_binding.json',raw(b));store.emit('run_started',binding=b,protocol=protocol())
        backend=backend_factory(m,trace);r.update(model_before=backend.before,strict_loads=backend.strict,imported_sources=backend.sources)
        for q,row in zip(m['requests'],rows,strict=True):
            a,rr=source_inputs(q)
            for cell in ARMS:
                args=splice(a,rr,cell);store.check();store.emit('history_ready',branch=q['index'],cell=cell,witness=witness(q,cell,args))
                backend.predict(args);call=trace.calls[-1];require(call['status']=='returned','complete validated native return')
                row['predictions'][cell]=deepcopy(call['outputs'])
                if cell=='T_RR':row['RR_verified']=rr_check(q,args,codec.restore(call['outputs'],load))
                store.emit('cell_saved',branch=q['index'],cell=cell,outputs=call['outputs'],RR_verified=row['RR_verified'] if cell=='T_RR' else None)
        require(len(trace.calls)==3*N and trace.guard.attempted==trace.observed_returns==trace.guard.returned==units() and store.tensor_bytes==protocol()['native_IO_bytes'],'all492 native paid/observed/validated bytes')
        r['model_after_predictions']=backend.receipt();require(r['model_after_predictions']==backend.before,'model unchanged before labels')
        store.emit('model_checked_before_labels',model=r['model_after_predictions'])
        r['prediction_seal']=store.put('predictions_seal.json',raw(seal_value(rows,expected)));store.emit('all_predictions_sealed',seal=r['prediction_seal'])
        for i,label in labels_after_seal(m,rows,store,r['prediction_seal'],expected):
            row=rows[i];row['label']=store.array(label);store.emit('label_saved',branch=i,label=row['label'])
            cells,obs=features(m['requests'][i],row,load);row['score']=row_score(cells,obs);store.emit('score_saved',branch=i,score=row['score'])
        r['pairs']=pairs(m,rows,load);r['summaries']=summaries(rows,r['pairs']);store.emit('all_scores_saved',pairs_digest=digest(r['pairs']),summaries_digest=digest(r['summaries']))
        require(store.tensor_bytes==protocol()['total_tensor_bytes'],'all native plus common labels')
        r['status']='all164_temporal_history_492_recorded'
    except BaseException as e:r['errors'].append(dict(type=type(e).__name__,message=str(e)))
    finally:
        if backend is not None:
            try:
                r['model_after']=backend.receipt();require(r['model_after']==backend.before,'model unchanged after scoring')
                backend.check_sources(m);r['memory']=backend.memory();r['attention_ops']=backend.attention_ops
            except BaseException as e:r['source_errors'].append(dict(type=type(e).__name__,message=str(e)))
        r['source_errors'].extend(trace.close())
        for path,h in m['bindings'].items():
            try:require(sha(path)==h,'final bound source/model/asset bytes')
            except BaseException as e:r['source_errors'].append(dict(path=path,type=type(e).__name__,message=str(e)))
        r.update(counts=dict(attempted=trace.guard.attempted,returned=trace.observed_returns,validated=trace.guard.returned),tensor_bytes=store.tensor_bytes,
            elapsed_seconds=time.monotonic()-store.started,external_prepare_bytes=store.external_bytes)
        if r['elapsed_seconds']>=WALL:r['errors'].append(dict(type='wall_cap',message='verify/load/dispatch/final exceeded900s'))
        if r['errors'] or r['source_errors']:r['status']='partial_or_failed'
        store.emit('run_finished',final=True,status=r['status'],calls_digest=digest(r['calls']),counts=r['counts'],errors=r['errors'],source_errors=r['source_errors'])
        r.update(artifacts=deepcopy(store.artifacts),journal_chain_tail=store.tail,output_bytes_before_report=store.total);store.put('report.json',raw(r),final=True)
    return dict(status=r['status'],path=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH and not os.environ.get('PYTHONPATH'),'explicit MAIN HOSTCUDA authority/no overlay')
    started=time.monotonic();m=verify(prepared,expected);return execute(m,prepared,expected,output,started=started)


def consume(output,expected):
    d=own(output);r=read(d/'report.json',expected);inv=r['artifacts']
    require(r['schema']==NAME+'.report' and r['protocol']==protocol() and len(r['rows'])==N and len(r['pairs'])==len(pair_schedule()) and
        r['P1_admission'] is r['P5_admission'] is False,'complete denominator/protocol closed gates')
    require(set(inv)|{'report.json'}=={p.name for p in d.iterdir()},'whole exact inventory')
    for name,rec in inv.items():
        p=d/name;require(Path(name).name==name and not p.is_symlink() and p.stat().st_size==rec['file_bytes'] and sha(p)==rec['sha256'],'all output bytes/SHA')
    require(sum(x['file_bytes'] for x in inv.values())==r['output_bytes_before_report'] and r['external_prepare_bytes']+r['output_bytes_before_report']+(d/'report.json').stat().st_size<=CAP,'full aggregate cap')
    events=list(codec.journal(d/'journal.jsonl'));require(events and events[-1]['event_sha256']==r['journal_chain_tail'],'canonical whole journal chain')
    if r['status']=='partial_or_failed':
        require(r['errors'] or r['source_errors'],'retained actual failure')
        return dict(schema=NAME+'.consume',status='blocked_partial_temporal',report_sha256=expected,branches=N,pairs=len(pair_schedule()),P1_admission=False,P5_admission=False,
            limitation='inventory/hashchain only; not complete semantics or scientific negative')
    b=r['execution_binding'];m=verify(b['prepared'],b['prepare_sha256'])
    require(r['status']=='all164_temporal_history_492_recorded' and not r['errors'] and not r['source_errors'] and r['elapsed_seconds']<WALL and
        r['external_prepare_bytes']==Path(b['prepared']).stat().st_size and r['inherited']==m['inherited'] and
        r['model_before']==r['model_after_predictions']==r['model_after']==m['record']['expected_model'] and r['strict_loads']==m['record']['expected_strict_loads'], 'actual same loaded model/schema and complete terminal state')
    require(all(m['bindings'].get(p)==sha(p) for p in r['imported_sources'].values()) and all(not any(k in op.lower() for k in ('flash_attention','efficient_attention','cudnn_attention')) for op in r['attention_ops']), 'actual source/observed attention guards')
    pos=0;idx=0;tensor_bytes=0;names={'execution_binding.json','journal.jsonl','predictions_seal.json'}
    def event(kind,**fields):
        nonlocal pos
        require(pos<len(events),'missing event');e=events[pos];pos+=1
        require(e==dict(sequence=e['sequence'],previous_sha256=e['previous_sha256'],event_sha256=e['event_sha256'],kind=kind,**fields),'exact durable event fields/order: '+kind)
    def load(rec):
        nonlocal idx,tensor_bytes
        require(rec['file']==f'tensor_{idx:05d}.npy' and inv.get(rec['file'])=={k:rec[k] for k in ('sha256','file_bytes')},'every actual tensor unique/ordered/inventoried')
        idx+=1;tensor_bytes+=rec['nbytes'];names.add(rec['file']);return short.abi.load_array(dict(directory=str(d),array=rec))
    reread=lambda rec:short.abi.load_array(dict(directory=str(d),array=rec))
    require(read(d/'execution_binding.json',inv['execution_binding.json']['sha256'])==b,'durable execution binding')
    event('run_started',binding=b,protocol=protocol());guard=Meter();fresh=empty_rows(m)
    require(len(r['calls'])==3*N,'exact492 no retries or padding')
    for q,row,base in zip(m['requests'],r['rows'],fresh,strict=True):
        require(all(row[k]==base[k] for k in ('index','root_slot','condition','task_id','selection_digest','inherited_T_AA')) and set(row['predictions'])==set(ARMS),'full identity/inherited AA/all three new slots')
        a,rr=source_inputs(q)
        for j,cell in enumerate(ARMS):
            c=r['calls'][3*q['index']+j];args=splice(a,rr,cell);event('history_ready',branch=q['index'],cell=cell,witness=witness(q,cell,args))
            actual,kw=codec.restore(c['inputs'],load);out=codec.restore(c['outputs'],load)
            require(c['index']==3*q['index']+j and c['T']==3 and c['status']=='returned' and kw=={} and len(actual)==3 and
                all(exact(x,y) for x,y in zip(actual,args,strict=True)), 'exact actual temporal inputs/action/latest')
            ni,no=c['native_inputs'],c['native_outputs']
            require(len(ni)==3 and all(s['shape']==list(x.shape) and s['dtype']==str(x.dtype) for s,x in zip(ni,actual,strict=True)),'actual input metadata vs NPY')
            u=guard.before('predictor',[x.shape for x in actual],[x['device'] for x in ni],[x['dtype'] for x in ni],[x['requires_grad'] for x in ni],c['inference'],c['TF32off'],c['requested_MATH'])
            require(c['units']==u['units'] and type(out) is tuple and len(out)==3 and out[1] is None and len(no)==2 and all(s['shape']==list(x.shape) and s['dtype']==str(x.dtype) and s['requires_grad'] is False for s,x in zip(no,(out[0],out[2]),strict=True)), 'native full3 visual/proprio shape/dtype/frozen metadata')
            guard.after([out[0].shape,None,out[2].shape],[x['device'] for x in no],[x['dtype'] for x in no])
            event('dispatch_intent',call={k:c[k] for k in ('index','T','units','native_inputs','inference','TF32off','requested_MATH')}|{'status':'attempted'})
            event('inputs_saved',index=c['index'],inputs=c['inputs']);event('return_observed',index=c['index']);event('outputs_saved',index=c['index'],outputs=c['outputs'],native_outputs=no)
            require(row['predictions'][cell]==c['outputs'],'full native output attribution')
            flags=rr_check(q,actual,out) if cell=='T_RR' else None
            if cell=='T_RR':require(row['RR_verified']==flags,'new RR bridge bitwise all native IO')
            event('cell_saved',branch=q['index'],cell=cell,outputs=c['outputs'],RR_verified=flags)
    require(tensor_bytes==protocol()['native_IO_bytes'],'all492 IO before labels')
    event('model_checked_before_labels',model=r['model_after_predictions']);seal=r['prediction_seal']
    require(seal==dict(file='predictions_seal.json',**inv['predictions_seal.json']) and read(d/seal['file'],seal['sha256'])==seal_value(r['rows'],b['prepare_sha256']),'whole full3 seal')
    event('all_predictions_sealed',seal=seal)
    for q,row in zip(m['requests'],r['rows'],strict=True):
        event('label_read_intent',branch=q['index'],source=q['label']);saved=load(row['label'])
        require(exact(saved,load_ref(q['label'])) and saved.shape==(2,256,384),'exact same batch2 labels')
        event('label_saved',branch=q['index'],label=row['label']);cells,obs=features(q,row,reread)
        require(row['score']==row_score(cells,obs),'all temporal MSE/effects independently recomputed');event('score_saved',branch=q['index'],score=row['score'])
    require(r['pairs']==pairs(m,r['rows'],reread) and r['summaries']==summaries(r['rows'],r['pairs']),'all80/fourstrata exact scalar/sign/null recomputation')
    event('all_scores_saved',pairs_digest=digest(r['pairs']),summaries_digest=digest(r['summaries']))
    counts=dict(attempted=units(),returned=units(),validated=units());require(r['counts']==counts and guard.returned==units(),'full492 actual counts')
    event('run_finished',status=r['status'],calls_digest=digest(r['calls']),counts=counts,errors=[],source_errors=[])
    require(pos==len(events) and names==set(inv) and tensor_bytes==r['tensor_bytes']==protocol()['total_tensor_bytes']<=TENSOR_CAP,'all events/artifacts/tensors counted')
    for p,h in m['bindings'].items():require(sha(p)==h,'final unchanged complete bindings')
    return dict(schema=NAME+'.consume',status='all164_temporal_two_corners_RR_bridge_and80_pairs_verified',report_sha256=expected,branches=N,pairs=len(pair_schedule()),
        predictor=units(),RR_all_native_IO_bitwise_verified=True,inherited_AA=True,prior_fullstrict_rebuilt=False,new_encoder=0,new_physics=0,P1_admission=False,P5_admission=False)


def main():
    require(N==164,'production fixed164')
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--output',required=True)
    for cmd in ('verify','run'):
        a=s.add_parser(cmd);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd=='run':a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=s.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha',required=True);a.add_argument('--receipt',required=True)
    q=p.parse_args()
    if q.command=='prepare':v=write_new(q.output,build())
    elif q.command=='verify':verify(q.prepared,q.prepare_sha);v=dict(status='temporal_prepare_verified',prior_fullstrict_rebuilt=False)
    elif q.command=='run':v=run(q.prepared,q.prepare_sha,q.output,q.authorization)
    else:v=write_new(q.receipt,consume(q.output,q.report_sha))
    print(json.dumps(v,sort_keys=True,allow_nan=False))
    if v['status'] in ('partial_or_failed','blocked_partial_temporal'):raise SystemExit(2)


if __name__=='__main__':main()
