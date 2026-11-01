"""Saved-only H1/H2/short-start common H2-label rescoring; no fullstrict/NN.

Requires an explicit actual COMPLETE short-start report AND external receipt.
Same labels do not make different startup/feedback histories equal protocols.
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

from experiments.spectral_reachability import agent_p5_short_start_full_roll_20261029_v1 as short

NAME='agent_p5_short_start_comparison_20261029_v1'
ROOT=short.ROOT
N=164
CAP,FLOOR,WALL=64*1024**2,200*1024**2,900
H1='agent_p5_raw164_jepa_cuda_20261029_v1'
H2='agent_p5_raw164_jepa_h2_cuda_20261029_v2'
SHORT=short.NAME
FIXED={
 'H1':dict(stem=H1,report_sha='e7ac4ef9eb7491f619de25b04c55f5e56560e2833a840dd5305f3db6207b3a5d',
    prepared_sha='222d917f262e2a7c8e347fd351e3b0927ec4ffe200035e7b3b971a7abf0000fe',
    receipt_sha='4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759'),
 'H2':dict(stem=H2,report_sha=short.abi.REPORT_SHA,prepared_sha=short.abi.PREPARE_SHA,receipt_sha=short.abi.STRICT_SHA)}
SOURCE_PINS={H1:'610523392e1923e4371556606109def7f882a4a90741e2e14b63af68880e2180',
 H2:'b661a7c27181ef29f8277e154006f9238ec1c94487c311c1065e4d42fcfc2241',
 SHORT:'6b62eead489a3b2c89f6d478cee77c9b3ec9d25253c27b9e293d01beffb56ece'}
PHASES=dict(H1_20=20,H2_15=15,H2_20=20,short_5=5,short_10=10,short_15=15,short_20=20)
COMPARISONS=(('H2_15','short_15'),('H1_20','H2_20'),('H1_20','short_20'),('H2_20','short_20'))
require,sha,raw,exact=short.require,short.sha,short.raw,short.exact
schedule=short.abi.pre.pair_schedule
expected_short_protocol=short.protocol


def protocol():
    return dict(branches=N,pairs=schedule(),phases=PHASES,contrasts=[dict(reference=a,candidate=b,definition='candidate minus reference') for a,b in COMPARISONS],
        common_labels='H2 batch3 slots1/2 at5/10; H2 batch2 slots0/1 at15/20',
        missing_old5_10='null: original full3 early outputs are not causal short-start predictions',
        zero='all pairs retained; exact zero norm -> null cosine; no epsilon/cohort selection',
        scoring='float64 C-order mean squared all256patch*384; no fitted Q or weights',
        startup_and_feedback_protocols_differ=True,extra_agent_proprio=True,RGB_only=False,fresh=False,known_roots=2,
        equal_FLOPs=False,causal_mechanism_certified=False,scientific_improvement_claim=False,
        upstream_fullstrict_rebuilt=False,new_neural_calls=0,new_physics_calls=0,new_training=0,
        output_cap=CAP,disk_floor=FLOOR,wall_seconds=WALL,P1_admission=False,P5_admission=False)


def checked(path,expected,used):
    p=Path(path).absolute()
    require(p.resolve()==p and p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)) and
        type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'explicit canonical regular file/SHA')
    require(str(p) not in used or used[str(p)]==expected,'consistent exact file binding')
    if str(p) not in used:require(sha(p)==expected,'actual file bytes SHA')
    used[str(p)]=expected;return p


def read(path,expected,used):
    p=checked(path,expected,used);require(p.stat().st_size<=CAP,'bounded metadata')
    v=json.loads(p.read_bytes(),object_pairs_hook=short.codec.bridge.pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    raw(v);return v


def inventory(directory,r,used):
    require(set(r['artifacts'])|{'report.json'}=={p.name for p in directory.iterdir()},'complete exact saved inventory')
    for name,ref in r['artifacts'].items():
        require(Path(name).name==name,'flat original artifact')
        p=checked(directory/name,ref['sha256'],used);require(p.stat().st_size==ref['file_bytes'],'original file byte count')
    tail=None
    for e in short.codec.journal(directory/'journal.jsonl'):tail=e['event_sha256']
    require(tail is not None and tail==r['journal_chain_tail'],'original canonical journal integrity, not fullstrict replay')


def descriptors(tree):
    if not isinstance(tree,dict):return []
    if '$array' in tree:return [tree['$array']]
    result=[]
    for value in tree.values():
        for child in value if isinstance(value,list) else value.values() if isinstance(value,dict) else []:result.extend(descriptors(child))
    return result


def resources(kind,r):
    zero=lambda:dict(api=0,batch_samples=0,frame_samples=0)
    counts={k:zero() for k in short.codec.KEYS};names=set();native_bytes=0
    expected_calls=(3*N+N if kind=='H1' else 4*N+N if kind=='H2' else 4*N)
    require(len(r['calls'])==expected_calls,'complete module calls, not chosen predictor-only subset')
    for i,c in enumerate(r['calls']):
        require(c['index']==i and c['status']=='returned','all ordered actual returns')
        args=c['inputs']['$tuple'][0]['$tuple'];shape=args[0]['$array']['shape']
        if kind=='short':
            key='predictor';t=(1,2,3,3)[i%4]
            require(c['T']==t and shape==[1,t,1,16,16,384],'four genuine short-start calls')
        else:
            width=3 if kind=='H1' else 4
            expected_key=(('official_encoder','proprio_encoder','predictor') if width==3 else
                ('official_encoder','proprio_encoder','predictor','predictor'))[i%width] if i<width*N else 'official_encoder'
            key=c['module'];require(key==expected_key and c['phase']==('prediction' if i<width*N else 'label') and
                c['branch']==(i//width if i<width*N else i-width*N),'literal original per-module schedule')
        u=dict(api=1,batch_samples=shape[0],frame_samples=shape[0] if key=='official_encoder' else shape[0]*shape[1])
        require(c['units']==u,'independent actual API/sample/frame arithmetic from native input shape')
        for k,v in u.items():counts[key][k]+=v
        for rec in descriptors(c['inputs'])+descriptors(c['outputs']):
            require(rec['file'] not in names and rec['dtype']=='float32' and rec['nbytes']==4*int(np.prod(rec['shape'])) and
                r['artifacts'][rec['file']]=={k:rec[k] for k in ('sha256','file_bytes')},'complete unique native IO descriptor census')
            names.add(rec['file']);native_bytes+=rec['nbytes']
    if kind=='short':
        require(r['counts']==dict(attempted=counts['predictor'],returned=counts['predictor'],validated=counts['predictor']) and
            native_bytes==7373160*N and r['tensor_bytes']==(7373160+1966080)*N,'separate short native and copied-label bytes')
    else:
        require(r['counts']['attempted']==r['counts']['returned']==counts==r['protocol']['caps'] and
            native_bytes==r['actual_tensor_io_bytes'],'complete old counters and IO effective bytes')
    return dict(attempted=deepcopy(counts),observed_returned=deepcopy(counts),validated_returned=deepcopy(counts),
        native_tensor_IO_bytes=native_bytes,total_saved_tensor_bytes=r['tensor_bytes'] if kind=='short' else r['actual_tensor_io_bytes'],
        wall_seconds=r['elapsed_seconds'],memory=deepcopy(r['memory']),model_before=deepcopy(r['model_before']),
        model_after=deepcopy(r['model_after']),loaded_config_checkpoint_bytes='inherited exact prepare file bindings + original load receipts, not newly reloaded',
        reported_AD_calls=r['protocol'].get('new_AD'),reported_physics_calls=r['protocol'].get('new_physics'),reported_training=r['protocol'].get('new_training'),
        inherited_cost_source=deepcopy(r.get('inherited_cost_source')),new_analysis_encoder=0,new_analysis_predictor=0,new_analysis_physics=0,
        FLOPs=None,internal_module_cost_not_added_to_top_API=True)


def bundle(kind,output,report_sha,receipt,receipt_sha,used):
    d=Path(output).absolute();stem=H1 if kind=='H1' else H2 if kind=='H2' else SHORT
    r=read(d/'report.json',report_sha,used);c=read(receipt,receipt_sha,used)
    status={'H1':('fixed164_actual15_conditioned_factual_H1_recorded','all164_factual_H1_and80_pairs_verified'),
        'H2':('fixed164_root10_open_loop_H2_recorded','all164_open_loop_H2_and80_pairs_verified'),
        'short':('all164_short_start_causal_roll_recorded','all164_short_start_causal_roll_saved_verified')}[kind]
    require(r['schema']==stem+'.report' and c['schema']==stem+'.consume' and (r['status'],c['status'])==status and
        c['report_sha256']==report_sha and r['errors']==r['source_errors']==[] and len(r['rows'])==N and
        r['model_before']==r['model_after_predictions']==r['model_after'] and
        all(x['P1_admission'] is x['P5_admission'] is False for x in (r,c)),'actual complete external report + own complete strict; never accept partial')
    require((c['branches'] if kind=='short' else c['requests'])==N and c['pairs']==len(schedule()),'complete explicit receipt denominators')
    binding=r['execution_binding'];p=read(binding['prepared'],binding['prepare_sha256'],used)
    require(p['schema']==stem+'.prepare','exact original prepare')
    if kind!='short':require(binding['prepare_sha256']==FIXED[kind]['prepared_sha'] and p['protocol']==r['protocol'] and c['counts']==r['counts'],'fixed original prepare/strict counts')
    else:require(p['authority']['protocol']==r['protocol']==expected_short_protocol() and c['predictor']==r['counts']['validated'],'new actual prepare/protocol/strict counter crosslink')
    source=ROOT/'experiments/spectral_reachability'/(stem+'.py')
    checked(source,SOURCE_PINS[stem],used)
    require(p['bindings'].get(str(source))==SOURCE_PINS[stem],'actual runtime source bound by its prepare')
    for path,h in p['bindings'].items():checked(path,h,used)
    require([{k:row[k] for k in spec} for row,spec in zip(r['pairs'],schedule(),strict=True)]==schedule() and
        all(x['missing']==[] for x in r['pairs']),'all original80 ordered pairs, none dropped including zero')
    inventory(d,r,used)
    cost=resources(kind,r)
    return dict(directory=d,report=r,prepared=p,receipt=c,resources=cost)


def load_array(b,rec,shape,used):
    require(Path(rec['file']).name==rec['file'] and rec['dtype']=='float32' and rec['shape']==list(shape) and
        rec['nbytes']==4*int(np.prod(shape)) and b['report']['artifacts'][rec['file']]=={k:rec[k] for k in ('sha256','file_bytes')},'exact selected native tensor pointer/inventory')
    p=checked(b['directory']/rec['file'],rec['sha256'],used);require(p.stat().st_size==rec['file_bytes']<=4*1024**2,'bounded original NPY')
    body=p.read_bytes();buf=io.BytesIO(body);a=np.load(buf,allow_pickle=False)
    require(buf.tell()==len(body) and a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'single full finite float32 NPY, no trailer')
    return a


def raw32(v,shape):
    a=np.asarray(v,np.float64);require(a.shape==shape and np.isfinite(a).all() and np.array_equal(a,a.astype(np.float32).astype(np.float64)),'original lossless raw float32 action history')
    return a.astype(np.float32)


def identity(i,b1,b2,bs):
    a,b=b1['prepared']['bridge']['requests'][i],b2['prepared']['bridge']['requests'][i]
    record=bs['prepared']['authority']['rows'][i]['record'];row=bs['report']['rows'][i]
    for x in (a,b,record,b1['report']['rows'][i],b2['report']['rows'][i],row['identity']):
        require((x['root_slot'],x['condition'])==((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41] and
            x['task_id']==a['task_id'] and x['selection_digest']==a['selection_digest'],'same exact root/condition/task/selection')
    require(all(x['index']==i for x in (a,b,record,row,b1['report']['rows'][i],b2['report']['rows'][i])),'same literal original branch')
    require(a['context_global_frames']==[5,10,15] and a['successor_global_frame']==20 and b['context_global_frames']==[0,5,10] and
        b['successor_global_frames']==[15,20] and record['parent_frames']==[0,5,10] and record['frames']==[0],'distinct original time protocols, no remapping')
    require(a['label']==b['labels'][1] and a['images'][0]==b['images'][1] and a['images'][2]==b['labels'][0] and
        [(x['phase'],x['step']) for x in a['images']]==[('warmup',5),('plan',0),('plan',5)] and
        [(x['phase'],x['step']) for x in b['images']]==[('warmup',0),('warmup',5),('warmup',10)] and
        record['image0']==b['images'][0] and (a['label']['phase'],a['label']['step'],a['label']['condition'],a['label']['overlay_index'])==
        ('plan',10,a['condition'],int(a['condition']=='on')),'same original actual RGB/terminal/overlay source history')
    require(exact(raw32(a['commands'],(15,2)),raw32(b['commands'],(20,2))[5:]) and
        exact(raw32(record['source_normalization']['commands'],(20,2)),raw32(b['commands'],(20,2))) and
        exact(np.asarray(a['packets'],np.float32),np.asarray(b['packets'],np.float32)[1:]),'unchanged original raw6..20 and raw1..20, no action swap')
    require(b1['prepared']['bridge']['stats']==b2['prepared']['bridge']['stats']==record['source_normalization']['stats'] and
        record['source_normalization']['packets']==b['packets'],'same actual normalizer/packet source')
    require(record['refs']['encoder_patch']['array']==b2['report']['rows'][i]['context_patch'] and
        record['refs']['encoder_patch']['directory']==str(b2['directory']) and
        record['inherited_authority']['report_sha256']==FIXED['H2']['report_sha'] and
        bs['prepared']['authority']['rows'][i]['labels']==[record['refs']['encoder_patch'],dict(directory=str(b2['directory']),array=b2['report']['rows'][i]['label_patch'],report_json_pointer=f'/rows/{i}/label_patch')],
        'new labels reference the ORIGINAL H2 bank, not a same-shaped replacement')
    return {k:a[k] for k in ('index','root_slot','condition','task_id','selection_digest')}|dict(history_source_seams=dict(
        frame10_H1=a['images'][1],frame10_H2=b['images'][2],frame10_same_source_record=a['images'][1]==b['images'][2],
        frame10_same_pixel_sha=a['images'][1]['pixel_sha256']==b['images'][2]['pixel_sha256'],
        reason='H1 plan0 is AFTER goal bind; H2 warmup10 is BEFORE goal bind; preserve, never relabel as matched history'))


def mse(pred,label):
    require(pred.shape==label.shape==(1,256,384),'same final-time all-patch feature support')
    return float(np.mean((pred.astype(np.float64)-label.astype(np.float64))**2,dtype=np.float64))


def response(pm,pp,obs_m,obs_p):
    dp=(pp.astype(np.float64)-pm.astype(np.float64)).ravel(order='C');do=(obs_p.astype(np.float64)-obs_m.astype(np.float64)).ravel(order='C')
    pn=float(np.linalg.norm(dp));on=float(np.linalg.norm(do));dot=float(np.dot(dp,do))
    return dict(predicted_L2=pn,observed_L2=on,dot=dot,cosine=None if pn==0 or on==0 else dot/(pn*on),
        predicted_zero=pn==0,observed_zero=on==0,response_error_L2=float(np.linalg.norm(dp-do)))


def sign(v):return 'null' if v is None else 'positive' if v>0 else 'negative' if v<0 else 'zero'


def collect(i,b1,b2,bs,used):
    identity_row=identity(i,b1,b2,bs);r1,r2,rs=(b['report']['rows'][i] for b in (b1,b2,bs))
    load=lambda b,rec,shape:load_array(b,rec,shape,used)
    h1=load(b1,r1['prediction'],(1,3,256,384))[:,-1]
    p15=load(b2,r2['prediction15'],(1,3,256,384))[:,-1];p20=load(b2,r2['prediction'],(1,3,256,384))[:,-1]
    ctx=load(b2,r2['context_patch'],(3,256,384));future=load(b2,r2['label_patch'],(2,256,384))
    native_refs=[b2['report']['calls'][4*i]['outputs']['$dict']['x_norm_patchtokens']['$array'],b2['report']['calls'][4*N+i]['outputs']['$dict']['x_norm_patchtokens']['$array']]
    require([r2['context_patch'],r2['label_patch']]==native_refs,'common H2 labels attached to actual encoder calls')
    obs={5:ctx[1:2].copy(),10:ctx[2:3].copy(),15:future[:1].copy(),20:future[1:2].copy()}
    label1=load(b1,r1['label_patch'],(1,256,384))
    require(mse(h1,label1)==r1['score']['successor_MSE'] and mse(p15,obs[15])==r2['score']['prediction15_MSE'] and mse(p20,obs[20])==r2['score']['successor_MSE'],'original saved scalar independently reproduced before common-label change')
    require(r1['prediction']==b1['report']['calls'][3*i+2]['outputs']['$tuple'][0]['$array'] and
        r2['prediction15']==b2['report']['calls'][4*i+2]['outputs']['$tuple'][0]['$array'] and
        r2['prediction']==b2['report']['calls'][4*i+3]['outputs']['$tuple'][0]['$array'],'exact original native final output refs')
    saved_labels=[load(bs,rs['labels'][j],shape) for j,shape in enumerate(((3,256,384),(2,256,384)))]
    require(exact(saved_labels[0],ctx) and exact(saved_labels[1],future),'new saved label bank byte-identical to common H2, batch seam never substituted')
    predictions=dict(H1_20=h1,H2_15=p15,H2_20=p20);provenance={}
    for j,t in enumerate((5,10,15,20)):
        stage=rs['stages'][j];call=bs['report']['calls'][4*i+j]
        require(stage==dict(target=t,frames=list(short.FRAMES[j]),call=4*i+j,prediction=call['outputs']),'new four explicit time/output identities')
        rec=stage['prediction']['$tuple'][0]['$array'];predictions[f'short_{t}']=load(bs,rec,(1,(1,2,3,3)[j],256,384))[:,-1]
        provenance[f'short_{t}']=dict(report_pointer=f'/rows/{i}/stages/{j}/prediction/$tuple/0/$array',array=rec)
    common={key:mse(p,obs[PHASES[key]]) for key,p in predictions.items()}
    require(rs['score']['MSE']==[common[f'short_{t}'] for t in (5,10,15,20)],'new declared scalars exactly recomputed on common labels')
    for key,b,rec,pointer in (('H1_20',b1,r1['prediction'],f'/rows/{i}/prediction'),('H2_15',b2,r2['prediction15'],f'/rows/{i}/prediction15'),('H2_20',b2,r2['prediction'],f'/rows/{i}/prediction')):
        provenance[key]=dict(report_pointer=pointer,array=rec)
    seam=dict(bitwise_equal=exact(label1,obs[20]),maxabs=float(np.max(np.abs(label1.astype(np.float64)-obs[20].astype(np.float64)))),
        MSE=mse(label1,obs[20]),old_H1_own_label_MSE=mse(h1,label1),old_H1_common_H2_label_MSE=common['H1_20'])
    row=dict(identity_row,common_MSE=common,contrasts={a+'__to__'+b:common[b]-common[a] for a,b in COMPARISONS},
        availability=dict(H1={str(t):common['H1_20'] if t==20 else None for t in (5,10,15,20)},
            H2={str(t):common['H2_'+str(t)] if t in (15,20) else None for t in (5,10,15,20)},short={str(t):common['short_'+str(t)] for t in (5,10,15,20)}),
        H1_label_batch_seam=seam,prediction_provenance=provenance,
        common_label_provenance=dict(batch3=r2['context_patch'],batch2=r2['label_patch'],report_sha256=FIXED['H2']['report_sha']))
    return row,dict(predictions=predictions,observations=obs,H1_original_label=label1)


def pair(spec,minus,plus):
    a,b=minus['predictions'],plus['predictions'];out={}
    for key,t in PHASES.items():
        v=response(a[key],b[key],minus['observations'][t],plus['observations'][t]);v['cosine_sign']=sign(v['cosine'])
        v['central_predicted_L2']=float(np.linalg.norm((b[key].astype(np.float64)-a[key].astype(np.float64)).ravel()/(2*spec['h'])))
        v['central_observed_L2']=float(np.linalg.norm((plus['observations'][t].astype(np.float64)-minus['observations'][t].astype(np.float64)).ravel()/(2*spec['h'])))
        out[key]=v
    legacy=response(a['H1_20'],b['H1_20'],minus['H1_original_label'],plus['H1_original_label'])
    return dict(spec,missing=[],phases=out,legacy_H1_label_observed_zero=legacy['observed_zero'],
        common_H2_label_observed_zero=out['H1_20']['observed_zero'],
        signs={a+'__to__'+b:[out[a]['cosine_sign'],out[b]['cosine_sign']] for a,b in COMPARISONS})


def summaries(rows,pairs):
    groups=[dict(scope='all',root_slot=None,condition=None)]+[dict(scope='root_condition',root_slot=r,condition=c) for r,c in ((0,'off'),(0,'on'),(1,'off'),(1,'on'))]
    result=[]
    for g in groups:
        accept=lambda x:g['scope']=='all' or (x['root_slot'],x['condition'])==(g['root_slot'],g['condition'])
        rs=[x for x in rows if accept(x)];ps=[x for x in pairs if accept(x)]
        result.append(dict(g,branches=len(rs),pairs=len(ps),mean_common_MSE={p:None if not rs else float(np.mean([x['common_MSE'][p] for x in rs],dtype=np.float64)) for p in PHASES},
            mean_contrasts={a+'__to__'+b:None if not rs else float(np.mean([x['contrasts'][a+'__to__'+b] for x in rs],dtype=np.float64)) for a,b in COMPARISONS},
            responses={p:dict(observed_zero=sum(x['phases'][p]['observed_zero'] for x in ps),predicted_zero=sum(x['phases'][p]['predicted_zero'] for x in ps),
                signs={s:sum(x['phases'][p]['cosine_sign']==s for x in ps) for s in ('positive','negative','zero','null')}) for p in PHASES},
            sign_transitions={a+'__to__'+b:{s+'__to__'+t:sum(x['signs'][a+'__to__'+b]==[s,t] for x in ps)
                for s in ('positive','negative','zero','null') for t in ('positive','negative','zero','null')} for a,b in COMPARISONS}))
    return result


def analyze(short_output,short_report_sha,short_receipt,short_receipt_sha):
    started=time.monotonic();used={}
    args=dict(short_output=str(Path(short_output).absolute()),short_report_sha=short_report_sha,short_receipt=str(Path(short_receipt).absolute()),short_receipt_sha=short_receipt_sha)
    # Authenticate NEW completion first; never silently fall back to old-only analysis.
    bs=bundle('short',args['short_output'],short_report_sha,args['short_receipt'],short_receipt_sha,used)
    old=[]
    for key in ('H1','H2'):
        pin=FIXED[key];old.append(bundle(key,ROOT/'results'/(pin['stem']+'_cuda_MAIN01'),pin['report_sha'],
            ROOT/'results'/(pin['stem']+'_consume_MAIN01.json'),pin['receipt_sha'],used))
    b1,b2=old
    require(b1['report']['model_before']==b2['report']['model_before']==bs['report']['model_before'] and
        b1['report']['strict_loads']==b2['report']['strict_loads']==bs['report']['strict_loads'],'same frozen actual state/load schemas across protocols')
    for path in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):checked(path,sha(path),used)
    rows=[];pair_rows=[]
    for start in range(0,N,41):
        cache={}
        for i in range(start,min(start+41,N)):
            require(time.monotonic()-started<WALL,'saved-only900s wall, no fallback/downselection')
            row,data=collect(i,b1,b2,bs,used);rows.append(row);cache[i]=data
        for spec in schedule():
            if start<=spec['minus_index']<min(start+41,N):pair_rows.append(pair(spec,cache[spec['minus_index']],cache[spec['plus_index']]))
    require(len(rows)==N and len(pair_rows)==len(schedule()) and [{k:r[k] for k in s} for r,s in zip(pair_rows,schedule(),strict=True)]==schedule(),'full164/80 exact order')
    for path,h in used.items():require(sha(path)==h,'all bound files stable after saved arithmetic')
    require(time.monotonic()-started<WALL,'saved job final hashes within900s')
    return dict(schema=NAME+'.analysis',status='all164_and80_common_H2_label_saved_rescored',arguments=args,protocol=protocol(),rows=rows,pairs=pair_rows,
        summaries=summaries(rows,pair_rows),resources=dict(H1=b1['resources'],H2=b2['resources'],short=bs['resources']),bindings=used,
        evidence=dict(new_report_sha256=short_report_sha,new_complete_strict_sha256=short_receipt_sha,
            old_fixed=deepcopy(FIXED),upstream_fullstrict_rebuilt=False,comparison_arithmetic_recomputed=True),
        bound_file_bytes=sum(Path(p).stat().st_size for p in used),new_neural_calls=0,new_physics_calls=0,P1_admission=False,P5_admission=False)


def write_new(output,value):
    p=Path(output).absolute();body=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and not any(x.is_symlink() for x in (p,*p.parents)) and
        len(body)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'exclusive own bounded saved output/floor')
    with p.open('xb') as f:require(f.write(body)==len(body),'complete metadata');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


def verify_analysis(analysis,expected):
    old=read(analysis,expected,{})
    require(old==analyze(**old['arguments']),'complete deterministic saved comparison independently recomputed')
    return dict(status='all164_and80_common_label_comparison_recomputed',analysis_sha256=expected,new_neural_calls=0,new_physics_calls=0)


def main():
    require(N==164,'fixed production denominator')
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('analyze')
    for key in ('short-output','short-report-sha','short-receipt','short-receipt-sha','output'):a.add_argument('--'+key,required=True)
    a=s.add_parser('verify');a.add_argument('--analysis',required=True);a.add_argument('--analysis-sha',required=True)
    q=p.parse_args()
    if q.command=='analyze':
        args=vars(q).copy();args.pop('command');output=args.pop('output');value=write_new(output,analyze(**args))
    else:value=verify_analysis(q.analysis,q.analysis_sha)
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
