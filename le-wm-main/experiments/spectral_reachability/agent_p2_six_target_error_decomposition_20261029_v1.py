"""Local saved-array arithmetic conditioned on MAIN strict; never imports fullstrict."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

NAME='agent_p2_six_target_error_decomposition_20261029_v1'
RGB='agent_p2_six_target_rgb_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
ARMS=('prior_FIRST','program_first_postcondition','arbitrary')
CP=(0,5,10,15)
META,ARRAY,CAP,FLOOR,WALL=128*1024**2,2*1024**2,16*1024**2,200*1024**2,120
ACTUAL={
 'prepare':(RGB+'_prepare_MAIN01.json','14aace0fad5c941fce475a41d45bb1dfa1bad9388cba9f767bdd4eb42aa4f393'),
 'report':(RGB+'_cuda_MAIN01/report.json','26e00d955ee2c106e4452b97ddca26a7731d721f7e00605f5750e3a38981460f'),
 'completion':(RGB+'_cuda_MAIN01/completion.json','f54b1ba7fc7657db1b738f1fe8be5ca280d790a14d5fae0a1a5326d0ac5bcb23'),
 'receipt':(RGB+'_consume_MAIN01.json','c16d99ba82c8d3adeb68515e2035cc191bd060ab6c273e07ff21cef9a23d870b')}
PINS={
 'innerdocs/agent_p2_six_target_error_decomposition_protocol_20261029_v1.md':'74f1d02c3282ebfaf7b30828156137dd9e1a4a207527d5694c0c7abe13ff59d9',
 'innerdocs/agent_p2_six_target_error_decomposition_rgb_abi_20261029_v1.md':'16c32ff20d249fdace43761f2c3da7c224826f310d04cbe4cd4f51c28cf2036f',
 'innerdocs/agent_p2_six_target_error_decomposition_authority_20261029_v1.md':'4946d6aa05df19f4151bad5685d84e80d25ffbb102bc986d65ef6d9e14c3d017',
 'experiments/spectral_reachability/'+RGB+'.py':'5a3ab9162f527ba6e51595b53cff095e095fbbcb596e8567eb29eac88bc2168e'}
SCALARS=('native7_distance','native7_reward','native7_squared','joint_xy_error','joint_xy_squared',
 'agent_xy_error','agent_xy_squared','block_xy_error','block_xy_squared','raw_angle_squared',
 'agent_velocity_error','agent_velocity_squared','native_angle_term','wrapped_angle_error')
ERRORS=tuple(k for k in SCALARS if k!='native7_reward')
GROUPS=('agent_xy_squared','block_xy_squared','raw_angle_squared','agent_velocity_squared')


def need(ok,message):
    if not ok:raise ValueError(message)


def encode(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def equal(a,b):return encode(a)==encode(b)
def digest(b):return hashlib.sha256(b).hexdigest()


def sha(path,cap=META):
    p=Path(path);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'bounded regular input '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def decode(b):
    def unique(pairs):
        out={}
        for k,v in pairs:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    v=json.loads(b,object_pairs_hook=unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    encode(v);return v


def read(path,h=None):
    actual=sha(path)
    if h is not None:need(actual==h,'external SHA '+str(path))
    b=Path(path).read_bytes();need(digest(b)==actual,'unchanged metadata read');return decode(b)


def actual_refs():return {k:dict(path=str(ROOT/'results'/p),sha256=h) for k,(p,h) in ACTUAL.items()}


def closure():
    out={}
    for name,h in PINS.items():
        p=ROOT/name;need(sha(p)==h,'frozen protocol/source pin');out[str(p)]=h
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):out[str(p)]=sha(p)
    return out


def f32(v,shape=(192,)):
    a=np.asarray(v)
    need(a.dtype.kind in 'fi' and a.shape==shape and np.isfinite(a).all(),'numeric finite original shape')
    x=a.astype(np.float32)
    need(np.isfinite(x).all() and np.array_equal(x.astype(np.float64),a.astype(np.float64)),'lossless original f32 values')
    return x


def vector(v):
    need(type(v) is np.ndarray and v.dtype==np.float32 and v.shape==(192,) and np.isfinite(v).all(),'original f32 latent192')
    return v.astype(np.float64)


def mean64(x):
    need(np.asarray(x).shape==(192,) and np.isfinite(x).all(),'finite f64 vector arithmetic')
    value=math.fsum(float(a) for a in x)/192;need(math.isfinite(value),'finite reduction');return value


def mse(a,b):
    d=vector(a)-vector(b);return mean64(d*d)


def decompose(p,z,t,g):
    pp,tt,gg=vector(p),vector(t),vector(g)
    sep=mse(t,g)
    def tracking(x):
        if x is None:return None
        xx=vector(x);own=mse(x,t);task=mse(x,g);cross=2*mean64((xx-tt)*(tt-gg))
        reconstructed=math.fsum((own,sep,cross))
        return dict(own_target_tracking_MSE=own,target_goal_geometric_separation=sep,
            signed_tracking_goal_cross_term=cross,original_task_goal_MSE=task,
            reconstructed_task_MSE=reconstructed,identity_roundoff_residual=task-reconstructed)
    pred_tracking=tracking(p);actual_tracking=tracking(z)
    residuals={}
    for name,r in (('task_goal',g),('tracking_target',t)):
        mp=mse(p,r)
        if z is None:
            residuals[name]=dict(model_goal_MSE=mp,actual_goal_MSE=None,prediction_residual_energy=None,
                signed_residual_cross_term=None,model_minus_actual_goal_MSE=None,reconstructed_model_MSE=None,identity_roundoff_residual=None)
        else:
            zz=vector(z);e=pp-zz;energy=mean64(e*e);cross=2*mean64(e*(zz-vector(r)));az=mse(z,r)
            reconstructed=math.fsum((az,energy,cross))
            residuals[name]=dict(model_goal_MSE=mp,actual_goal_MSE=az,prediction_residual_energy=energy,
                signed_residual_cross_term=cross,model_minus_actual_goal_MSE=mp-az,
                reconstructed_model_MSE=reconstructed,identity_roundoff_residual=mp-reconstructed)
    return dict(residual=residuals,tracking=dict(model=pred_tracking,actual=actual_tracking),target_goal_geometric_separation=sep)


def native_scalars(state,goal):
    s,g=np.asarray(state),np.asarray(goal)
    need(s.shape==g.shape==(7,) and s.dtype.kind in 'fi' and g.dtype.kind in 'fi','original numeric state7')
    s,g=s.astype(np.float64),g.astype(np.float64);need(np.isfinite(s).all() and np.isfinite(g).all(),'finite native7')
    d=s-g;sq=d*d;xy=float(np.linalg.norm(d[:4]));angle=float(min(abs(d[4]),2*np.pi-abs(d[4])))
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))))
    return dict(state=s.tolist(),goal=g.tolist(),residual7=d.tolist(),native7_distance=float(np.linalg.norm(d)),
        native7_reward=-float(np.linalg.norm(d)),native7_squared=float(np.sum(sq,dtype=np.float64)),joint_xy_error=xy,
        joint_xy_squared=float(np.sum(sq[:4],dtype=np.float64)),agent_xy_error=float(np.linalg.norm(d[:2])),
        agent_xy_squared=float(np.sum(sq[:2],dtype=np.float64)),block_xy_error=float(np.linalg.norm(d[2:4])),
        block_xy_squared=float(np.sum(sq[2:4],dtype=np.float64)),raw_angle_squared=float(sq[4]),
        agent_velocity_error=float(np.linalg.norm(d[5:])),agent_velocity_squared=float(np.sum(sq[5:],dtype=np.float64)),
        native_angle_term=angle,negative_native_angle=bool(angle<0),wrapped_angle_error=wrapped,
        native_success=bool(xy<20 and angle<np.pi/9),wrapped_success_diagnostic=bool(xy<20 and wrapped<np.pi/9),
        state7_order=['agent_x','agent_y','block_x','block_y','block_angle_mod2pi','agent_vx','agent_vy'])


def native_decomposition(n):
    if n is None:return None
    groups={k:n[k] for k in GROUPS};total=math.fsum(groups.values())
    return dict(groups=groups,group_sum=total,native7_squared=n['native7_squared'],identity_roundoff_residual=n['native7_squared']-total)


def classify(a,b):
    for x in (a,b):need(x is None or (type(x) in (int,float) and math.isfinite(x)),'finite sign input or null')
    if a is None or b is None:return dict(kind='null',zero_kind=None)
    if a==0 or b==0:return dict(kind='tie',zero_kind='both_zero' if a==b==0 else 'first_zero' if a==0 else 'second_zero')
    return dict(kind='strict_reversal' if ((a<0 and b>0) or (a>0 and b<0)) else 'same_nonzero_sign',zero_kind=None)


def sub(a,b):return None if a is None or b is None else a-b


def original_rows(p,r,latents):
    """Recompute only scalar/identity rows; no upstream modules/consumer called."""
    need(len(p['rows'])==len(r['rows'])==len(latents)==24,'all24 positions')
    out=[];new=[]
    for i,(s,row,z) in enumerate(zip(p['rows'],r['rows'],latents,strict=True)):
        key=(i,i//4,(i//4)//3,ARMS[(i//4)%3],CP[i%4])
        need((s['index'],s['slot'],s['root'],s['arm'],s['cp_public_step'])==key and s['condition']=='off','fixed slot/root/cp identity')
        mutable={'status','latent','pixels','preprocessed','modules'}
        need(equal({k:v for k,v in s.items() if k not in mutable},{k:v for k,v in row.items() if k not in mutable}),'sealed input fields identical')
        pp,t,g=(f32(s[k]) for k in ('predicted192','target192','goal192'))
        need(digest(pp.tobytes())==s['predicted_sha256'],'original prediction bytes SHA')
        need(s['prediction_origin']==('observed_history' if i%4==0 else 'paid_actual_deployment_tokens'),'actual-token predictions only')
        need(s['prediction_pointer']['index']==([0,2] if i%4==0 else [0,0,i%4-1]),'actual checkpoint prediction index')
        if z is not None:
            vector(z);need(row['status']=='encoded' and s['pixel_source'] is not None,'only actual encoded available frame')
            if i%4==0:need(z.tobytes()==pp.tobytes(),'cp0 exact')
        else:need(row['latent'] is None and s['pixel_source'] is None,'complete input missing only at native missing checkpoint')
        frame=s['frame'];native=None
        if frame is not None:
            need(frame['step']==s['cp_public_step'] and equal(frame['state'],frame['snapshot']['state']),'same-time fullstate')
            if i%4:need(s['response']['step']==s['cp_public_step'] and equal(s['response']['state'],frame['state']),'same-time actual return')
            native=native_scalars(frame['state'],s['native_goal'])
        else:need(s['pixel_source'] is None,'no substitute image')
        o=dict(index=i,slot=s['slot'],root=s['root'],arm=s['arm'],cp_public_step=s['cp_public_step'],missing=z is None,
            model_tracking_Q=mse(pp,t),model_task_Q=mse(pp,g),actual_tracking_Q=None if z is None else mse(z,t),
            actual_task_Q=None if z is None else mse(z,g),prediction_vs_encoded_MSE=None if z is None else mse(pp,z),
            model_minus_actual_task_Q=None if z is None else mse(pp,g)-mse(z,g),native=native,
            source=s['pixel_source'],prediction_source=s['prediction_pointer'],response=s['response'],task_id=s['task_id'],
            goal_source=s['goal_source'],raw15_sha256=s['raw15_sha256'],source_time=s['source_time'],target_origin=s['target_origin'],
            physical_benefit_claim=False,P2_mechanism=False)
        out.append(o);new.append(dict(index=i,slot=s['slot'],root=s['root'],arm=s['arm'],cp_public_step=s['cp_public_step'],
            missing=z is None,reason=s['reason'],arithmetic=decompose(pp,z,t,g),native_squared=native_decomposition(native),
            input_sha256=dict(predicted=digest(pp.tobytes()),actual=None if z is None else digest(z.tobytes()),target=digest(t.tobytes()),goal=digest(g.tobytes())),
            source_time=s['source_time'],target_origin=s['target_origin']))
    for slot in range(6):
        base=out[4*slot]
        for o in out[4*slot:4*slot+4]:
            o['model_task_reduction_from_cp0']=base['model_task_Q']-o['model_task_Q']
            o['actual_task_reduction_from_cp0']=sub(base['actual_task_Q'],o['actual_task_Q'])
            o['native_error_gain_from_cp0']=None if o['native'] is None or base['native'] is None else {k:base['native'][k]-o['native'][k] for k in ERRORS}
            o['native_reward_gain_from_cp0']=None if o['native'] is None or base['native'] is None else o['native']['native7_reward']-base['native']['native7_reward']
    for root in (0,1):
        first=p['rows'][12*root]
        for s in p['rows'][12*root:12*root+12]:
            need(f32(s['goal192']).tobytes()==f32(first['goal192']).tobytes() and s['task_id']==first['task_id'] and
                 s['original_z_sha256']==first['original_z_sha256'] and s['original_a_sha256']==first['original_a_sha256'],'same task/goal/fullhistory per root')
    return out,new


def delta_terms(a,b):
    """b minus a; common-goal prediction residual identity, with no null filling."""
    x,y=(v['arithmetic']['residual']['task_goal'] for v in (a,b))
    fields=('model_goal_MSE','actual_goal_MSE','prediction_residual_energy','signed_residual_cross_term')
    terms={k:sub(y[k],x[k]) for k in fields}
    rhs=None if any(terms[k] is None for k in fields[1:]) else math.fsum(terms[k] for k in fields[1:])
    return dict(**terms,reconstructed_model_difference=rhs,identity_roundoff_residual=sub(terms['model_goal_MSE'],rhs))


def pairs(rows,new):
    old=[];out=[]
    for root in (0,1):
        for left,right in itertools.combinations(range(3*root,3*root+3),2):
            for j,cp in enumerate(CP):
                a,b=rows[4*left+j],rows[4*right+j];ok=not a['missing'] and not b['missing']
                nd=None if not ok else {k:float(b['native'][k])-float(a['native'][k]) for k in SCALARS}
                old.append(dict(root=root,left_slot=left,right_slot=right,cp_public_step=cp,comparable=ok,
                    right_minus_left_actual_task_Q=None if not ok else b['actual_task_Q']-a['actual_task_Q'],
                    native_delta=nd,cross_target_tracking_loss_difference=None))
                model=b['model_task_Q']-a['model_task_Q'];actual=sub(b['actual_task_Q'],a['actual_task_Q'])
                native_delta=None if a['native'] is None or b['native'] is None else {k:b['native'][k]-a['native'][k] for k in SCALARS}
                xy=None if native_delta is None else native_delta['joint_xy_error'];n7=None if native_delta is None else native_delta['native7_distance']
                groups=None if native_delta is None else {k:native_delta[k] for k in GROUPS}
                grouped=None if groups is None else math.fsum(groups.values())
                out.append(dict(root=root,left_slot=left,right_slot=right,cp_public_step=cp,direction='right minus left',
                    common_goal=delta_terms(new[4*left+j],new[4*right+j]),native_delta=native_delta,native_squared_groups=groups,
                    native_squared_group_sum=grouped,native_squared_roundoff_residual=None if grouped is None else native_delta['native7_squared']-grouped,
                    breakpoint_axes=dict(A=[model,actual],B=[actual,xy],C=[actual,n7]),
                    breakpoints=dict(A=classify(model,actual),B=classify(actual,xy),C=classify(actual,n7)),
                    cross_target_tracking_loss_difference=None))
    counts={}
    for stage in ('A','B','C'):
        counts[stage]={k:sum(v['breakpoints'][stage]['kind']==k for v in out) for k in ('same_nonzero_sign','strict_reversal','tie','null')}
        counts[stage]['zero_detail']={k:sum(v['breakpoints'][stage]['zero_kind']==k for v in out) for k in ('both_zero','first_zero','second_zero')}
        need(sum(counts[stage][k] for k in ('same_nonzero_sign','strict_reversal','tie','null'))==24,'full per-stage denominator')
    return old,out,counts


def temporal(rows,new):
    cp0=[];adjacent=[]
    for slot in range(6):
        for dest in range(4):
            for start,output in [(0,cp0)]+([] if dest==0 else [(dest-1,adjacent)]):
                a,b=rows[4*slot+start],rows[4*slot+dest]
                nd=None if a['native'] is None or b['native'] is None else {k:b['native'][k]-a['native'][k] for k in SCALARS}
                output.append(dict(slot=slot,root=slot//3,arm=ARMS[slot%3],from_cp=CP[start],to_cp=CP[dest],direction='later minus earlier',
                    common_goal=delta_terms(new[4*slot+start],new[4*slot+dest]),native_delta=nd,
                    tracking={route:None if new[4*slot+start]['arithmetic']['tracking'][route] is None or new[4*slot+dest]['arithmetic']['tracking'][route] is None else
                        {k:new[4*slot+dest]['arithmetic']['tracking'][route][k]-new[4*slot+start]['arithmetic']['tracking'][route][k]
                         for k in new[4*slot+start]['arithmetic']['tracking'][route]} for route in ('model','actual')}))
    return dict(cp0=cp0,adjacent=adjacent)


def analyze(p,r,c,latents):
    original,new=original_rows(p,r,latents)
    need(equal(original,c['rows']),'independent original scalar/identity/gain rows exactly match MAIN receipt')
    old,pair,counters=pairs(original,new)
    need(equal(old,c['pairs']),'independent original same-time pairs exactly match MAIN receipt')
    return dict(denominator=dict(slots=6,positions=24,pairs=24,cp0_differences=24,adjacent_differences=18),
        positions=new,pairs=pair,breakpoint_counts=counters,temporal=temporal(original,new),original_rows_recomputed=original,
        winner=None,scientific_admission=False,known_roots=2,fresh=False,prior_physical_time=None,equal_generation_budget=False)


def documents(p,r,d,c,refs):
    need(p['schema']==RGB+'.prepare' and r['schema']==RGB+'.report' and d['schema']==RGB+'.completion' and c['schema']==RGB+'.consume','exact actual four schemas')
    need(r['completed'] is True and r['errors']==[] and c['status']=='all24_saved_RGB_positions_verified_science_blocked','complete actual RGB, no partial fallback')
    need(r['prepare_sha256']==c['prepare_sha256']==refs['prepare']['sha256'] and Path(r['prepared']).resolve()==Path(refs['prepare']['path']).resolve() and
        c['report_sha256']==d['report_sha256']==refs['report']['sha256'] and c['completion_sha256']==refs['completion']['sha256'],'exact four-way file join')
    need(d['complete'] is True and d['boundary']==p['contract']['wall_boundary']==r['contract']['wall_boundary'] and
        0<=r['wall_before_report']<=d['wall_inclusive']==c['whole_wall_through_report_fsync_hash']<=900,'original inclusive900 completion')
    need(equal(p['contract'],r['contract']) and c['denominator']==dict(slots=6,rows=24,pairs=24) and len(c['pairs'])==24,'original full denominator')
    need(c['scientific_admission'] is False and c['known_roots']==2 and c['fresh'] is False and c['winner'] is None and
        all(c[k]==0 for k in ('new_predict','new_physics','new_AD')),'imported scope unchanged')
    need(equal(p['imported_paid_receipt'],c['inherited_paid_receipt']) and equal(p['imported_controller_receipt'],c['original_controller_receipt']) and
        equal(p['imported_native_receipt']['counts'],c['inherited_native_counts']) and equal(p['bindings'],c['bindings']),'full imported receipts/closure equal, no nested reauthentication')
    need(p['imported_paid_receipt']['report_sha256']=='c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd' and
        p['imported_native_receipt']['physical_report_sha256']=='ddbb339f87e0f49b1d99d32ae4f41340d72e6c19f8199fe69e6d0261f78bca2b' and
        p['imported_controller_receipt']['report_sha256']=='b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934','actual ancestors, not normalized substitution')
    need(r['model_before']==r['model_after']==p['expected_model_hash'] and equal(r['counts'],c['counts']) and equal(r['resources'],c['resources']),'original frozen encoder/resource metadata')


def load_latent(folder,desc,inventory):
    need(set(desc)=={'file','sha256','array_sha256','shape','dtype'} and Path(desc['file']).name==desc['file'] and
        desc['file'].endswith('.npy.gz') and desc['shape']==[192] and desc['dtype']=='float32','exact latent descriptor')
    path=Path(folder)/desc['file'];need(inventory.get(desc['file'])==desc['sha256'] and sha(path,ARRAY)==desc['sha256'],'latent in sealed inventory')
    raw=path.read_bytes();need(digest(raw)==desc['sha256'],'unchanged latent file bytes')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:b=stream.read(ARRAY+1)
    need(len(b)<=ARRAY,'bounded expanded NPY');stream=io.BytesIO(b)
    version=np.lib.format.read_magic(stream)
    need(version in ((1,0),(2,0)),'bounded native NPY header version')
    reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    shape,fortran,dtype=reader(stream,max_header_size=4096)
    need(shape==(192,) and dtype==np.float32 and not fortran,'NPY header checked before allocation')
    stream.seek(0);a=np.load(stream,allow_pickle=False)
    need(stream.tell()==len(b) and a.dtype==np.float32 and a.shape==(192,) and np.isfinite(a).all() and digest(a.tobytes())==desc['array_sha256'],'original NPY EOF/dtype/shape/data SHA')
    return a


def source_join(p,r,check):
    """Local metadata pointers only; never reads PNGs, original checkpoints or full bindings."""
    for i,(s,row) in enumerate(zip(p['rows'],r['rows'],strict=True)):
        check();native=p['imported_native_receipt']['rows'][i//4];plan=native['plan']
        need(s['task_id']==plan['task_id'] and s['task_sha256']==plan['task_sha256'] and s['blind_sha256']==plan['blind_sha256'] and
            s['selection_digest']==plan['selection_digest'] and s['full_raw_bits']==plan['raw_float32_bits'] and s['raw15_sha256']==plan['raw_float32_sha256'],'original task/raw/selection join')
        raw=np.frombuffer(bytes.fromhex(s['full_raw_bits']),dtype='<f4')
        need(raw.shape==(30,) and np.isfinite(raw).all() and (np.abs(raw)<=1).all() and digest(raw.tobytes())==s['raw15_sha256'] and
            raw[:2*s['cp_public_step']].tobytes().hex()==s['raw_prefix_bits'],'exact ordered same-time raw prefix')
        need(f32(plan['commands'],(15,2)).tobytes()==raw.tobytes() and equal(s['target_origin'],plan['target_origin']) and
            s['original_z_sha256']==plan['payload_sha256']['z'] and s['original_a_sha256']==plan['payload_sha256']['a'],'same full causal history/target source')
        need(f32(s['target192']).tobytes()==f32(plan['controller_payload']['target'],(1,192)).tobytes(),'original tracking target bytes')
        need(equal(s['goal_source'],p['goal_sources'][i//12]),'same sealed original task-goal source')
        cp=s['cp_public_step'];found=[f for f in native['checkpoints'] if f['step']==cp]
        if s['frame'] is None:need(found==[] and row['latent'] is None,'real missing cp, no substitute')
        else:
            need(len(found)==1 and equal(found[0],{k:s['frame'][k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}) and
                equal(s['native_goal'],native['goal']),'native checkpoint/state/goal exact')
            src=s['pixel_source'];need(Path(src['path']).name==s['frame']['png'] and
                p['bindings'].get(src['path'])==src['file_sha256']==s['frame']['file_sha256'] and src['pixel_sha256']==s['frame']['pixel_sha256'],'PNG identity already in imported closure')
            if cp:need(equal(s['response'],native['responses'][cp-1]),'same-time native response')


def build(refs=None,clock=time.monotonic):
    start=clock()
    def check():need(clock()-start<=WALL,'local analysis whole120 wall')
    refs=actual_refs() if refs is None else refs;need(refs==actual_refs(),'fixed actual four file authorities')
    docs={}
    for k in ('prepare','report','completion','receipt'):
        check();docs[k]=read(refs[k]['path'],refs[k]['sha256'])
    p,r,d,c=(docs[k] for k in ('prepare','report','completion','receipt'));documents(p,r,d,c,refs)
    need(len(p['rows'])==len(r['rows'])==24,'complete24 roster')
    folder=Path(refs['report']['path']).parent;sealpath=folder/'native_seal.json';seal=read(sealpath,r['native_seal_sha256'])
    need(r['artifact_sha256'].get('native_seal.json')==r['native_seal_sha256'] and equal(seal,{k:r[k] for k in ('rows','modules','counts','errors')}),'actual native seal/reference exact')
    sources={v['path']:v['sha256'] for v in refs.values()};sources[str(sealpath)]=r['native_seal_sha256'];sources.update(closure())
    source_join(p,r,check);latents=[]
    for row in r['rows']:
        check();desc=row['latent'];latents.append(None if desc is None else load_latent(folder,desc,r['artifact_sha256']))
        if desc is not None:sources[str(folder/desc['file'])]=desc['sha256']
    result=analyze(p,r,c,latents)
    for path,h in sources.items():check();need(sha(path)==h,'local input/source bytes unchanged')
    check()
    return dict(schema=NAME+'.analysis',authorities=refs,local_sources=sources,**result,
        upstream_authentication_imported=True,independent_fullstrict=False,independent_gradient=False,
        inherited_resources=dict(RGB=c['resources'],paid=c['inherited_paid_receipt'],controller=c['original_controller_receipt'],native=c['inherited_native_counts']),
        new_calls=dict(encoder=0,predict=0,AD=0,physics=0),FLOPs=None,
        arithmetic='f32 inputs, f64 arithmetic, fsum/192; signed cross terms and roundoff retained',
        elapsed_not_embedded_for_reproducible_verify=True)


def write(path,v):
    p=Path(path).absolute();b=encode(v)
    need(p.parent.resolve()==ROOT/'results' and p.name.startswith(NAME+'_') and not p.is_symlink() and len(b)<=CAP and
        shutil.disk_usage(p.parent).free>=FLOOR+len(b),'own exclusive16MiB output/floor200MiB')
    with p.open('xb') as f:need(f.write(b)==len(b),'complete metadata write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=sha(p))


def verify(path,h):
    v=read(path,h);need(equal(v,build(v['authorities'])),'entire local arithmetic analysis reconstructed');return dict(verified=True,independent_fullstrict=False,new_neural=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('build','verify'))
    defaults=actual_refs()
    for k,v in defaults.items():
        parser.add_argument('--'+k,default=v['path']);parser.add_argument('--'+k+'-sha256',default=v['sha256'])
    parser.add_argument('--output',type=Path);parser.add_argument('--summary',type=Path);parser.add_argument('--summary-sha256');a=parser.parse_args(argv)
    start=time.monotonic()
    if a.command=='build':
        refs={k:dict(path=str(Path(getattr(a,k)).resolve()),sha256=getattr(a,k+'_sha256')) for k in defaults}
        out=write(a.output,build(refs))
    else:out=verify(a.summary,a.summary_sha256)
    need(time.monotonic()-start<=WALL,'inclusive local read/arithmetic/write120 wall; retain output on timeout')
    print(encode(out).decode())


if __name__=='__main__':main()
