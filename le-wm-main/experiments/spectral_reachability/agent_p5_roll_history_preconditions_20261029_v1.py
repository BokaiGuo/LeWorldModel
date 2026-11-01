"""Metadata-only history identifiability inventory; no NN/physics/full strict.

This is a blocked precondition report, NOT an execution prepare or loaded-byte
certification. Pure ABI helpers do not acquire missing history or run rollouts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

NAME='agent_p5_roll_history_preconditions_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CAP,FLOOR=32*1024**2,200*1024**2
PINS={'H1':('agent_p5_raw164_jepa_cuda_20261029_v1','222d917f262e2a7c8e347fd351e3b0927ec4ffe200035e7b3b971a7abf0000fe',
    '610523392e1923e4371556606109def7f882a4a90741e2e14b63af68880e2180'),
 'H2':('agent_p5_raw164_jepa_h2_cuda_20261029_v2','09070780c3ce782a71ae01492e0c4b801de976e9a36cdf9ee1bf1a53be98e924',
    'b661a7c27181ef29f8277e154006f9238ec1c94487c311c1065e4d42fcfc2241')}
SHAPES=((1,3,1,16,16,384),(1,3,10),(1,3,256,16))
NATIVE_SOURCES={
 'external/jepa-wms/app/plan_common/models/AdaLN_vit.py':'77e20661849a33199c581b4ab47e2d0417ef2e81571e4998a30cc242b68abe97',
 'external/jepa-wms/src/models/utils/modules.py':'aabc4baa04036c05142a4e314d1712141a38e31611121d56d21674750f303fe8',
 'experiments/spectral_reachability/p5_jepa_native_action_response_20261027_v8.py':'fc915ddee2e53ed58366994684fe2190da5eaaeb8323ad608ffe3292534c936a'}


def require(test,message):
    if not test:raise ValueError(message)


def raw(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def pairs(items):
    d={}
    for k,v in items:require(k not in d,'duplicate JSON field');d[k]=v
    return d


def read(path,expected):
    p=Path(path).absolute()
    require(type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'explicit external SHA256')
    require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)) and p.stat().st_size<=CAP and sha(p)==expected,'bounded nonsymlink metadata exact SHA')
    v=json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    raw(v);require(sha(p)==expected,'metadata stable');return v


def f32(value,shape):
    a=np.asarray(value,dtype=np.float32)
    require(a.shape==shape and np.isfinite(a).all(),'finite exact float32 shape');return a


def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes(order='C')==b.tobytes(order='C')


def normalized(raw_proprio,commands,stats):
    """Same original float32 pixel32 round trip, no changed normalizer."""
    p=np.asarray(raw_proprio,dtype=np.float32);u=np.asarray(commands,dtype=np.float64)
    require(p.ndim==2 and p.shape[0] in (1,2,3) and p.shape[1]==4 and np.isfinite(p).all(),'one/two/three actual proprio slots, no replication')
    require(u.ndim==2 and u.shape[1]==2 and u.shape[0] in (15,20,30) and np.isfinite(u).all(),'15/20/30 literal raw actions, no padding')
    require(set(stats)=={'ACTION_MEAN','ACTION_STD','PROPRIO_MEAN','PROPRIO_STD'},'original four stats only')
    am,sd=f32(stats['ACTION_MEAN'],(2,)),f32(stats['ACTION_STD'],(2,))
    pm,ps=f32(stats['PROPRIO_MEAN'],(4,)),f32(stats['PROPRIO_STD'],(4,))
    require(np.all(sd>0) and np.all(ps>0),'original strictly positive std')
    dataset=(100.*u).astype(np.float32)/np.float32(100.)
    return (p-pm)/ps,((dataset-am)/sd).reshape(-1,10)


def full3_abi():
    calls=[]
    for i,target in enumerate((5,10,15,20)):
        frames=[target-15,target-10,target-5]
        calls.append(dict(index=i,predicts_global_frame=target,context_global_frames=frames,
            context_origin=['actual' if x<=0 else 'sealed_prediction' for x in frames],
            action_raw_inclusive=[[x+1,x+5] for x in frames],packet_indices=[i,i+1,i+2],
            predicted_output='native_visual_and_proprio_full3:last_time_slot_only_for_rollover'))
    return dict(schema=NAME+'.roll_abi',packet_raw_steps=5,start_actual_frames=[-10,-5,0],
        required_raw_action_range_inclusive=[-9,20],calls=calls,
        initial_visual_shape=list(SHAPES[0]),initial_proprio_shape=list(SHAPES[2]),action_call_shape=list(SHAPES[1]),
        final_context_global_frames=[5,10,15],final_context_origin=['sealed_prediction']*3,
        no_teacher_feedback_after_frame0=True,padding_allowed=False,shortened_context_allowed=False,
        original_weights_or_normalizer_change=False,goal_overlay='preserve original native warmup/plan binding schedule; no redraw or assumed continuity',
        extra_actual_agent_position_velocity=True,not_RGB_only=True)


def abi():
    calls=[]
    for i,frames in enumerate(([0],[0,5],[0,5,10],[5,10,15])):
        calls.append(dict(index=i,predicts_global_frame=(i+1)*5,context_global_frames=frames,
            context_origin=['actual' if x==0 else 'sealed_prediction' for x in frames],
            action_raw_inclusive=[[x+1,x+5] for x in frames],packet_indices=[x//5 for x in frames],
            input_visual_shape=[1,len(frames),1,16,16,384],input_proprio_shape=[1,len(frames),256,16],
            input_action_shape=[1,len(frames),10],output_visual_shape=[1,len(frames),256,384],
            predicted_output='native visual AND proprio last token; earlier full3 teacher-conditioned slots cannot substitute'))
    return dict(schema=NAME+'.short_start_roll_abi',packet_raw_steps=5,start_actual_frames=[0],
        required_raw_action_range_inclusive=[1,20],calls=calls,context_lengths=[1,2,3,3],
        native_forward_source_structurally_permits_dynamic_T=True,HOSTCUDA_T1_T2_execution_validated=False,
        requires_NEW_explicit_startup_protocol=True,original_full3_contract_unchanged=False,
        final_context_global_frames=[5,10,15],final_context_origin=['sealed_prediction']*3,
        padding_allowed=False,shorter_context_is_explicit_not_silent=True,no_teacher_feedback_after_frame0=True,
        weights_normalizer_unchanged=True,extra_actual_agent_position_velocity=True,not_RGB_only=True,
        goal_overlay='same original warmup/plan schedule and condition; no redraw; source frame0 before goal bind',
        production_ready=False)


def pair_schedule():
    return [dict(slot=s,root_slot=r,condition=c,h=h,coord=k,minus_index=41*s+1+20*j+2*k,plus_index=41*s+2+20*j+2*k)
        for s,(r,c) in enumerate(((0,'off'),(0,'on'),(1,'off'),(1,'on'))) for j,h in enumerate((.01,.001)) for k in range(10)]


def missing_history(available_actual_frames,available_raw_action_steps,available_predicted_frames):
    require(all(type(x) is int for xs in (available_actual_frames,available_raw_action_steps,available_predicted_frames) for x in xs),
        'literal integer time coordinates, not bool')
    actual=set(available_actual_frames);actions=set(available_raw_action_steps);pred=set(available_predicted_frames)
    frames=[x for x in (-10,-5,0) if x not in actual];steps=[x for x in range(-9,21) if x not in actions]
    return dict(full3_missing_actual_start_frames=frames,full3_missing_raw_action_steps=steps,missing_saved_rolled_predicted_frames=[x for x in (5,10) if x not in pred],
        full3_bootstrap_constructible=not frames and not steps,existing_early_prediction_assets_complete={5,10}<=pred,
        short_start_missing_actual_frames=[0] if 0 not in actual else [],
        short_start_missing_raw_action_steps=[x for x in range(1,21) if x not in actions],
        short_start_pending=['explicit MAIN protocol approval','native T1/T2 HOSTCUDA shape validation','original actual0 native feature IO external binding','new sealed causal pred5 and pred10'],
        sufficient_for_new_contrast=False,reason='even valid bootstrap does not certify distinct realized native inputs; byte preflight remains mandatory')


def bootstrap_packets(start_frames,raw_action_steps,start_proprio,commands,stats):
    """Pure ABI normalization only. Actual RGB/proprio/trajectory authority required separately."""
    require(all(type(x) is int for x in start_frames+raw_action_steps) and start_frames==[0] and raw_action_steps==list(range(1,21)),
        'explicit short-start actual0 and all20 ordered literal actions required; no padding')
    prop,packets=normalized(start_proprio,commands,stats)
    require(prop.shape==(1,4) and packets.shape==(4,10),'one actual0 proprio and four complete five-raw packets')
    return dict(normalized_start_proprio=prop,packet_bank=packets,
        call_packets=[packets[indices].copy() for indices in ([0],[0,1],[0,1,2],[1,2,3])])


def advance_history(frames,visual,proprio,prediction):
    """Pure no-arithmetic native rollover. No prediction is generated here."""
    require(all(type(x) is int for x in frames) and frames in ([0],[0,5],[0,5,10],[5,10,15]),'only explicit no-padding short-start rollout states')
    t=len(frames)
    require(isinstance(visual,np.ndarray) and visual.shape==(1,t,1,16,16,384) and visual.dtype==np.float32 and np.isfinite(visual).all() and
        isinstance(proprio,np.ndarray) and proprio.shape==(1,t,256,16) and proprio.dtype==np.float32 and np.isfinite(proprio).all(),
        'complete existing native history, no fabricated missing slots')
    require(type(prediction) is tuple and len(prediction)==3 and prediction[1] is None and
        all(isinstance(a,np.ndarray) and a.dtype==np.float32 and a.shape==s and np.isfinite(a).all()
            for a,s in ((prediction[0],(1,t,256,384)),(prediction[2],(1,t,256,16)))),
        'full visual AND predicted proprio native return, no observed successor feedback')
    v=visual.reshape(1,t,256,384);new_frames=(frames+[frames[-1]+5])[-3:]
    return dict(frames=new_frames,
        visual=np.concatenate([v,prediction[0][:,-1:]],axis=1)[:,-3:].reshape(1,len(new_frames),1,16,16,384),
        proprio=np.concatenate([proprio,prediction[2][:,-1:]],axis=1)[:,-3:])


def contrast_preflight(reference,candidate,reference_identity,candidate_identity):
    """Pure fixture/future runtime guard; NOT a production receipt or new experiment."""
    keys={'branch','root_slot','condition','task_id','selection_digest','aligned_frames','model_state_sha256',
        'normalizer_source_sha256','common_label_identity','goal_overlay_history_identity'}
    require(set(reference_identity)==set(candidate_identity)==keys and reference_identity==candidate_identity and
        type(reference_identity['branch']) is int and type(reference_identity['root_slot']) is int and
        reference_identity['aligned_frames']==[5,10,15],'exact branch/model/time/normalizer/label/overlay identity')
    require(len(reference)==len(candidate)==3,'three native inputs')
    for xs in (reference,candidate):
        require(all(isinstance(a,np.ndarray) and a.dtype==np.float32 and a.shape==s and np.isfinite(a).all()
            for a,s in zip(xs,SHAPES,strict=True)),'full native float32 IO, no downcast or partial tokens')
    require(exact(reference[1],candidate[1]),'all three action packets unchanged')
    changed={k:[not exact(a[:,i:i+1],b[:,i:i+1]) for i in range(3)] for k,a,b in
        (('visual',reference[0],candidate[0]),('proprio',reference[2],candidate[2]))}
    require(any(changed[k][i] for k in changed for i in (0,1)),
        'not a distinct EARLY-history contrast: first2 slots identical; latest-only/repeat cannot be renamed')
    # Signed-zero-only differences are not a numerically distinct history contrast.
    require(any(np.any(reference[j][:,:2]!=candidate[j][:,:2]) for j in (0,2)),
        'signed-zero byte difference alone is not numeric early-history variation')
    return dict(schema=NAME+'.distinct_input_check',changed_by_slot=changed,distinct_early_input=True,
        prediction_or_physical_validity_certified=False,P1_admission=False,P5_admission=False)


def validate_manifests(h1,h2):
    for key,m in (('H1',h1),('H2',h2)):
        name,_,source_sha=PINS[key]
        require(m['schema']==name+'.prepare' and len(m['bridge']['requests'])==164,'fixed complete H1/H2 prepare schema')
        path=str(ROOT/'experiments/spectral_reachability'/(name+'.py'))
        require(m['bindings'].get(path)==source_sha,'frozen original runtime source binding')
    require(h1['bridge_prepared']==h2['bridge_prepared'] and h1['bridge_prepare_sha256']==h2['bridge_prepare_sha256'] and
        h1['bridge']['stats']==h2['bridge']['stats'],'same original bridge and normalizer')
    require(h1['protocol']['pairs']==h2['protocol']['pairs']==pair_schedule(),'all80 fixed original pair identities, no omission/selection')
    rows=[]
    for i,(a,b) in enumerate(zip(h1['bridge']['requests'],h2['bridge']['requests'],strict=True)):
        require(all(type(a[k]) is type(b[k]) and a[k]==b[k] for k in ('index','root_slot','condition','task_id','selection_digest')) and
            type(a['index']) is int and a['index']==i and type(a['root_slot']) is int and
            (a['root_slot'],a['condition'])==((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41], 'fixed full164 branch/root/condition/task/selection')
        require(a['context_global_frames']==[5,10,15] and a['successor_global_frame']==20 and
            b['context_global_frames']==[0,5,10] and b['successor_global_frames']==[15,20] and
            b['history_semantics']=='original pre-constructed-goal-bind warmup0/5/10; native default goal_pose overlay, never redrawn',
            'actual H1/H2 time and overlay chronology, no pred5/10 relabel')
        require(len(a['images'])==len(b['images'])==3 and len(b['labels'])==2 and a['label']==b['labels'][1],
            'same factual terminal label pointer, no label decoding')
        require((a['label']['phase'],a['label']['step'])==('plan',10) and
            [(r['phase'],r['step']) for r in b['labels']]==[('plan',5),('plan',10)],'literal15/20 label phases')
        require([(r['phase'],r['step']) for r in a['images']]==[('warmup',5),('plan',0),('plan',5)] and
            [(r['phase'],r['step']) for r in b['images']]==[('warmup',0),('warmup',5),('warmup',10)],'true physical image times')
        for r in a['images']+b['images']+[a['label']]+b['labels']:
            require(r['condition']==a['condition'] and r['overlay_index']==int(a['condition']=='on') and
                type(r['pixel_sha256']) is str and len(r['pixel_sha256'])==64 and type(r['file_sha256']) is str and len(r['file_sha256'])==64,
                'same-condition original image authority')
        u1=np.asarray(a['commands'],np.float64);u2=np.asarray(b['commands'],np.float64)
        require(u1.shape==(15,2) and u2.shape==(20,2) and np.isfinite(u1).all() and np.isfinite(u2).all() and
            exact(u1,u2[5:]),'exact raw6..20 suffix, no inferred missing negative actions')
        pa,ua=normalized(a['raw_proprio'],u1,h1['bridge']['stats']);pb,ub=normalized(b['raw_proprio'],u2,h2['bridge']['stats'])
        require(exact(pa,f32(a['normalized_proprio'],(3,4))) and exact(pb,f32(b['normalized_proprio'],(3,4))) and
            exact(ua,f32(a['packets'],(3,10))) and exact(ub,f32(b['packets'],(4,10))) and exact(ua,ub[1:]),
            'original source normalization and all three outgoing action packets exact')
        require(exact(f32(a['raw_proprio'],(3,4))[:2],f32(b['raw_proprio'],(3,4))[1:]) and
            all(a['images'][j]['pixel_sha256']==b['images'][j+1]['pixel_sha256'] for j in (0,1)),
            'shared actual5/10 observations, no predicted early history')
        rows.append(dict(index=i,root_slot=a['root_slot'],condition=a['condition'],task_id=a['task_id'],selection_digest=a['selection_digest'],
            actual_image_assets=[dict(global_frame=t,record=r) for t,r in zip((0,5,10,15),[b['images'][0],a['images'][0],a['images'][1],a['images'][2]],strict=True)],
            source_proprio_frames=[0,5,10,15],raw_action_range_inclusive=[1,20],available_rolled_prediction_semantics=[15,20],
            existing_auxiliary_full3_output_slots=dict(successor_frames=[5,10,15],
                caution='H2 first-call full3 contains numeric early output slots; pred10 uses ACTUAL5 input, not pred5. Not a sealed causal pred5->pred10 rollout.'),
            prediction_semantics_scope='declared H2 protocol only; no output artifacts or full strict consumed here',
            missing=missing_history([0,5,10,15],list(range(1,21)),[15,20])))
    return rows


def inspect(h1_manifest,h1_sha,h2_manifest,h2_sha):
    require(h1_sha==PINS['H1'][1] and h2_sha==PINS['H2'][1],'exact sealed H1/H2 manifests, no replacement')
    h1=read(h1_manifest,h1_sha);h2=read(h2_manifest,h2_sha);rows=validate_manifests(h1,h2)
    used={str(Path(h1_manifest).absolute()):h1_sha,str(Path(h2_manifest).absolute()):h2_sha}
    for key in ('H1','H2'):
        name,_,h=PINS[key];p=ROOT/'experiments/spectral_reachability'/(name+'.py')
        require(sha(p)==h,'actual sealed runtime source stable');used[str(p)]=h
    for name,h in NATIVE_SOURCES.items():
        p=ROOT/name;require(sha(p)==h and h1['bindings'].get(str(p))==h2['bindings'].get(str(p))==h,'original dynamic-T/mask/loader source bound by BOTH manifests')
        used[str(p)]=h
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):used[str(p)]=sha(p)
    return dict(schema=NAME+'.inventory',status='blocked_missing_causal_pred5_pred10_and_short_start_validation',
        arguments=dict(h1_manifest=str(Path(h1_manifest).absolute()),h1_sha=h1_sha,h2_manifest=str(Path(h2_manifest).absolute()),h2_sha=h2_sha),
        branches=rows,branch_count=164,pair_count=80,pairs=pair_schedule(),roll_abi=abi(),full3_only_alternative=full3_abi(),bindings=used,
        global_missing=dict(sealed_causal_rolled_visual_and_proprio_frames=[5,10],
            short_start_missing_RGB_or_raw_actions=[],short_start_pending=['MAIN approval','T1/T2 native shape validation','external actual0 native feature IO binding'],
            full3_only_missing_actual_RGB_and_proprio_frames=[-10,-5],full3_only_missing_raw_actions_inclusive=[-9,0],
            full3_only_continuity='no authenticated predecessor ending at the exact old frame0 reset; do not prepend new simulation'),
        acquisition_cost=dict(short_start_new_physics=0,short_start_new_encoders_if_original_actual0_IO_verified=0,
            existing_actual0_IO_reuse_NOT_certified_here=True,
            full3_only_needed_new_raw_controls_per_branch_lower_bound=10,full3_only_native_substeps_per_branch_lower_bound_at_original10=100,
            full3_only_constructor_reset_cost='unknown; cannot guarantee an exact old-root predecessor',
            future_full_roll_predictions_per_branch=4,future_all164_predict_calls=656,
            future_all164_token_frame_samples=1476,
            counts_are_design_not_authorization_or_actual_receipts=True),
        entire_panel_execution_ready=False,distinct_early_contrast_certified=False,full_strict_consumed=False,
        PNG_or_latent_decoded=False,loaded_weight_bytes_certified=False,production_prepare=False,
        new_neural_calls=0,new_physics_calls=0,new_training=0,P1_admission=False,P5_admission=False,
        limitations=['Only explicit original manifest fields and bound runtime source hashes inspected; not all historical assets scanned.',
            'Declared prediction times are not proof of completed predicted artifacts; no outputs or outcome scores read.',
            'T1/T2 startup is an explicit NEW protocol, source structurally supported but not run/validated here; never pad absent frames.',
            'New predecessor acquisition is not guaranteed to recover this exact old root; changed roots require a new prospective panel.',
            'Extra native agent position/velocity, not RGB-only or matched cross-model control benefit.'])


def write_new(value,path):
    p=Path(path).absolute();body=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and
        not any(x.is_symlink() for x in (p,*p.parents)) and len(body)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(body),
        'own exclusive new metadata only, bounded/no overwrite')
    with p.open('xb') as f:require(f.write(body)==len(body),'complete write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(status=value['status'],path=str(p),sha256=sha(p))


def verify(path,expected):
    value=read(path,expected)
    require(value['schema']==NAME+'.inventory' and value==inspect(**value['arguments']),'metadata inventory deterministically rebuilt; no full strict')
    return dict(status='blocked_history_inventory_recomputed',sha256=expected,production_ready=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('inspect')
    for k in ('h1-manifest','h1-sha','h2-manifest','h2-sha','output'):a.add_argument('--'+k,required=True)
    a=s.add_parser('verify');a.add_argument('--inventory',required=True);a.add_argument('--inventory-sha',required=True)
    q=vars(p.parse_args());command=q.pop('command')
    if command=='inspect':output=q.pop('output');r=write_new(inspect(**q),output)
    else:r=verify(q['inventory'],q['inventory_sha'])
    print(json.dumps(r,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
