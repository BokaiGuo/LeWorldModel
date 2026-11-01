"""Saved overlay8 physical comparison + own RGB consume -> separate task errors.

No model/physics/prepare/selection. No equivalence of latent and task improvement.
External existing report SHAs are mandatory; missing checkpoints remain null.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_moving_overlay_recompile_comparison_20261027_v1 as physical

NAME='moving_overlay_task_metric_bridge_20261027_v1'
COMPARISON=physical.NAME
RGB='p1_moving_overlay_recompile_rgb_cuda_20261027_v1'
CELLS=physical.CELLS
COMPARISON_FILE=Path(physical.__file__).resolve()
RGB_FILE=COMPARISON_FILE.with_name(RGB+'.py')
RGB_SOURCE_SHA='70c03508d775b8b488c12133da929bae607d08d6163d5e1c6578edd9a2422bcc'
CAP,FLOOR=2*1024**2,200*1024**2
STEPS=(0,5,10)
SHAPES=dict(pixels=[1,3,224,224],hidden=[1,257,192],cls=[1,192],projected=[1,192])
METRIC_NAMES=('actual_latent_Q','agent_position_l2','object_position_l2','all_position_l2',
              'absolute_wrapped_angle','velocity_l2','native_distance','wrapped_joint_state_l2')
require=physical.require


def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def hash_value(value):return physical.digest(value)


def path_value(value):
    require(type(value) is str and Path(value).is_absolute() and str(Path(value))==value and '..' not in Path(value).parts,'literal absolute path')
    return value


def parse(raw):
    value=physical.parse(raw);canonical(value);return value


def vector(value):
    require(type(value) is list and len(value)==7 and all(type(x) in (int,float) for x in value),'literal state7, not bool')
    return physical.vector(value)


def index_rows(rows):
    return physical.index_rows(rows)


def index_steps(cps):
    require(type(cps) is list,'explicit checkpoints')
    result={}
    for cp in cps:
        t=cp['step'];require(type(t) is int and t in STEPS and t not in result,'unique literal checkpoint step')
        result[t]=cp
    return result


def decompose(goal,state):
    """Native coordinate/velocity units stay separate; never add latent Q."""
    d=vector(state)-vector(goal)
    def block(delta):return dict(delta=delta.tolist(),squared_l2=float(delta@delta),l2=float(np.linalg.norm(delta)))
    angle=float(d[4]);wrapped=float(np.arctan2(np.sin(angle),np.cos(angle)))
    native=physical.metrics(goal,state)
    result=dict(agent_position=block(d[:2]),object_position=block(d[2:4]),all_position=block(d[:4]),
        velocity=block(d[5:]),angle=dict(raw_delta=angle,absolute_raw=abs(angle),wrapped_delta=wrapped,
            absolute_wrapped=abs(wrapped),squared_raw=angle**2,squared_wrapped=wrapped**2,
            native_term=native['native_angle_term']),raw_squared_total=float(d@d),
        wrapped_squared_total=float(d[:4]@d[:4]+wrapped**2+d[5:]@d[5:]),native=native)
    canonical(result);return result


def metric_values(cp):
    e=cp['errors']
    return dict(actual_latent_Q=cp['actual_latent_Q'],**({k:None for k in METRIC_NAMES[1:]} if e is None else dict(
        agent_position_l2=e['agent_position']['l2'],object_position_l2=e['object_position']['l2'],
        all_position_l2=e['all_position']['l2'],absolute_wrapped_angle=e['angle']['absolute_wrapped'],
        velocity_l2=e['velocity']['l2'],native_distance=e['native']['native_distance'],
        wrapped_joint_state_l2=e['native']['wrapped_joint_state_l2'])))


def difference(after,before):
    return {k:None if after[k] is None or before[k] is None else after[k]-before[k] for k in METRIC_NAMES}


def assemble(base,report,receipt):
    """RAM arithmetic join. material() supplies externally hashed own-consume output."""
    canonical(base);canonical(report);canonical(receipt)
    require(base['schema']==COMPARISON+'.report' and base['status']=='saved_physical_comparison_not_admitted'
        and type(base['branch_denominator']) is int and base['branch_denominator']==8 and base['root_denominator']==2
        and all(base[k] is False for k in ('P1_admission','model_repaired','independent_confirmation','equal_total_FLOPs')),
        'fixed-eight nonadmitted physical comparison')
    require(report['schema']==RGB+'.report' and report['status']=='encoded_sealed' and
        all(report[k] is True for k in ('cuda_verified','source_verified','model_unchanged')) and
        report['errors']==[] and report['P1_admission'] is False and report['training'] is False and
        report['prediction_status']=='not_recomputed_encoding_only' and report['shapes']==SHAPES,
        'new own-consumed frozen257 RGB output, no prediction or scientific admission')
    ref=base['binding']['files']['physical_report'];hash_value(ref['sha256']);path_value(ref['file'])
    require(report['input_binding']==dict(physical_report=ref['file'],physical_sha=ref['sha256']),
        'same actual physical report absolute path AND SHA, not filename inference')
    originals=index_rows(base['rows']);rgbrows=index_rows(report['rows']);consumed=index_rows(receipt)
    images=report['images'];require(type(images) is list and 4<=len(images)<=28,'four goals plus at most24 checkpoints')
    for i,image in enumerate(images):
        require(type(image['index']) is int and image['index']==i,'image index identity')
    for i,(s,c) in enumerate((s,c) for s in (0,1) for c in ('off','on')):
        g=images[i];require((g['kind'],g['root_slot'],g['condition'])==('goal',s,c),'four own true condition goals')
        path_value(g['path']);hash_value(g['file_sha256']);hash_value(g['pixel_sha256'])
    rows=[];used=[];slot_identity={}
    for row_index,k in enumerate(CELLS):
        s,c,arm=k;b=originals[k];r=rgbrows[k];o=consumed[k]
        require(b['selection_digest']==r['selection_digest']==o['selection_digest'],'actual compiler selection digest')
        require(all(o[f]==r[f] for f in ('root_slot','condition','arm','selection_digest','checkpoints')),'ownconsume exact raw report row')
        goal=vector(b['goal']);hash_value(b['root_sha256'])
        identity=(b['root_sha256'],goal.tolist())
        require(s not in slot_identity or slot_identity[s]==identity,'same original root/true goal across condition/arms')
        slot_identity[s]=identity
        goal_index=s*2+(c=='on');require(type(r['goal_image']) is int and r['goal_image']==goal_index,'same condition true goal latent')
        cps=index_steps(o['checkpoints']);refs=index_steps(b['checkpoint_refs'])
        require(set(cps)==set(STEPS),'three explicit checkpoint slots, missing remains null')
        missing=b['missing_checkpoints']
        require(type(missing) is list and all(type(t) is int for t in missing) and
            len(set(missing))==len(missing) and set(missing)==set(STEPS)-set(refs),'physical missing labels exactly match available frames')
        out=dict(root_slot=s,condition=c,arm=arm,selection_digest=b['selection_digest'],root_sha256=b['root_sha256'],
            goal=b['goal'],goal_rgb=dict(path=images[goal_index]['path'],file_sha256=images[goal_index]['file_sha256'],
                pixel_sha256=images[goal_index]['pixel_sha256'],latent=deepcopy(images[goal_index]['latent'])),
            source_status=b['source_status'],checkpoints=[])
        for t,name in zip(STEPS,('initial','prefix5','terminal10'),strict=True):
            cp=cps[t];p=refs.get(t)
            metric=b[name]
            value=dict(step=t,status=cp['status'],actual_latent_Q=cp['Q'],latent=deepcopy(cp['latent']),
                       errors=None,source_checkpoint=None,predicted_Q=None,prediction_error=None)
            if p is None:
                require(cp['image'] is None and cp['latent'] is None and cp['Q'] is None and metric is None
                    and cp['status']=='missing_physical_checkpoint','missing is unknown, no zero or prefix-as-terminal')
            else:
                require(type(cp['image']) is int and 4<=cp['image']<len(images),'actual checkpoint image index')
                image=images[cp['image']];raw=image['physical']
                require(image['kind']=='checkpoint' and image['row']==row_index and
                    (image['root_slot'],image['condition'],image['step'])==(s,c,t),'same arm/condition/step real image')
                require(image['status']==cp['status']=='encoded' and image['latent']==cp['latent'] and
                    type(cp['Q']) is float and np.isfinite(cp['Q']) and cp['Q']>=0,'own native latent and finite nonnegative real Q')
                require(type(p['png']) is str and Path(p['png']).name==p['png'] and p['png'] not in ('','.','..'),'flat explicitly bound PNG')
                require(image['path']==str(Path(ref['file']).parent/p['png']) and
                    all(raw[f]==v for f,v in p.items()) and all(image[f]==p[f] for f in ('file_sha256','pixel_sha256')),
                    'comparison vs encoded PNG path/fileSHA/pixelSHA/step binding')
                for f in ('file_sha256','pixel_sha256'):hash_value(p[f])
                require(raw['snapshot']['state']==raw['state'],'original state/snapshot equality')
                state=vector(raw['state']);errors=decompose(goal.tolist(),state.tolist())
                require(metric==errors['native'],'original comparison metrics exactly recomputed from actual state/goal')
                if t==0:
                    require(digest(np.concatenate([state,goal]).tobytes())==b['root_sha256'],'original initial7/true goal7 root identity')
                value.update(errors=errors,source_checkpoint=deepcopy(raw));used.append(cp['image'])
            out['checkpoints'].append(value)
        m=[metric_values(cp) for cp in out['checkpoints']]
        out['changes']=dict(prefix5_minus_initial0=difference(m[1],m[0]),terminal10_minus_initial0=difference(m[2],m[0]),
                            terminal10_minus_prefix5=difference(m[2],m[1]))
        rows.append(out)
    require(sorted(used)==list(range(4,len(images))),'every declared checkpoint used exactly once; no hidden dropping/dedup')
    by=index_rows(rows);paired=[];arm_contrasts=[]
    for s in (0,1):
        for arm in physical.ARMS:
            off,on=by[s,'off',arm],by[s,'on',arm]
            paired.append(dict(root_slot=s,arm=arm,checkpoints=[dict(step=t,
                delta=difference(metric_values(on['checkpoints'][i]),metric_values(off['checkpoints'][i]))) for i,t in enumerate(STEPS)],
                semantics='on FULL reoptimization minus off FULL reoptimization, own condition targets; descriptive, not causal'))
        for c in ('off','on'):
            gn,d2p=by[s,c,physical.ARMS[0]],by[s,c,physical.ARMS[1]]
            arm_contrasts.append(dict(root_slot=s,condition=c,checkpoints=[dict(step=t,
                delta=difference(metric_values(d2p['checkpoints'][i]),metric_values(gn['checkpoints'][i]))) for i,t in enumerate(STEPS)],
                semantics='D2P minus GN within same condition/goal; descriptive, not equal total FLOPs'))
    return dict(schema=NAME+'.report',status='saved_task_metric_bridge_not_admitted',branch_denominator=8,root_denominator=2,
        rows=rows,paired_on_minus_off=paired,paired_D2P_minus_GN=arm_contrasts,physical_comparison=deepcopy(base),
        physical_counts=deepcopy(base['physical_counts']),planner_cost_receipt=deepcopy(base.get('planner_cost_receipt')),
        prediction_status='unknown_not_recomputed',latent_improvement_implies_task_improvement=False,
        P1_admission=False,model_repaired=False,independent_confirmation=False,equal_total_FLOPs=False,
        new_neural_calls=0,new_physics_calls=0,new_selection=False,training=False,
        global_freshness='unknown',physical_root_equivalence='not_established_by_this_bridge',
        limitations=['Two reused known roots; no inferential claim or new independent roots.',
            'Position, angle, velocity and latent Q have distinct units; no fitted/weighted combined score.',
            'Off/on latent Q uses separately rendered true condition goals, not a common observation target.',
            'Native success omits velocity and its original angle term can be negative; both facts retained.',
            'No prediction residual or raw-zero control inferred; no existing blocker is cleared.'])


def load_local(name):
    path_value(name);p=Path(name)
    require(p.is_file() and p.resolve()==p and not p.is_symlink() and p.stat().st_size<=CAP,'bounded regular nonsymlink metadata file')
    raw=p.read_bytes();require(len(raw)<=CAP,'bounded metadata read');return raw


def material(comparison,comparison_sha,rgb_output,rgb_report_sha,*,read=load_local,consumer=None):
    hash_value(comparison_sha);hash_value(rgb_report_sha);path_value(comparison);path_value(rgb_output)
    rp=str(Path(rgb_output)/'report.json');base_raw=read(comparison);rgb_raw=read(rp)
    require(len(base_raw)<=CAP and digest(base_raw)==comparison_sha,'external existing comparison SHA')
    require(len(rgb_raw)<=CAP and digest(rgb_raw)==rgb_report_sha,'external existing RGB report SHA before consume')
    base=parse(base_raw);report=parse(rgb_raw)
    require(digest(read(str(COMPARISON_FILE)))==base['binding']['source_sha256'],'same comparison arithmetic implementation')
    require(digest(read(str(RGB_FILE)))==RGB_SOURCE_SHA,'frozen RGB saved consumer code identity')
    if consumer is None:
        from experiments.spectral_reachability import p1_moving_overlay_recompile_rgb_cuda_20261027_v1 as rgb
        consumer=rgb.consume
    receipt=consumer(rgb_output,rgb_report_sha)  # Saved-only; never run()/prepare()/model loading.
    out=assemble(base,report,receipt)
    require(read(comparison)==base_raw and read(rp)==rgb_raw,'postconsume metadata TOCTOU')
    out['inputs']=dict(comparison=dict(file=comparison,sha256=comparison_sha),rgb_report=dict(file=rp,sha256=rgb_report_sha),
        rgb_source_sha256=RGB_SOURCE_SHA,comparison_source_sha256=base['binding']['source_sha256'])
    out['consume_receipt_sha256']=digest(canonical(receipt))
    return out


def write_new(path,value):
    path_value(path);p=Path(path);require(not p.exists() and not p.is_symlink(),'exclusive NEW output, never overwrite')
    raw=canonical(value);require(len(raw)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+CAP,'2MiB cap plus200MiB disk floor')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'full output write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return digest(raw)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('comparison','comparison-sha','rgb-output','rgb-report-sha','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args(argv);out=material(a.comparison,a.comparison_sha,a.rgb_output,a.rgb_report_sha)
    h=write_new(a.output,out);print(json.dumps(dict(file=a.output,sha256=h,status=out['status'],P1_admission=False)));return 0


if __name__=='__main__':raise SystemExit(main())
