"""Saved-only retrospective same-root, same-step H15 physical comparisons."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p2_h15_physical_adapter_20261029_v1 as new
from experiments.spectral_reachability import agent_p2_intersection_physical_20261029_v2 as old

NAME='p2_h15_physical_pairing_20261029_v1'
ROOT=old.ROOT
NEW_DIR=ROOT/'results/p2_h15_physical_adapter_20261029_v1_MAIN01'
NEW_SHA='aecf685083df2160df0d11929b2bd67f3366f2dbcc59bb111d879c1a78c9d185'
NEW_PREP=ROOT/'results/p2_h15_physical_adapter_20261029_v1_prepare_MAIN02.json'
NEW_PREP_SHA='a68ec15d03780345c1c9d2d8c68e741859a0b56e1bc8c3f6fb71be7a911d4dc8'
OLD_DIR=ROOT/'results/agent_p2_intersection_physical_20261029_v2_MAIN01'
OLD_SHA='aba81af75d6de452f517f8b28a8d60a4baa7599877e96376f25d04c884e56a88'
OLD_RECEIPT=ROOT/'results/agent_p2_intersection_physical_20261029_v2_consume_MAIN01.json'
OLD_RECEIPT_SHA='e96462847db675dc0b0c5376067df3654edd40805d6e374ffa3f8f1d5224e6c4'
need=old.require
FIELDS=('native_distance','object_position_error','wrapped_object_angle_error','native_success')


def pair(a,b):
    need(a['arm']=='h15_sra' and b['arm'] in ('joint','composition') and
         a['root_slot']==b['root_slot'] and a['condition']==b['condition']=='off' and
         a['initial_exact'] is b['initial_exact'] is True,'same known off root')
    for k in ('task_id','root_sha256','root_state','goal','goal_pose','goal_pixel_sha256'):
        need(a[k]==b[k], 'same root/goal field '+k)
    for row in (a,b):
        steps=[x['step'] for x in row['responses']]
        need(steps==list(range(1,row['own_returned']+1)) and row['own_returned']<=15 and
             row['status'] in ('horizon_completed','native_stopped'), 'complete actual step sequence')
    left={x['step']:x for x in a['responses']};right={x['step']:x for x in b['responses']}
    def snapshot(row):
        return {**{k:row['native_metrics'][k] for k in FIELDS},'terminated':row['terminated'],'truncated':row['truncated']}
    timeline=[]
    for step in range(1,16):
        x=snapshot(left[step]) if step in left else None
        y=snapshot(right[step]) if step in right else None
        comp=None if x is None or y is None else dict(
            native_distance_delta=x['native_distance']-y['native_distance'],
            object_position_error_delta=x['object_position_error']-y['object_position_error'],
            wrapped_object_angle_error_delta=x['wrapped_object_angle_error']-y['wrapped_object_angle_error'])
        timeline.append(dict(step=step,h15=x,control=y,comparison=comp,
                             missing_arm=None if x is not None and y is not None else ('h15_sra' if x is None else b['arm'])))
    return dict(root_slot=a['root_slot'],control=b['arm'],task_id=a['task_id'],root_sha256=a['root_sha256'],
                h15_status=a['status'],control_status=b['status'],h15_last_step=a['own_returned'],
                control_last_step=b['own_returned'],common_steps=sorted(set(left)&set(right)),
                step14=timeline[13],step15=timeline[14],timeline=timeline)


def build():
    newer=new.consume(NEW_DIR,NEW_SHA,NEW_PREP,NEW_PREP_SHA)
    old_saved=old.read(OLD_RECEIPT,old.META)
    need(old.sha(OLD_RECEIPT)==OLD_RECEIPT_SHA,'external six-slot receipt SHA')
    six=old.consume(str(OLD_DIR),OLD_SHA)
    need(old.raw(six)==old.raw(old_saved) and six['status']=='intersection6_native_recorded_not_admitted' and
         newer['status']=='two_root_native_recorded_not_admitted','independent saved consumer reconstruction')
    need(len(newer['rows'])==2 and len(six['rows'])==6,'fixed original denominators')
    pairs=[];costs=[]
    host_report=old.read(new.HOST_DIR/'report.json',old.META)
    for root in (0,1):
        h=[r for r in newer['rows'] if r['root_slot']==root];need(len(h)==1,'one h15 root')
        hs=old.read(new.HOST_DIR/('root'+str(root))/'report.json',old.META)
        need(old.sha(new.HOST_DIR/('root'+str(root))/'report.json')==host_report['slots'][root]['report_sha256'],'HOST slot SHA')
        budget=hs['resources']['attempted']
        for arm in ('joint','composition'):
            ctrl=[r for r in six['rows'] if r['root_slot']==root and r['arm']==arm]
            acc=[x for x in six['planner_authority']['accounts'] if x['root']==root and x['arm']==arm]
            need(len(ctrl)==len(acc)==1,'one control and account')
            pairs.append(pair(h[0],ctrl[0]))
            costs.append(dict(root_slot=root,control=arm,h15_predict_samples=budget['predict_samples'],
                              h15_vjp_directions=budget['VJP_directions'],
                              control_predict_api=acc[0]['used']['predict_API'],control_vjp_directions=acc[0]['used']['VJP'],
                              h15_and_control_counts_not_equal_flops=True))
    return dict(schema=NAME+'.report',source=dict(h15_physical_report_sha256=NEW_SHA,h15_prepare_sha256=NEW_PREP_SHA,
        six_physical_report_sha256=OLD_SHA,six_receipt_sha256=OLD_RECEIPT_SHA,host_report_sha256=new.HOST_SHA,
        analysis_code_sha256=old.sha(__file__)),pairs=pairs,costs=costs,
        metric_semantics='native_distance and object_position_error: smaller numeric value; wrapped_object_angle_error: radians; delta = h15 minus control',
        endpoint_semantics='per-step only, no terminal carry-forward; step15 unavailable for stopped root1 joint',
        interpretation='descriptive retrospective known roots, not independent or prospective; outcome-matched steps, not identical actions or trajectories',
        cost_semantics='neural API/sample axes and VJP are different units; not FLOPs or fair total budget',
        independent_generalization=False,equal_compute=False,physical_release=False,scientific_admission=False)


def write_new(path):
    path=Path(path).absolute();need(path.name.startswith(NAME+'_') and not path.is_symlink(),'exclusive output')
    data=(json.dumps(build(),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with path.open('xb') as f:need(f.write(data)==len(data),'full write');f.flush();os.fsync(f.fileno())
    old.paired.fsync_dir(path.parent);return old.sha(path)


def verify(path,expected):
    need(old.sha(path)==expected,'external SHA')
    need(old.read(path,old.META)==build(),'full saved-only rebuild')
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('build','verify'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--sha256')
    a=ap.parse_args()
    if a.command=='build':need(a.sha256 is None,'no build SHA');result=dict(sha256=write_new(a.output))
    else:need(a.sha256 is not None,'external SHA required');verify(a.output,a.sha256);result=dict(verified=True)
    print(json.dumps(result))


if __name__=='__main__':main()
