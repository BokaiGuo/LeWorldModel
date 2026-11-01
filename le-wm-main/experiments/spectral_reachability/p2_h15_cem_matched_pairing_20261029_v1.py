"""Saved-only same-root, same-step H15 vs matched-domain CEM physical audit."""
import argparse
import json
import math
import os
from pathlib import Path
from experiments.spectral_reachability import p2_h15_physical_adapter_20261029_v1 as h15
from experiments.spectral_reachability import p2_h15_cem_matched_physical_20261029_v1 as cem

NAME='p2_h15_cem_matched_pairing_20261029_v1'
ROOT=h15.ROOT
H_DIR=ROOT/'results/p2_h15_physical_adapter_20261029_v1_MAIN01'
H_SHA='aecf685083df2160df0d11929b2bd67f3366f2dbcc59bb111d879c1a78c9d185'
H_PREP=ROOT/'results/p2_h15_physical_adapter_20261029_v1_prepare_MAIN02.json'
H_PREP_SHA='a68ec15d03780345c1c9d2d8c68e741859a0b56e1bc8c3f6fb71be7a911d4dc8'
C_DIR=ROOT/'results/p2_h15_cem_matched_physical_20261029_v1_physics_CODEX01'
C_SHA='80d30187315ad70c4acdf66c4044e0529d96c612db56942486cf747134da6317'
C_PREP=ROOT/'results/p2_h15_cem_matched_physical_20261029_v1_prepare_CODEX01.json'
C_PREP_SHA='677443313c2754b74dcffc3888231e39461056242a88468a960a72b052ca3efd'
CH_SHA='3b675cb7bb99591152704a8fe1ef1e579524e4f3c6c839f37d7a9c78934ab799'
need=h15.need
METRICS=('native_distance','object_position_error','wrapped_object_angle_error')


def pair(a,b):
    need(a['arm']=='h15_sra' and b['arm']=='cem_matched' and a['root_slot']==b['root_slot'] and
         a['condition']==b['condition']=='off' and a['initial_exact'] is b['initial_exact'] is True,'arm/known root')
    for k in ('task_id','root_sha256','root_state','goal','goal_pose','goal_pixel_sha256'):
        need(a[k]==b[k],'root/goal mismatch '+k)
    for r in (a,b):
        n=r['own_returned'];need(type(n) is int and 0<=n<=15 and
            r['status'] in ('horizon_completed','native_stopped') and
            (r['status']!='horizon_completed' or n==15) and
            [x['step'] for x in r['responses']]==list(range(1,n+1)), 'contiguous returned timeline')
        need(all(not (x['terminated'] or x['truncated']) for x in r['responses'][:-1]),'no post-stop responses')
    def snap(x):
        m=x['native_metrics'];need(all(type(m[k]) in (float,int) and math.isfinite(m[k]) for k in METRICS) and
                                    type(m['native_success']) is bool,'finite native metrics')
        return {**{k:m[k] for k in METRICS},'native_success':m['native_success'],
                'terminated':x['terminated'],'truncated':x['truncated']}
    left={x['step']:snap(x) for x in a['responses']};right={x['step']:snap(x) for x in b['responses']};timeline=[]
    for step in range(1,16):
        x=left.get(step);y=right.get(step)
        timeline.append(dict(step=step,h15=x,cem=y,delta=None if x is None or y is None else {k:x[k]-y[k] for k in METRICS}))
    return dict(root_slot=a['root_slot'],task_id=a['task_id'],root_sha256=a['root_sha256'],
                h15_status=a['status'],cem_status=b['status'],h15_last_step=a['own_returned'],cem_last_step=b['own_returned'],
                common_steps=sorted(set(left)&set(right)),timeline=timeline)


def compare_axes(a,b):
    def sub(x,y):return x-y if type(x) in (int,float) and type(y) in (int,float) else None
    return {k:dict(h15=a.get(k),cem=b.get(k),difference_h15_minus_cem=sub(a.get(k),b.get(k))) for k in sorted(set(a)|set(b))}


def per_root_physical(report,root):
    counts={k:dict(attempted=0,completed=0) for k in report['counts']}
    for op in report['actual_operation_receipts']:
        if op['branch']==root:
            need(op['resource'] in counts,'physical axis');counts[op['resource']]['attempted']+=1
            counts[op['resource']]['completed']+=op['status']=='returned'
    return counts


def build():
    # Each consumer reconstructs its own upstream model trace and physical journal/PNGs.
    h=h15.consume(H_DIR,H_SHA,H_PREP,H_PREP_SHA);c=cem.consume(C_DIR,C_SHA,C_PREP,C_PREP_SHA)
    need(h['status']==c['status']=='two_root_native_recorded_not_admitted' and len(h['rows'])==len(c['rows'])==2,'complete consumers')
    hp=h15.physical.read(H_PREP,h15.physical.META);cp=h15.physical.read(C_PREP,h15.physical.META)
    hr=h15.physical.read(H_DIR/'report.json',h15.physical.META);cr=h15.physical.read(C_DIR/'report.json',h15.physical.META)
    hhost=h15.physical.read(h15.HOST_DIR/'report.json',h15.physical.META)
    chost=h15.physical.read(cem.HOST_DIR/'report.json',h15.physical.META)
    need(h15.physical.sha(h15.HOST_DIR/'report.json')==h15.HOST_SHA and h15.physical.sha(cem.HOST_DIR/'report.json')==CH_SHA,'host report SHA')
    pairs=[];resources=[]
    for root in (0,1):
        ha=h['rows'][root];ca=c['rows'][root];need(ha['root_slot']==ca['root_slot']==root,'root roster')
        need(hp['plans'][root]['task_sha256']==cp['plans'][root]['task_sha256'] and
             hp['source_roots'][root]==cp['source_roots'][root] and hp['warm_traces'][root]==cp['warm_traces'][root], 'task/native root/warm trace')
        pairs.append(pair(ha,ca))
        slots=[]
        for report,directory in ((hhost,h15.HOST_DIR),(chost,cem.HOST_DIR)):
            slot=report['slots'][root];path=directory/('root'+str(root))/'report.json'
            need(slot['root']==root and slot['status']=='complete' and h15.physical.sha(path)==slot['report_sha256'],'host slot source')
            slots.append(h15.physical.read(path,h15.physical.META)['resources'])
        aa,bb=slots
        physical_a=per_root_physical(hr,root);physical_b=per_root_physical(cr,root)
        resources.append(dict(root_slot=root,neural_attempted=compare_axes(aa['attempted'],bb['attempted']),
            neural_completed=compare_axes(aa['completed'],bb['completed']),neural_caps=compare_axes(aa['caps'],bb['caps']),
            neural_wall_ns=dict(h15=aa['wall_ns'],cem=bb['wall_ns']),
            neural_unknowns={k:dict(h15=aa[k],cem=bb[k]) for k in ('FLOPs','cuda_ns','backward_engine_calls')},
            physical_attempted=compare_axes({k:v['attempted'] for k,v in physical_a.items()},{k:v['attempted'] for k,v in physical_b.items()}),
            physical_completed=compare_axes({k:v['completed'] for k,v in physical_a.items()},{k:v['completed'] for k,v in physical_b.items()})))
    return dict(schema=NAME+'.report',sources=dict(h15_physical_sha256=H_SHA,cem_physical_sha256=C_SHA,
        h15_prepare_sha256=H_PREP_SHA,cem_prepare_sha256=C_PREP_SHA,h15_host_sha256=h15.HOST_SHA,cem_host_sha256=CH_SHA,
        analysis_code_sha256=h15.physical.sha(__file__)),pairs=pairs,resources=resources,
        whole_run=dict(h15_host_elapsed_seconds=hhost['elapsed_seconds'],cem_host_elapsed_seconds=chost['elapsed_seconds'],
            h15_cuda_peak=hhost['peak'],cem_cuda_peak=chost['peak'],h15_physical_wall_seconds=hr['wall_seconds'],cem_physical_wall_seconds=cr['wall_seconds']),
        difference='H15 minus CEM; native distance and object position error in native units; wrapped object angle error radians',
        endpoint='only actually returned same step; post-stop null, no terminal carry-forward',
        interpretation='known roots, retrospective separate trajectories, different algorithms; not independent generalization or causal effect',
        resource_boundary='same predictor samples and action domain, not equal FLOPs/total compute; CPU candidate construction is not fully metered; wall includes instrumentation',
        equal_total_compute=False,independent_generalization=False,physical_release=False,scientific_admission=False,
        new_neural_calls=0,new_physics_calls=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('build','verify'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--sha256');a=ap.parse_args()
    need(a.output.is_absolute() and a.output.name.startswith(NAME+'_') and not a.output.is_symlink(),'absolute exclusive namespace')
    if a.command=='build':
        need(a.sha256 is None,'build arguments');data=(json.dumps(build(),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        with a.output.open('xb') as f:need(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
        h15.physical.paired.fsync_dir(a.output.parent);result=dict(sha256=h15.physical.sha(a.output))
    else:
        need(a.sha256 and h15.physical.sha(a.output)==a.sha256,'external report SHA')
        need(h15.physical.read(a.output,h15.physical.META)==build(),'full saved rebuild');result=dict(verified=True)
    print(json.dumps(result))


if __name__=='__main__':main()
