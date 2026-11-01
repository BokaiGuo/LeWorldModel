"""Four-method, known-root saved physical comparison; no CUDA or physics."""
import argparse
import json
import os
from pathlib import Path
from experiments.spectral_reachability import p2_h15_gn_cem_threeway_pairing_20261029_v1 as three
from experiments.spectral_reachability import p2_h15_gradient_matched_physical_20261029_v2 as gradient

NAME='p2_h15_fourway_pairing_20261029_v1'
P=three.P
need=three.need
METHODS=('h15','cem','gn','gradient')
GRAD_DIR=gradient.ROOT/'results/p2_h15_gradient_matched_physical_20261029_v2_physics_CODEX01'
GRAD_SHA='f9d81f99760368579fca0d252c1ea40e192aa2af052900612fae45c38145e463'
GRAD_PREP=gradient.ROOT/'results/p2_h15_gradient_matched_physical_20261029_v2_prepare_CODEX01.json'
GRAD_PREP_SHA='cb35597351babcbdfce7ab299e2dc62bea09c9b2429e5ccb3a7eb3a47834deda'
GRAD_HOST_SHA='c544804ab9579fea0ef72fccac61c36ed7cc9450f74c551679f8d34b7b059ee1'


def pair_four(previous,h,grad):
    need(grad['arm']=='gradient_matched','gradient arm identity')
    # Existing pair validates root, goal, order, stops and native metrics.
    pair=three.old.pair(h,dict(grad,arm='cem_matched'))
    need(pair['root_slot']==previous['root_slot'] and pair['task_id']==previous['task_id'] and
         pair['root_sha256']==previous['root_sha256'],'four-way identity')
    timeline=[]
    for p,q in zip(previous['timeline'],pair['timeline'],strict=True):
        need(p['step']==q['step'] and p['h15']==q['h15'],'same H15 saved observation')
        g=q['cem'];entry=dict(p,gradient=g)
        for method in ('h15','cem','gn'):
            other=p[method]
            entry['gradient_minus_'+method]=None if g is None or other is None else {
                k:g[k]-other[k] for k in three.old.METRICS}
        timeline.append(entry)
    status=dict(previous['status'],gradient=grad['status'])
    last=dict(previous['last_step'],gradient=grad['own_returned'])
    return dict(root_slot=previous['root_slot'],task_id=previous['task_id'],root_sha256=previous['root_sha256'],
        status=status,last_step=last,common_steps=[x['step'] for x in timeline if all(x[m] is not None for m in METHODS)],timeline=timeline)


def extend_axes(existing,new):
    keys=set(existing)|set(new)
    return {k:{**{m:existing.get(k,{}).get(m) for m in three.METHODS},'gradient':new.get(k)} for k in sorted(keys)}


def build():
    prior=three.build()  # Revalidates each of the existing three physical/HOST sources.
    g=gradient.consume(GRAD_DIR,GRAD_SHA,GRAD_PREP,GRAD_PREP_SHA)
    need(g['status']=='two_root_native_recorded_not_admitted' and len(g['rows'])==2 and
         g['host_report_sha256']==GRAD_HOST_SHA,'complete gradient source')
    h=three.old.h15.consume(three.old.H_DIR,three.old.H_SHA,three.old.H_PREP,three.old.H_PREP_SHA)
    hp=P.read(three.old.H_PREP,P.META);gp=P.read(GRAD_PREP,P.META)
    pr=P.read(GRAD_DIR/'report.json',P.META)
    need(P.sha(gradient.HOST_DIR/'report.json')==GRAD_HOST_SHA,'gradient HOST SHA')
    gh=P.read(gradient.HOST_DIR/'report.json',P.META)
    pairs=[];resources=[]
    for root in (0,1):
        need(hp['plans'][root]['task_sha256']==gp['plans'][root]['task_sha256'] and
             hp['source_roots'][root]==gp['source_roots'][root] and
             hp['warm_traces'][root]==gp['warm_traces'][root],'task/root/goal/warm binding')
        need(h['rows'][root]['root_slot']==g['rows'][root]['root_slot']==root,'fixed root order')
        pairs.append(pair_four(prior['pairs'][root],h['rows'][root],g['rows'][root]))
        slot=gh['slots'][root];path=gradient.HOST_DIR/('root'+str(root))/'report.json'
        need(slot['root']==root and slot['status']=='complete' and P.sha(path)==slot['report_sha256'],'gradient slot sha')
        r=P.read(path,P.META)['resources'];base=prior['resources'][root]
        physical=three.old.per_root_physical(pr,root)
        resources.append(dict(root_slot=root,
            neural_attempted=extend_axes(base['neural_attempted'],r['attempted']),
            neural_completed=extend_axes(base['neural_completed'],r['completed']),
            neural_caps=extend_axes(base['neural_caps'],r['caps']),
            neural_wall_ns=dict(base['neural_wall_ns'],gradient=r['wall_ns']),
            neural_inclusive_operation_wall_ns=extend_axes(base['neural_inclusive_operation_wall_ns'],r['inclusive_operation_wall_ns']),
            neural_unknowns=extend_axes(base['neural_unknowns'],{k:r[k] for k in ('FLOPs','cuda_ns','backward_engine_calls')}),
            physical_attempted=extend_axes(base['physical_attempted'],{k:v['attempted'] for k,v in physical.items()}),
            physical_completed=extend_axes(base['physical_completed'],{k:v['completed'] for k,v in physical.items()})))
    return dict(schema=NAME+'.report',sources=dict(prior['sources'],gradient_physical_sha256=GRAD_SHA,
        gradient_prepare_sha256=GRAD_PREP_SHA,gradient_host_sha256=GRAD_HOST_SHA,
        previous_analysis_code_sha256=prior['sources']['analysis_code_sha256'],analysis_code_sha256=P.sha(__file__)),
        pairs=pairs,resources=resources,
        physical_caps_shared_two_roots=extend_axes(prior['physical_caps_shared_two_roots'],pr['contract']['caps']),
        whole_run=dict(prior['whole_run'],gradient=dict(host_elapsed_seconds=gh['elapsed_seconds'],cuda_peak=gh['peak'],
                                                        physical_wall_seconds=pr['wall_seconds'])),
        endpoint='warm10 then same returned control step; after stop null; different trajectories and wall times',
        interpretation='two known retrospective roots; not independent generalization or randomized causal effect',
        resource_boundary='unequal predictor and VJP budgets; FLOPs and CPU cost incomplete; no equal-total-compute claim',
        missing='FLOPs CUDA time backward engine calls unknown; CPU solve and other axes incompletely instrumented',
        equal_total_compute=False,independent_generalization=False,physical_release=False,scientific_admission=False,
        new_neural_calls=0,new_physics_calls=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('build','verify'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--sha256');a=ap.parse_args()
    need(a.output.is_absolute() and a.output.name.startswith(NAME+'_') and not a.output.is_symlink(),'exclusive namespace')
    if a.command=='build':
        need(a.sha256 is None,'build arguments');data=(json.dumps(build(),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        with a.output.open('xb') as f:need(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
        P.paired.fsync_dir(a.output.parent);result=dict(sha256=P.sha(a.output))
    else:
        need(a.sha256 and P.sha(a.output)==a.sha256,'external report SHA')
        need(P.read(a.output,P.META)==build(),'full saved rebuild');result=dict(verified=True)
    print(json.dumps(result))


if __name__=='__main__':main()
