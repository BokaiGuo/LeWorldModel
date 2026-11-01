"""Saved-only three-way known-root physical pairing; no inference or physics."""
import argparse
import json
import os
from pathlib import Path
from experiments.spectral_reachability import p2_h15_cem_matched_pairing_20261029_v1 as old
from experiments.spectral_reachability import p2_h15_gn_matched_physical_20261029_v1 as gn

NAME='p2_h15_gn_cem_threeway_pairing_20261029_v1'
P=old.h15.physical
need=old.need
G_DIR=gn.ROOT/'results/p2_h15_gn_matched_physical_20261029_v1_physics_CODEX01'
G_SHA='e158ae8de24543dfd7200f37676dc7121803a2de8fdce7efa7c40295ca93dbc0'
G_PREP=gn.ROOT/'results/p2_h15_gn_matched_physical_20261029_v1_prepare_CODEX01.json'
G_PREP_SHA='feb2078a5070d043e52b1ac58ffd8c77c86e3e4677b89fccc4d0e433b974e9a3'
G_HOST_SHA='2d41a16c0ad568c79c9b627fffe6e9cd608b0a38a2c26503542b3b935b7a1ceb'
METHODS=('h15','cem','gn')


def pair_three(h,c,g):
    hc=old.pair(h,c)
    # Reuse the exact same identity and timeline validator, without changing
    # the original GN row or interpreting it as a CEM physical observation.
    fake=dict(g,arm='cem_matched')
    need(g['arm']=='gn_matched','GN arm identity')
    hg=old.pair(h,fake)
    need(hc['root_slot']==hg['root_slot'] and hc['task_id']==hg['task_id'] and
         hc['root_sha256']==hg['root_sha256'],'same three-way identity')
    timeline=[]
    for x,y in zip(hc['timeline'],hg['timeline'],strict=True):
        need(x['step']==y['step'] and x['h15']==y['h15'],'same H15 observation')
        a,b,gstep=x['h15'],x['cem'],y['cem']
        def diff(left,right):
            return None if left is None or right is None else {k:left[k]-right[k] for k in old.METRICS}
        timeline.append(dict(step=x['step'],h15=a,cem=b,gn=gstep,
                             h15_minus_cem=diff(a,b),h15_minus_gn=diff(a,gstep),gn_minus_cem=diff(gstep,b)))
    return dict(root_slot=hc['root_slot'],task_id=hc['task_id'],root_sha256=hc['root_sha256'],
        status={m:r['status'] for m,r in zip(METHODS,(h,c,g),strict=True)},
        last_step={m:r['own_returned'] for m,r in zip(METHODS,(h,c,g),strict=True)},
        common_steps=[x['step'] for x in timeline if all(x[m] is not None for m in METHODS)],timeline=timeline)


def axes(values):
    need(set(values)==set(METHODS),'three method resources')
    keys=set().union(*(set(x) for x in values.values()))
    return {k:{m:values[m].get(k) for m in METHODS} for k in sorted(keys)}


def build():
    # Re-runs both existing physical consumers including their upstream HOST consumers.
    prior=old.build()
    g=gn.consume(G_DIR,G_SHA,G_PREP,G_PREP_SHA)
    need(g['status']=='two_root_native_recorded_not_admitted' and len(g['rows'])==2,'complete GN physical consumer')
    # Existing three consumers each independently validate journal, actions and upstream source closure.
    h=old.h15.consume(old.H_DIR,old.H_SHA,old.H_PREP,old.H_PREP_SHA)
    c=old.cem.consume(old.C_DIR,old.C_SHA,old.C_PREP,old.C_PREP_SHA)
    prepares=[P.read(p,P.META) for p in (old.H_PREP,old.C_PREP,G_PREP)]
    physical_reports=[P.read(p/'report.json',P.META) for p in (old.H_DIR,old.C_DIR,G_DIR)]
    host_dirs=[old.h15.HOST_DIR,old.cem.HOST_DIR,gn.HOST_DIR]
    host_shas=[old.h15.HOST_SHA,old.CH_SHA,G_HOST_SHA]
    hosts=[]
    for path,sha in zip(host_dirs,host_shas,strict=True):
        need(P.sha(path/'report.json')==sha,'host report sha')
        hosts.append(P.read(path/'report.json',P.META))
    pairs=[];resources=[]
    for root in (0,1):
        for p in prepares[1:]:
            need(prepares[0]['plans'][root]['task_sha256']==p['plans'][root]['task_sha256'] and
                 prepares[0]['source_roots'][root]==p['source_roots'][root] and
                 prepares[0]['warm_traces'][root]==p['warm_traces'][root],'same task root and warm history')
        rows=[x['rows'][root] for x in (h,c,g)]
        need(all(r['root_slot']==root for r in rows),'fixed root roster')
        pair=pair_three(*rows)
        need(pair['timeline'][0]['h15']==prior['pairs'][root]['timeline'][0]['h15'],'existing pairing binding')
        pairs.append(pair)
        slots={}
        for method,report,directory in zip(METHODS,hosts,host_dirs,strict=True):
            item=report['slots'][root];path=directory/('root'+str(root))/'report.json'
            need(item['root']==root and item['status']=='complete' and P.sha(path)==item['report_sha256'],'host slot source')
            slots[method]=P.read(path,P.META)['resources']
        physical={m:old.per_root_physical(r,root) for m,r in zip(METHODS,physical_reports,strict=True)}
        resources.append(dict(root_slot=root,
            neural_attempted=axes({m:slots[m]['attempted'] for m in METHODS}),
            neural_completed=axes({m:slots[m]['completed'] for m in METHODS}),
            neural_caps=axes({m:slots[m]['caps'] for m in METHODS}),
            neural_wall_ns={m:slots[m]['wall_ns'] for m in METHODS},
            neural_inclusive_operation_wall_ns=axes({m:slots[m]['inclusive_operation_wall_ns'] for m in METHODS}),
            neural_unknowns=axes({m:{k:slots[m][k] for k in ('FLOPs','cuda_ns','backward_engine_calls')} for m in METHODS}),
            physical_attempted=axes({m:{k:v['attempted'] for k,v in physical[m].items()} for m in METHODS}),
            physical_completed=axes({m:{k:v['completed'] for k,v in physical[m].items()} for m in METHODS})))
    return dict(schema=NAME+'.report',sources=dict(h15_physical_sha256=old.H_SHA,cem_physical_sha256=old.C_SHA,
        gn_physical_sha256=G_SHA,h15_prepare_sha256=old.H_PREP_SHA,cem_prepare_sha256=old.C_PREP_SHA,
        gn_prepare_sha256=G_PREP_SHA,h15_host_sha256=old.h15.HOST_SHA,cem_host_sha256=old.CH_SHA,
        gn_host_sha256=G_HOST_SHA,analysis_code_sha256=P.sha(__file__)),pairs=pairs,resources=resources,
        physical_caps_shared_two_roots=axes({m:report['contract']['caps'] for m,report in zip(METHODS,physical_reports,strict=True)}),
        whole_run={m:dict(host_elapsed_seconds=host['elapsed_seconds'],cuda_peak=host['peak'],physical_wall_seconds=pr['wall_seconds'])
                   for m,host,pr in zip(METHODS,hosts,physical_reports,strict=True)},
        endpoint='warm10 then same returned control step; after stop null; different physical trajectories and wall times',
        interpretation='two known retrospective roots; not independent generalization or randomized causal effect',
        resource_boundary='common action domain, unequal predictor and VJP counts, not equal FLOPs or total compute; numpy GN solves and other CPU work not fully metered; wall includes audits',
        missing='FLOPs CUDA time backward engine calls unknown; CPU solve axis is not complete; no terminal forward filling',
        equal_total_compute=False,independent_generalization=False,physical_release=False,scientific_admission=False,
        new_neural_calls=0,new_physics_calls=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('build','verify'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--sha256');a=ap.parse_args()
    need(a.output.is_absolute() and a.output.name.startswith(NAME+'_') and not a.output.is_symlink(),'absolute own namespace')
    if a.command=='build':
        need(a.sha256 is None,'build arguments');data=(json.dumps(build(),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        with a.output.open('xb') as f:need(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
        P.paired.fsync_dir(a.output.parent);result=dict(sha256=P.sha(a.output))
    else:
        need(a.sha256 and P.sha(a.output)==a.sha256,'external report SHA')
        need(P.read(a.output,P.META)==build(),'full rebuild');result=dict(verified=True)
    print(json.dumps(result))


if __name__=='__main__':main()
