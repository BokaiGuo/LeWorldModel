"""Pinned saved-only H15 vs existing bounded GRASP and CEM, no simulator or NN."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

NAME = 'p2_h15_strong_baseline_saved_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'h15': ('results/p2_h15_physical_adapter_20261029_v1_MAIN01/report.json', 'aecf685083df2160df0d11929b2bd67f3366f2dbcc59bb111d879c1a78c9d185'),
    'six': ('results/agent_p2_intersection_physical_20261029_v2_consume_MAIN01.json', 'e96462847db675dc0b0c5376067df3654edd40805d6e374ffa3f8f1d5224e6c4'),
    'cem': ('results/agent_lewm_bounded_cem_h3_physical_20261029_v1_consume_MAIN01.json', '1e69a2fe082af3c5da32ec28ac8ba10b2375b5a593fa0bf69a60367d7cbc9b86'),
    'host': ('results/p2_h15_controlled_host_20261029_v1_cuda_MAIN01/report.json', '188fe5552b269db22739a77042a318d1f60d1c1c69cddcf4c5aead20501fe6ce'),
}
FIELDS = ('task_id','root_sha256','root_state','goal','goal_pose','goal_pixel_sha256')
METRICS = ('native_distance','object_position_error','wrapped_object_angle_error','native_success')


def need(ok, message):
    if not ok: raise ValueError(message)


def read(path, sha):
    data = Path(path).read_bytes()
    need(hashlib.sha256(data).hexdigest()==sha, 'source SHA mismatch')
    def pairs(items):
        out = {}
        for k,v in items:
            need(k not in out, 'duplicate key'); out[k]=v
        return out
    def invalid(_): raise ValueError('non-finite JSON')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def pair(a,b):
    need(a['arm']=='h15_sra' and b['arm'] in ('grasp','bounded_native_CEM_H3_actualmean'), 'arm')
    need(a['root_slot']==b['root_slot'] and a['condition']==b['condition']=='off' and
         a['initial_exact'] is b['initial_exact'] is True, 'root and condition')
    for k in FIELDS: need(a[k]==b[k], 'identity mismatch: '+k)
    for row in (a,b):
        n=row['own_returned']
        need(type(n) is int and 0<=n<=15 and row['status'] in ('native_stopped','horizon_completed') and
             (row['status']!='horizon_completed' or n==15) and
             [x['step'] for x in row['responses']]==list(range(1,n+1)), 'timeline')
    def snap(x):
        m=x['native_metrics']
        need(all(type(m[k]) in (int,float) and math.isfinite(m[k]) for k in METRICS[:3]) and
             type(m['native_success']) is bool, 'valid metrics')
        return {**{k:m[k] for k in METRICS},'terminated':x['terminated'],'truncated':x['truncated']}
    left={x['step']:snap(x) for x in a['responses']}; right={x['step']:snap(x) for x in b['responses']}
    timeline=[]
    for i in range(1,16):
        x=left.get(i); y=right.get(i)
        timeline.append(dict(step=i,h15=x,control=y,delta_native_distance=None if x is None or y is None else x['native_distance']-y['native_distance'],
            delta_object_position=None if x is None or y is None else x['object_position_error']-y['object_position_error']))
    return dict(root_slot=a['root_slot'],task_id=a['task_id'],root_sha256=a['root_sha256'],control=b['arm'],
        h15_last_step=a['own_returned'],control_last_step=b['own_returned'],control_status=b['status'],
        common_steps=sorted(set(left)&set(right)),timeline=timeline)


def build():
    s={k:read(ROOT/p,h) for k,(p,h) in SOURCES.items()}
    need(s['h15']['status']=='two_root_native_recorded_not_admitted' and s['six']['status']=='intersection6_native_recorded_not_admitted' and
         s['cem']['status']=='bounded_CEM_H3_known2_native_recorded_not_admitted', 'source status')
    need(len(s['h15']['rows'])==2 and len(s['six']['rows'])==6 and len(s['cem']['rows'])==2, 'fixed roster')
    out=[]; costs=[]
    for root in (0,1):
        h=[r for r in s['h15']['rows'] if r['root_slot']==root]
        need(len(h)==1, 'h15 roster')
        for arm,source in [('grasp','six'),('bounded_native_CEM_H3_actualmean','cem')]:
            c=[r for r in s[source]['rows'] if r['root_slot']==root and r['arm']==arm]
            need(len(c)==1,'baseline roster'); out.append(pair(h[0],c[0]))
        acc=[x for x in s['six']['planner_authority']['accounts'] if x['root']==root and x['arm']=='grasp']
        need(len(acc)==1,'GRASP account'); costs.append(dict(root=root,grasp=acc[0]['used']))
    host=s['host']; slots=host['slots']; need(len(slots)==2,'host roster')
    for root in (0,1):
        need(slots[root]['root']==root, 'host slot')
        slot=read(ROOT/('results/p2_h15_controlled_host_20261029_v1_cuda_MAIN01/root%d/report.json'%root),slots[root]['report_sha256'])
        costs[root]['h15_attempted']=slot['resources']['attempted']
    return dict(schema=NAME+'.report',sources={k:dict(path=str(ROOT/p),sha256=h) for k,(p,h) in SOURCES.items()},
        pairs=out,per_root_costs=costs,cem_both_roots_resources=s['cem']['planner_authority']['resources'],
        physical_counts=dict(h15=s['h15']['counts'],cem=s['cem']['counts'],six_all_arms=s['six']['counts']),
        trust='Pinned complete existing receipts and H15 report trusted; no full upstream reconstruction or new execution here.',
        interpretation='Retrospective same root, native goal and time only; different actions, histories and domains; not causal or independent.',
        cost_scope='six physical counts include all six arms; CEM model resources are two-root totals; axes differ and are not FLOPs.',
        delta='H15 minus control; lower native distance/object position error is numerically smaller; missing post-stop values remain null.',
        fresh_roots=0,equal_compute=False,fair_ranking=False,official_reproduction=False,new_neural_calls=0,new_physics_calls=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    need(args.output.is_absolute() and args.output.name.startswith(NAME+'_'), 'exclusive absolute output')
    data=(json.dumps(build(),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with args.output.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    print(hashlib.sha256(data).hexdigest())


if __name__=='__main__': main()
