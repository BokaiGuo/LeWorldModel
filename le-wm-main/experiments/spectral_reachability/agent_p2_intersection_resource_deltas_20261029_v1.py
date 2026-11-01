"""Saved-only same-root resource-axis differences; never infer FLOPs."""
import argparse
import json
import math
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p2_intersection_v2_saved_analysis_20261029_v1 as prior
from experiments.spectral_reachability import agent_p2_common_resource_intersection_20261029_v1 as box

ROOT=Path(__file__).resolve().parents[2]
NAME='agent_p2_intersection_resource_deltas_20261029_v1'
PANEL=ROOT/'results/agent_p2_intersection_v2_saved_analysis_20261029_v1_MAIN01.json'
PANEL_SHA='5cc0c8e0bf370ba24a366555b2cbc7c29ef275f46206b4906f904cc64549a1b9'
PRIOR_SHA='af5c37de55fc99e319c1565794e184d4b7aa5a25be2f58849cba09082d911fbd'
BOX_SHA='fe02ea6917e49444f6bbae01b85533fb059e0739e032034e08fc11c710d332c3'
PAIRS=(('joint','composition'),('joint','grasp'),('composition','grasp'))
need=prior.need


def differences(a,b):
    need(set(a)==set(b),'matching axis names')
    out={}
    for k in sorted(a):
        x,y=a[k],b[k]
        need(all(v is None or (type(v) in (int,float) and math.isfinite(v)) for v in (x,y)), 'numeric or unknown')
        out[k]=None if x is None or y is None else x-y
    return out


def analyze(panel):
    need(panel['schema']==prior.NAME+'.analysis' and panel['fixed_denominator']==6 and
         panel['total_FLOPs'] is None and panel['equal_FLOPs'] is False and len(panel['rows'])==6,'six-slot panel ABI')
    by={}
    for i,(root,arm) in enumerate(prior.ORDER):
        row=panel['rows'][i]
        need((row['slot'],row['root'],row['arm'])==(i,root,arm),'fixed slot order')
        axes=row['resource_axes27'];need(set(axes)==set(box.AXES) and axes['FLOPs'] is None,'exact axes and unknown FLOPs')
        need(row['complete_account']['used']==axes and row['complete_account']['side']==row['side_axes'], 'account join')
        by[root,arm]=row
    pairs=[]
    for root in (0,1):
        for left,right in PAIRS:
            a,b=by[root,left],by[root,right]
            need(a['task_id']==b['task_id'] and a['goal_sha256']==b['goal_sha256'],'same root task and goal')
            pairs.append(dict(root=root,left=left,right=right,left_slot=a['slot'],right_slot=b['slot'],
                axes_left_minus_right=differences(a['resource_axes27'],b['resource_axes27']),
                side_left_minus_right=differences(a['side_axes'],b['side_axes']),
                elapsed_seconds_left_minus_right=a['elapsed_seconds']-b['elapsed_seconds']))
    return dict(schema=NAME+'.analysis',fixed_slots=6,fixed_same_root_pairs=6,pairs=pairs,
        input_panel_sha256=PANEL_SHA,total_FLOPs=None,equal_FLOPs=None,
        semantics='left minus right, per recorded axis; unknown propagates; no exchange rate or winner',
        authority='inherits MAIN strict receipt through prior panel; no independent neural or AD verification',
        scientific_admission=False)


def build():
    need(prior.sha(Path(prior.__file__))==PRIOR_SHA,'prior source SHA')
    need(prior.sha(Path(box.__file__))==BOX_SHA,'axis source SHA')
    need(prior.sha(PANEL)==PANEL_SHA and PANEL.stat().st_size<=32*1024**2,'panel SHA/size')
    raw=PANEL.read_bytes();panel=json.loads(raw)
    need(raw==prior.raw(panel) and panel==prior.build(),'prior panel fully rebuilt from fixed authorities')
    result=analyze(panel)
    need(prior.sha(PANEL)==PANEL_SHA and prior.sha(Path(prior.__file__))==PRIOR_SHA and
         prior.sha(Path(box.__file__))==BOX_SHA,'stable before/after')
    return result


def write(path,value):
    p=Path(path).absolute();data=prior.raw(value)
    need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(data)<=4*1024**2,'exclusive bounded output')
    with p.open('xb') as f:
        need(f.write(data)==len(data),'full write');f.flush();os.fsync(f.fileno())
    return dict(path=str(p),sha256=prior.sha(p))


def main():
    q=argparse.ArgumentParser();q.add_argument('command',choices=('build','verify'));q.add_argument('--output',required=True);q.add_argument('--sha256');a=q.parse_args()
    if a.command=='build':result=write(a.output,build())
    else:
        need(a.sha256 and prior.sha(a.output)==a.sha256,'output SHA')
        data=Path(a.output).read_bytes();value=json.loads(data)
        need(data==prior.raw(value) and value==build(),'whole reconstruction')
        result=dict(verified=True,sha256=a.sha256)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
