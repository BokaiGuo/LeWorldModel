"""Project the exact saved intersection v2 physical prepare into P1 goal spec.

This is a pre-outcome input projection, not an execution or timing authority.
"""
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
NAME='agent_p1_intersection_goal_adapter_20261029_v1'
PHYSICAL='agent_p2_intersection_physical_20261029_v1'
SPEC='agent_p1_prospective_goal_semantics_20261029_v1.spec'
PREP=ROOT/'results/agent_p2_intersection_physical_20261029_v1_prepare_MAIN01.json'
PREP_SHA='fa4a694e24652fbd7edb804eb45814e7f70ae40e5579ff87d4352e5b4ba8b527'
SOURCE=ROOT/'experiments/spectral_reachability/agent_p2_intersection_physical_20261029_v1.py'
SOURCE_SHA='c6dc8d641f679a6e17079a921d3fd101dd25bc1f59620e6925cf97ee239eab3d'
RECEIPT=ROOT/'results/agent_p2_intersection_host_cuda_20261029_v2_consume_MAIN01.json'
RECEIPT_SHA='bcaba6c3c4cbdd05dafc475ad45ad859930efd3587fdb8e8adba47f7da7b7432'
ROOTS=('a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7','df038a122a660da57675fa281ec7dd358c9f9f67cb387aca9595f31711ca60ad')
ORDER=((0,'off','joint'),(0,'off','composition'),(0,'off','grasp'),(1,'off','grasp'),(1,'off','composition'),(1,'off','joint'))

def need(v,msg):
    if not v: raise ValueError(msg)

def sha(b): return hashlib.sha256(b).hexdigest()

def read_bound(path,expected):
    need(path.is_absolute() and path.is_file() and not path.is_symlink(),'absolute regular input')
    b=path.read_bytes(); need(sha(b)==expected,'input SHA mismatch'); return b

def goal_bits(g):
    need(isinstance(g,list) and len(g)==7 and all(type(x) in (float,int) and math.isfinite(x) for x in g),'goal7')
    return struct.pack('<7d',*g)

def project(p,prepare_path,prepare_sha):
    need(p['schema']==PHYSICAL+'.prepare','physical schema')
    c=p['contract']; need(c['schema']==PHYSICAL+'.contract' and c['order']==[list(x) for x in ORDER] and c['checkpoints']==[0,5,10,15] and c['horizon']==15 and c['condition']=='off','fixed contract')
    need(c['fixed_denominator']==4 and len(c['order'])==6,'unexpected inherited denominator; requires MAIN review')
    need(c['known_roots']==2 and c['fresh_roots']==0 and c['selection_uses_physics'] is False and c['P1_admission'] is False,'scientific boundary')
    a=p['planner_authority']; need(a['producer']=='agent_p2_intersection_host_cuda_20261029_v2' and a['planner_receipt_sha']==RECEIPT_SHA and a['complete_consume_rebuilt'] is True and a['persisted_receipt_equal'] is True and a['selection_uses_physics'] is False and a['missing_exports']==[], 'planner provenance')
    roots=p['source_roots']; plans=p['plans']; need(len(roots)==2 and len(plans)==6,'fixed six slots')
    for i,r in enumerate(roots): need(r['root_slot']==i and r['root_sha256']==ROOTS[i] and len(goal_bits(r['goal']))==56,'source root/goal')
    slots=[]
    for i,(plan,(root,condition,arm)) in enumerate(zip(plans,ORDER)):
        need(plan is not None and (plan['index'],plan['root_slot'],plan['condition'],plan['arm'])==(i,root,condition,arm),'fixed slot identity')
        native=plan['native_root']; need(native['root_slot']==root and native['root_sha256']==ROOTS[root] and goal_bits(native['goal'])==goal_bits(roots[root]['goal']),'physical native goal identity')
        need(plan['selection_uses_physics'] is False and isinstance(plan['selection_digest'],str) and len(plan['selection_digest'])==64,'sealed selection identity')
        slots.append(dict(slot_id=f'{i}:{arm}',root_id=ROOTS[root],goal7=roots[root]['goal'],goal_authority_sha256=prepare_sha))
    return dict(schema=SPEC,slots=slots,checkpoints=[0,5,10,15],rgb_pairs=[],authority=dict(path=str(prepare_path),sha256=prepare_sha))

def build():
    p=json.loads(read_bound(PREP,PREP_SHA)); read_bound(SOURCE,SOURCE_SHA); read_bound(RECEIPT,RECEIPT_SHA)
    need(p['bindings'][str(SOURCE)]==SOURCE_SHA,'source seal inside physical prepare')
    return project(p,PREP,PREP_SHA)

def encoded(x): return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['project','verify']);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    data=encoded(build());need(a.output.is_absolute() and a.output.name.startswith(NAME+'_'),'own absolute output')
    if a.command=='project':
        with a.output.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:need(a.output.read_bytes()==data,'projection mismatch')
    print(a.command,sha(data))

if __name__=='__main__':main()
