"""Physical-v2 spec projection only; no authority over physical execution."""
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
NAME='agent_p1_intersection_goal_adapter_20261029_v2'
PHYSICAL='agent_p2_intersection_physical_20261029_v2'
SPEC='agent_p1_prospective_goal_semantics_20261029_v1.spec'
SOURCE=ROOT/'experiments/spectral_reachability'/f'{PHYSICAL}.py'
RECEIPT=ROOT/'results/agent_p2_intersection_host_cuda_20261029_v2_consume_MAIN01.json'
RECEIPT_SHA='bcaba6c3c4cbdd05dafc475ad45ad859930efd3587fdb8e8adba47f7da7b7432'
ROOTS=('a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7','df038a122a660da57675fa281ec7dd358c9f9f67cb387aca9595f31711ca60ad')
ORDER=((0,'off','joint'),(0,'off','composition'),(0,'off','grasp'),(1,'off','grasp'),(1,'off','composition'),(1,'off','joint'))

def need(ok,msg):
    if not ok: raise ValueError(msg)

def digest(data): return hashlib.sha256(data).hexdigest()

def read(path,expected):
    need(path.is_absolute() and path.is_file() and not path.is_symlink(),'absolute regular input')
    need(isinstance(expected,str) and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'explicit SHA256')
    data=path.read_bytes();need(digest(data)==expected,'SHA mismatch: '+str(path));return data

def goal_bits(g):
    need(isinstance(g,list) and len(g)==7 and all(type(x) in (float,int) and math.isfinite(x) for x in g),'goal7')
    return struct.pack('<7d',*g)

def project(p,path,prepare_sha,source_path,source_sha):
    need(p['schema']==PHYSICAL+'.prepare','physical v2 schema')
    c=p['contract']
    need(c['schema']==PHYSICAL+'.contract' and c['fixed_denominator']==6 and c['order']==[list(x) for x in ORDER] and c['checkpoints']==[0,5,10,15] and c['horizon']==15 and c['condition']=='off','v2 six-slot contract')
    need(c['known_roots']==2 and c['fresh_roots']==0 and c['selection_uses_physics'] is False and c['P1_admission'] is False,'scientific boundary')
    need(p['bindings'][str(source_path)]==source_sha,'v2 source binding')
    a=p['planner_authority'];need(a['producer']=='agent_p2_intersection_host_cuda_20261029_v2' and a['planner_receipt_sha']==RECEIPT_SHA and a['complete_consume_rebuilt'] is True and a['persisted_receipt_equal'] is True and a['selection_uses_physics'] is False and a['missing_exports']==[],'planner authority')
    roots=p['source_roots'];plans=p['plans'];need(len(roots)==2 and len(plans)==6,'fixed denominator')
    for i,r in enumerate(roots):need(r['root_slot']==i and r['root_sha256']==ROOTS[i] and len(goal_bits(r['goal']))==56,'root')
    slots=[]
    for i,(plan,(root,condition,arm)) in enumerate(zip(plans,ORDER)):
        need(plan is not None and (plan['index'],plan['root_slot'],plan['condition'],plan['arm'])==(i,root,condition,arm),'slot')
        native=plan['native_root'];need(native['root_slot']==root and native['root_sha256']==ROOTS[root] and goal_bits(native['goal'])==goal_bits(roots[root]['goal']),'native goal')
        need(plan['selection_uses_physics'] is False and isinstance(plan['selection_digest'],str) and len(plan['selection_digest'])==64,'plan seal')
        slots.append(dict(slot_id=f'{i}:{arm}',root_id=ROOTS[root],goal7=roots[root]['goal'],goal_authority_sha256=prepare_sha))
    return dict(schema=SPEC,slots=slots,checkpoints=[0,5,10,15],rgb_pairs=[],authority=dict(path=str(path),sha256=prepare_sha))

def build(prepared,prepare_sha,source_sha):
    need(prepared.name.startswith(PHYSICAL+'_'),'v2 physical prepare path')
    obj=json.loads(read(prepared,prepare_sha));read(SOURCE,source_sha);read(RECEIPT,RECEIPT_SHA)
    return project(obj,prepared,prepare_sha,SOURCE,source_sha)

def encode(obj): return (json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['project','verify']);ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--prepare-sha',required=True);ap.add_argument('--source-sha',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    data=encode(build(a.prepared,a.prepare_sha,a.source_sha));need(a.output.is_absolute() and a.output.name.startswith(NAME+'_'),'own absolute output')
    if a.command=='project':
        with a.output.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:need(a.output.read_bytes()==data,'projection mismatch')
    print(a.command,digest(data))

if __name__=='__main__':main()
