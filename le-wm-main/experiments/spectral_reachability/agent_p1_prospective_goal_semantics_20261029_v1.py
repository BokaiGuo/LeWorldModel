"""Prospective, saved-only P1 goal semantics contract; no environment or model."""
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

NAME = 'agent_p1_prospective_goal_semantics_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / '.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
ENV_SHA = 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'
PROVENANCE = ROOT / 'innerdocs/agent_p1_native_goal_velocity_provenance_20261029_v1.md'
PROVENANCE_SHA = '54db9b7ae09ca48bcd5b4fbd93a946232f5f634ebed61441ea323f1cb7d75e6e'

def need(value, message):
    if not value:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path, expected):
    p = Path(path)
    need(p.is_absolute() and p.is_file() and not p.is_symlink(), 'absolute regular file')
    b = p.read_bytes()
    need(sha(b) == expected, 'external SHA mismatch: ' + str(p))
    return b

def vec(value):
    need(isinstance(value, list) and len(value) == 7 and all(type(v) in (int,float) and math.isfinite(v) for v in value), 'finite goal/state7')
    return [float(v) for v in value]

def bits(value):
    return struct.pack('<7d', *vec(value)).hex()

def metrics(goal, state):
    g, s = vec(goal), vec(state)
    d = [x-y for x,y in zip(g,s)]
    xy = math.fsum(x*x for x in d[:4]); angle = d[4]*d[4]; vel = math.fsum(x*x for x in d[5:])
    native_sq = math.fsum(x*x for x in d)
    native_angle = min(abs(d[4]), 2*math.pi-abs(d[4]))
    return dict(delta_goal_minus_state=d, agent_xy_squared=math.fsum(x*x for x in d[:2]), object_xy_squared=math.fsum(x*x for x in d[2:4]), position_squared=xy, raw_angle_squared=angle, velocity_squared=vel, native7_squared=native_sq, native7_distance=math.sqrt(native_sq), native_angle_term=native_angle, native_success=math.sqrt(xy)<20 and native_angle<math.pi/9)

def prepare(spec):
    need(set(spec) == {'schema','slots','checkpoints','rgb_pairs','authority'}, 'spec fields')
    need(spec['schema'] == NAME+'.spec', 'spec schema')
    cps = spec['checkpoints']; slots=spec['slots']; pairs=spec['rgb_pairs']
    need(isinstance(cps,list) and cps and all(type(x) is int and x>=0 for x in cps) and cps == sorted(set(cps)), 'fixed checkpoint order')
    need(isinstance(slots,list) and slots, 'fixed slots')
    need(all(isinstance(s,dict) and set(s)=={'slot_id','root_id','goal7','goal_authority_sha256'} and isinstance(s['slot_id'],str) and s['slot_id'] and isinstance(s['root_id'],str) and s['root_id'] and isinstance(s['goal_authority_sha256'],str) and len(s['goal_authority_sha256'])==64 and all(c in '0123456789abcdef' for c in s['goal_authority_sha256']) for s in slots), 'slot identity')
    for s in slots: vec(s['goal7'])
    ids=[s['slot_id'] for s in slots]; need(len(ids)==len(set(ids)), 'duplicate slot')
    need(isinstance(pairs,list) and all(isinstance(p,list) and len(p)==2 and p[0] in ids and p[1] in ids and p[0]!=p[1] for p in pairs), 'fixed RGB pair roster')
    need(len({tuple(sorted(p)) for p in pairs})==len(pairs), 'duplicate pair')
    authority=spec['authority']; need(isinstance(authority,dict) and set(authority)=={'path','sha256'}, 'spec authority')
    read(authority['path'],authority['sha256'])
    read(ENV,ENV_SHA); read(PROVENANCE,PROVENANCE_SHA)
    return dict(schema=NAME+'.prepare',spec=spec,source=dict(path=str(ENV),sha256=ENV_SHA),provenance=dict(path=str(PROVENANCE),sha256=PROVENANCE_SHA),native_gate_unchanged=True,P1_admission=False,ready_for_execution=False,interpretation='saved evidence contract only; authority file hash does not independently authenticate its meaning')

def analyze(prepared, receipt):
    need(prepared == prepare(prepared['spec']), 'prepare mismatch')
    need(set(receipt)=={'schema','authority','rows'} and receipt['schema']==NAME+'.receipt', 'receipt fields')
    need(set(receipt['authority'])=={'path','sha256'}, 'receipt authority')
    read(receipt['authority']['path'],receipt['authority']['sha256'])
    spec=prepared['spec']; slots=spec['slots']; cps=spec['checkpoints']; rows=receipt['rows']
    need(isinstance(rows,list) and len(rows)==len(slots), 'fixed slot denominator')
    outputs=[]; images={}
    for slot,row in zip(slots,rows):
        need(set(row)=={'slot_id','root_id','goal7','goal_authority_sha256','goal_rgb','points'} and row['slot_id']==slot['slot_id'] and row['root_id']==slot['root_id'] and row['goal_authority_sha256']==slot['goal_authority_sha256'] and bits(row['goal7'])==bits(slot['goal7']), 'original goal identity')
        image=row['goal_rgb']
        if image is not None:
            need(isinstance(image,dict) and set(image)=={'path','sha256','encoding'} and image['encoding']=='RGB224_raw_u8', 'RGB descriptor')
            pix=read(image['path'],image['sha256']); need(len(pix)==224*224*3, 'RGB size')
            images[slot['slot_id']]=pix
        points=row['points']; need(isinstance(points,list) and len(points)==len(cps), 'checkpoint denominator')
        results=[]
        for cp,p in zip(cps,points):
            need(isinstance(p,dict) and set(p)=={'checkpoint','state7','native_success','native_distance','missing_reason'} and p['checkpoint']==cp, 'checkpoint identity')
            if p['state7'] is None:
                need(p['native_success'] is None and p['native_distance'] is None and isinstance(p['missing_reason'],str) and bool(p['missing_reason']), 'missing point')
                results.append(dict(checkpoint=cp,metrics=None,returned_success=None,returned_distance=None,missing_reason=p['missing_reason']))
            else:
                need(p['missing_reason'] is None and type(p['native_success']) is bool and type(p['native_distance']) in (int,float) and math.isfinite(p['native_distance']) and p['native_distance']>=0, 'native return')
                m=metrics(slot['goal7'],p['state7']); need(m['native_success']==p['native_success'], 'native success mismatch')
                results.append(dict(checkpoint=cp,metrics=m,returned_success=p['native_success'],returned_distance=p['native_distance'],distance_recomputed_minus_returned=m['native7_distance']-p['native_distance'],missing_reason=None))
        outputs.append(dict(slot_id=slot['slot_id'],root_id=slot['root_id'],goal7=slot['goal7'],goal7_float64_bits=bits(slot['goal7']),goal_rgb_sha256=None if image is None else image['sha256'],points=results))
    lookup={s['slot_id']:s for s in slots}; comparisons=[]
    for a,b in spec['rgb_pairs']:
        ga,gb=lookup[a]['goal7'],lookup[b]['goal7']
        known=a in images and b in images
        # Same visual-state coordinates are required; rendering setup provenance remains external.
        same_pose=struct.pack('<5d',*vec(ga)[:5])==struct.pack('<5d',*vec(gb)[:5])
        velocity_bits_diff=struct.pack('<2d',*vec(ga)[5:])!=struct.pack('<2d',*vec(gb)[5:])
        velocity_diff=vec(ga)[5:] != vec(gb)[5:]
        same_rgb=(images[a]==images[b]) if known else None
        comparisons.append(dict(slots=[a,b],same_goal_first5_bits=same_pose,velocity_bits_differ=velocity_bits_diff,velocity_values_differ=velocity_diff,same_rgb=same_rgb,observed_rgb_collision=(same_pose and velocity_diff and same_rgb) if known else None))
    return dict(schema=NAME+'.analysis',receipt_authority=receipt['authority'],slots=outputs,rgb_pairs=comparisons,denominator=dict(slots=len(slots),checkpoints_per_slot=len(cps),rgb_pairs=len(spec['rgb_pairs'])),P1_admission=False,interpretation='same-image collision is a specific observed ambiguity, not a universal proof; source and receipt authority inherited, no upstream strict')

def encoded(x):
    return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def write(path,data):
    p=Path(path); need(p.is_absolute() and p.name.startswith(NAME+'_'), 'exclusive own absolute path')
    with p.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','analyze','verify']); p.add_argument('--spec',required=True); p.add_argument('--spec-sha',required=True); p.add_argument('--receipt'); p.add_argument('--receipt-sha'); p.add_argument('--output',required=True); a=p.parse_args()
    spec=json.loads(read(a.spec,a.spec_sha)); result=prepare(spec)
    if a.command!='prepare':
        need(a.receipt is not None and a.receipt_sha is not None, 'receipt and SHA required')
        receipt=json.loads(read(a.receipt,a.receipt_sha)); result=analyze(result,receipt)
    data=encoded(result)
    if a.command=='verify': need(Path(a.output).read_bytes()==data,'whole saved mismatch')
    else: write(a.output,data)
    print(a.command,sha(data))

if __name__=='__main__': main()
