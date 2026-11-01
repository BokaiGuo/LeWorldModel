"""Two frozen historical schema adapters. Saved metadata, never certification.

No discovery, checkpoint loading, producer/model/environment import or writes.
CLI reads exactly two historical sourcepacks and one pinned metadata capture.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import moving_root_exclusion_manifest_20261027_v1 as old

NAME='agent_p4_fresh_schema_projection_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
PINS=(
 ('conditional','results/p1_conditional_calibration2_main_20261027_v1/sourcepack.json','728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057'),
 ('observed','results/p1_observed_history_source_20261027_v1_main01/sourcepack.json','0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546'),
 ('capture','results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json','e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591'))
CAPS={'conditional':128*1024,'observed':128*1024,'capture':4*1024**2}
need=old.require


def sha(b):return hashlib.sha256(b).hexdigest()
def pointer(path):return '/'+ '/'.join(str(x).replace('~','~0').replace('/','~1') for x in path)


def parse(raw):
    def unique(items):
        out={}
        for k,v in items:need(k not in out,'duplicate metadata key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:need(False,'nonfinite metadata'))


def patterns(kind):
    p=[('schema',)]
    if kind=='conditional':
        b=('slots','*');a=b+('proposals','*','attempts','*')
        p += [b+('slot',k) for k in ('candidate_id','root_seed')]
        p += [b+('slot','reset_seed_span','*'),b+('proposals','*','proposal_index')]
        p += [a+(k,) for k in ('attempt_index','candidate_id','proposal_index','random_reset_seed','placement_reset_seed','root_sha256')]
        p += [a+('state','*'),a+('state',)]
    else:
        b=('rows','*');m=b+('metadata',)
        p += [b+(k,) for k in ('candidate_id','random_reset_seed','placement_reset_seed')]
        p += [m+(k,) for k in ('candidate_id','root_sha256','parent_root_sha256')]
        p += [m+('parent_source','root_seed'),m+('parent_source','reset_seed_span','*')]
        p += [b+('checkpoints','*',k) for k in ('file','sha256','rgb_sha256','public_step')]
    return p


def project(raw,kind):
    need(kind in ('conditional','observed') and len(raw)<=CAPS[kind],'bounded known schema input')
    p=old.Projection(BytesIO(raw),patterns(kind));p.limit=CAPS[kind]
    p.walk();p.whitespace();need(p.peek() is None,'trailing metadata')
    schema='p1_conditional_procedural_source_20261027_v1.sourcepack' if kind=='conditional' else 'p1_observed_history_source_20261027_v1.sourcepack'
    need(p.leaves.get(('schema',))==schema,'exact historical source schema')
    return p


def get(p,path):
    need(path in p.leaves,'required metadata '+pointer(path));return p.leaves[path]


def integer(v):old.integer(v);return v


def span(p,path):
    need(p.containers.get(path)==[0,1],'exact half-open reset span')
    v=[integer(get(p,path+(i,))) for i in (0,1)];need(v[0]<v[1],'nonempty reset span');return v


def record(prefix):
    return dict(record_id=pointer(prefix),root_state=None,root_state_sha256=None,physical_pose_key=None,
                root_snapshot_sha256=None,history_pixels_sha256=None,raw_history_sha256=None,
                maps_to=[],physical_equivalence_proved=False,identity_complete=False)


def conditional(raw):
    p=project(raw,'conditional');need(p.containers.get(('slots',))==[0,1],'all two conditional slots')
    slots=[];attempts=[]
    for i in range(2):
        b=('slots',i);sid=integer(get(p,b+('slot','candidate_id')));need(sid==i,'original conditional slot order')
        seed=integer(get(p,b+('slot','root_seed')));bounds=span(p,b+('slot','reset_seed_span'))
        need(bounds[1]-bounds[0]==4,'original four attempts per slot reservation')
        slots.append(dict(record_id=pointer(b),root_seed=seed,reset_seed_span=bounds))
        need(p.containers.get(b+('proposals',))==[0,1],'all two proposal rows, not selected only')
        for j in range(2):
            pb=b+('proposals',j)
            need(get(p,pb+('proposal_index',))==j and type(get(p,pb+('proposal_index',))) is int,'proposal index')
            need(p.containers.get(pb+('attempts',))==[0,1],'all two attempt rows')
            for k in range(2):
                a=pb+('attempts',k);r=record(a)
                for key,want in (('candidate_id',i),('proposal_index',j),('attempt_index',k)):
                    v=get(p,a+(key,));need(type(v) is int and v==want,'original attempt coordinates')
                reset=integer(get(p,a+('random_reset_seed',)));placement=integer(get(p,a+('placement_reset_seed',)))
                need(reset==bounds[0]+2*j+k and placement==0,'original reset span order and explicit placement seed')
                h=p.leaves.get(a+('root_sha256',))
                if h is not None:old.hash_value(h)
                r.update(root_seed=seed,reset_seed=reset,placement_reset_seed=placement,root_goal_sha256=h)
                state=p.containers.get(a+('state',))
                if state is not None:
                    need(state==list(range(7)),'exact state7 without missing components')
                    v=[get(p,a+('state',n)) for n in range(7)]
                    need(all(type(x) in (int,float) for x in v),'numeric state not bool')
                    s=np.asarray(v,dtype='<f8');need(np.isfinite(s).all(),'finite saved state')
                    r.update(root_state=v,**old.state_identity(v))
                else:need(p.leaves.get(a+('state',)) is None,'malformed state payload')
                r['missing']=['root_snapshot_sha256','history_pixels_sha256','raw_history_sha256']
                if r['root_state'] is None:r['missing'].append('root_state')
                if h is None:r['missing'].append('root_goal_sha256')
                attempts.append(r)
    return dict(schema=p.leaves[('schema',)],slots=slots,attempts=attempts,slot_denominator=2,
                proposal_denominator=4,attempt_denominator=8,original_execution_bound_certified=False)


def observed(raw):
    p=project(raw,'observed');need(p.containers.get(('rows',))==[0,1],'all two observed rows')
    rows=[]
    for i in range(2):
        b=('rows',i);m=b+('metadata',);r=record(b)
        need(integer(get(p,b+('candidate_id',)))==i and integer(get(p,m+('candidate_id',)))==i,'observed candidate identity')
        reset=integer(get(p,b+('random_reset_seed',)));placement=integer(get(p,b+('placement_reset_seed',)))
        bounds=span(p,m+('parent_source','reset_seed_span'));need(bounds[0]<=reset<bounds[1] and placement==0,'observed actual reset and parent reservation')
        root=get(p,m+('root_sha256',));parent=get(p,m+('parent_root_sha256',))
        old.hash_value(root);old.hash_value(parent)
        checkpoints=[];indices=p.containers.get(b+('checkpoints',),[])
        need(len(indices)<=3,'at most original three checkpoints, partial retained')
        for j in indices:
            q=b+('checkpoints',j);v={key:get(p,q+(key,)) for key in ('file','sha256','rgb_sha256','public_step')}
            need(type(v['file']) is str and Path(v['file']).name==v['file'] and v['file'].endswith('.npz'),'flat checkpoint descriptor only')
            old.hash_value(v['sha256']);old.hash_value(v['rgb_sha256'])
            need(type(v['public_step']) is int and v['public_step']==(0,5,10)[j],'original checkpoint sequence')
            checkpoints.append(dict(pointer=pointer(q),**v))
        r.update(root_seed=None,reset_seed=reset,placement_reset_seed=placement,root_goal_sha256=root,
                 parent_root_goal_sha256=parent,parent_source=dict(root_seed=integer(get(p,m+('parent_source','root_seed'))),reset_seed_span=bounds),
                 checkpoints=checkpoints,checkpoint_payloads_opened=False,
                 missing=['own_root_state_from_original_checkpoint','root_snapshot_sha256','history_pixels_sha256','raw_history_sha256','MAIN_identity_mapping_approval'])
        rows.append(r)
    return dict(schema=p.leaves[('schema',)],rows=rows,row_denominator=2,
                observed_own_state_unknown=True,physical_equivalence_proved=False)


def inherited(raw):
    need(len(raw)<=CAPS['capture'],'capture metadata cap');v=parse(raw)
    need(v['schema']=='root_exclusion_authority_resolution_20261027_v1.resolution_request','fixed resolution capture schema')
    files=v['history_file_requests'];need(type(files) is list and len(files)==618,'all 618 captured sources')
    paths=set();rows=[];refs=[]
    for f in files:
        need(f['path'] not in paths,'unique capture paths');paths.add(f['path']);old.hash_value(f['sha256'])
        need(type(f['inherited_refs']) is list and type(f['blockers']) is list,'explicit inherited metadata')
        for ref in f['inherited_refs']:
            need(ref['parent'] in ('moving_prepare','reachable_prepare') and type(ref['pointer']) is str and type(ref['original_reason']) is str,'original reference coordinates/reason')
            refs.append(ref)
        rows.append({k:deepcopy(f[k]) for k in ('path','sha256','inherited_refs','blockers')})
    need(len(refs)==960 and len({f['path'] for f in files if f['inherited_refs']})==480,'all 960 occurrences /480 path groups')
    need(len({(f['path'],r['parent'],r['pointer']) for f in files for r in f['inherited_refs']})==960,'unique original reference coordinates')
    need(sum(r['original_reason']=='historical_reset_attempt_bound_not_attested' for r in refs)==16,'all 16 reset-bound references retained')
    return dict(sources=rows,summary=deepcopy(v['summary']),authority_requests=deepcopy(v['authority_requests']),
                inherited_unknown_occurrences=960,inherited_unique_paths=480,reset_bound_unattested_refs=16,
                original_unknowns_discharged=0)


def build(payloads):
    need(set(payloads)=={'conditional','observed','capture'},'exact three inputs, no following references')
    c=conditional(payloads['conditional']);o=observed(payloads['observed']);h=inherited(payloads['capture'])
    return dict(schema=NAME+'.projection',status='schema_projected_historical_gate_BLOCKED',conditional=c,observed=o,
                inherited=h,input_payload_sha256={k:sha(v) for k,v in payloads.items()},
                source_run_preconditions_met=False,coverage_approved=False,freshness_certified=False,
                confirmed=False,P1_admission=False,P4_admission=False,physical_release=False,
                original_manifest_compatible=False,
                remaining=['observed_own_state_snapshot_history_unknown','conditional_snapshot_history_unknown',
                           'all_960_original_unknowns_retained','original_A_H_reset_bound_and_MAIN_scope_approval_required'])


def saved():
    payloads={}
    for kind,relative,h in PINS:
        p=ROOT/relative;need(p.resolve()==p and p.is_file() and not p.is_symlink(),'pinned canonical metadata only')
        need(p.stat().st_size<=CAPS[kind],'bounded input file');b=p.read_bytes()
        need(len(b)<=CAPS[kind] and sha(b)==h,'frozen original bytes SHA');payloads[kind]=b
    result=build(payloads)
    for kind,relative,h in PINS:need(sha((ROOT/relative).read_bytes())==h,'input unchanged after projection')
    closure=(Path(__file__).resolve(),Path(old.__file__).resolve(),Path(old.lexical.__file__).resolve())
    result['source_sha256']={str(p):sha(p.read_bytes()) for p in closure}
    result['input_paths']={kind:str(ROOT/relative) for kind,relative,_ in PINS}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    out=json.dumps(saved(),sort_keys=True,allow_nan=False)
    need(len(out.encode())<=2*1024**2,'2MiB metadata output cap');print(out)


if __name__=='__main__':main()
