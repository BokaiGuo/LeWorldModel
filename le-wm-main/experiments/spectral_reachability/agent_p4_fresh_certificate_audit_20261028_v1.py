"""Saved-only P4 certificate inventory and fail-closed consumer.

This checks only pinned metadata and the two existing construction NPZ headers.
It never opens a reserved fresh root, runs a producer, model, environment, or
physics, and never upgrades construction evidence into freshness authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

NAME='agent_p4_fresh_certificate_audit_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
CAP=64*1024**2
require=lambda ok,msg: (_ for _ in ()).throw(ValueError(msg)) if not ok else None

PINS={
 'root_manifest':('results/chat_freshroots_20261027_root_manifest.json','3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838'),
 'consumer_manifest':('results/chat_freshroots_20261027_consumer_manifest_v2.json','2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7'),
 'coverage':('results/agent_freshroots_resume_20261027_coverage.json','00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb'),
 'next_delta':('results/agent_freshroots_resume_20261027_next_delta.json','5ef07ab7815fff0c392feef59309c0116596e95d541eb95413d1d966a8b2a29b'),
 'inventory':('results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json','e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591'),
 'closure':('results/moving_authority_identity_link_closure_20261027_v1_MAIN01.json','7c7e1792c5a17986aaf3a38539a280ec46d5910eb4d9351babbe9bf2a70b3576'),
 'exclusion':('results/moving_root_exclusion_manifest_20261027_v1_inventory8_main01.json','a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060'),
 'sourcepack':('results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json','c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'),
 'source_prepare':('results/p1_constructive_moving_history_source_20261027_v1_prepare01.json','c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9'),
 'certificate_s0':('results/p1_constructive_moving_history_source_20261027_v1_main01/s0_construction_certificate.npz','96c3d5b644696f17852d42b900a05e3e9ab183060c89a36db883988caa4dbf9d'),
 'certificate_s1':('results/p1_constructive_moving_history_source_20261027_v1_main01/s1_construction_certificate.npz','6dcaf39084cfbe50783de81a95454afd78095a9b12348c1af7654afbbf3b1050'),
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_json(key):
    rel,h=PINS[key];p=ROOT/rel
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'bounded pinned '+key)
    require(sha(p)==h,'pinned SHA '+key)
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(p.read_bytes(),object_pairs_hook=pairs)

def npz_header(key):
    rel,h=PINS[key];p=ROOT/rel;require(sha(p)==h,'pinned certificate SHA '+key)
    with zipfile.ZipFile(p) as z:
        names=sorted(x.filename for x in z.infolist())
        require(names==['future_states.npy','goal.npy','goal_rgb.npy','observed_states.npy','raw_action_history.npy','root_history.npy','witness_commands.npy','witness_logits.npy'],'exact construction certificate members')
        shapes={}
        for name in names:
            raw=z.read(name);require(len(raw)<=2*1024**2,'bounded construction member')
            require(raw[:6]==b'\x93NUMPY','NPY member header')
            # Header-only inventory: no values are used to certify freshness.
            import numpy as np
            a=np.load(__import__('io').BytesIO(raw),allow_pickle=False,mmap_mode=None)
            shapes[name]=dict(shape=list(a.shape),dtype=str(a.dtype))
    return dict(path=str(p),sha256=h,members=names,arrays=shapes,disposition='construction_only_not_fresh_or_history_certificate')

def audit():
    v={k:read_json(k) for k in PINS if not k.startswith('certificate_')}
    root,consumer,cov,delta,inv,closure,excl,pack,prep=[v[k] for k in ('root_manifest','consumer_manifest','coverage','next_delta','inventory','closure','exclusion','sourcepack','source_prepare')]
    require(root['schema']=='chat_freshroots_20261027.root_manifest.v1' and root['freshness_certified'] is False and root['source_run_authorized'] is False and root.get('history_certificate') is None,'root manifest not certified')
    require(consumer['schema']=='chat_freshroots_20261027.consumer_manifest.v2' and consumer['certified_fresh_roots']==[] and consumer['freshness_certified'] is False and consumer['source_run_authorized'] is False and consumer.get('history_certificate') is None,'consumer not certified')
    require(cov['coverage_complete'] is False and cov['original_execution_bounds_certified']==0 and cov['source_run_authorized'] is False,'history coverage incomplete')
    require(excl['schema']=='moving_fresh_source_exclusions.v1' and excl['approval']['approved'] is False and excl['source_run_preconditions_met'] is False and excl['freshness_certified'] is False,'exclusion authority blocked')
    require(pack['schema']=='p1_constructive_moving_history_source_20261027_v1.sourcepack' and pack['sourcepack_complete'] is True and pack['fixed_denominator']==2 and pack['reachable_count']==2 and
            pack['global_freshness']=='reused_recipe_calibration_not_fresh_not_same_task' and pack['new_independent_confirmation'] is False,'construction sourcepack is not fresh authority')
    require(prep.get('schema')=='p1_constructive_moving_history_source_20261027_v1.prepare','source prepare schema')
    certs=[npz_header('certificate_s0'),npz_header('certificate_s1')]
    unresolved=list(excl.get('unresolved_sources',[]))
    blockers=[
        'no certified_fresh_roots in root/consumer manifests',
        'history_certificate is null in root/consumer manifests',
        'coverage_complete=false and original_execution_bounds_certified=0',
        'moving exclusion approval=false and source_run_preconditions_met=false',
        'construction sourcepack explicitly reused calibration, not fresh and not same task',
        'new_independent_confirmation=false; two construction certificates cannot certify freshness',
        'reserved-root identity/access/disclosure evidence remains unknown',
    ]
    require(unresolved,'expected unresolved historical authority blockers')
    bindings={str(ROOT/rel):h for rel,h in PINS.values()}
    return dict(schema=NAME+'.report',status='blocked_no_legal_fresh_certificate',consume_status='blocked',
        certificate_candidates=certs,certificate_count=len(certs),certificate_disposition='construction_artifacts_present_but_not_consumable_as_freshness_or_history',
        sourcepack=dict(path=str(ROOT/PINS['sourcepack'][0]),sha256=PINS['sourcepack'][1],fixed_denominator=2,reachable_count=2,global_freshness=pack['global_freshness'],new_independent_confirmation=False),
        freshness=dict(root_manifest=False,consumer_manifest=False,exclusion_manifest=False,certified_fresh_roots=[],source_run_authorized=False),
        history=dict(history_certificate=None,coverage_complete=False,original_execution_bounds_certified=0,unresolved_count=len(unresolved)),
        blockers=blockers,unresolved_source_ids=[x.get('source_id') for x in unresolved if isinstance(x,dict)],
        reserved_roots_opened=False,new_neural_calls=0,new_physics_calls=0,approval_created=False,certificate_created=False,
        bindings=bindings,limitations=['metadata absence is not proof of nonexistence','no access/disclosure ledger for reserved roots',
            'construction certificates contain state/RGB/history arrays but are not fresh same-task evidence'])

def consume(path,expected):
    p=Path(path).absolute();require(p.resolve()==p and p.is_file() and not p.is_symlink(),'audit report path')
    require(sha(p)==expected,'audit report SHA');raw=json.loads(p.read_bytes());a=audit()
    require(raw==a and a['status']=='blocked_no_legal_fresh_certificate' and a['consume_status']=='blocked','deterministic blocked consume')
    return dict(schema=NAME+'.consume',status='blocked_no_legal_fresh_certificate',report_sha256=expected,
        certified_fresh_roots=[],history_certificate=None,source_run_authorized=False,approval_created=False,certificate_created=False,
        reserved_roots_opened=False,blockers=a['blockers'])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('audit','consume'));ap.add_argument('--output');ap.add_argument('--report');ap.add_argument('--report-sha');a=ap.parse_args()
    if a.command=='audit':
        out=audit()
        if a.output:
            p=Path(a.output).absolute();raw=json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False).encode();require(p.name.startswith(NAME+'_') and not p.exists() and p.parent==ROOT/'results' and len(raw)<=CAP,'exclusive audit output')
            with p.open('xb') as f:require(f.write(raw)==len(raw),'short audit write');f.flush();os.fsync(f.fileno())
            out=dict(output=str(p),report_sha256=sha(p),status='blocked_no_legal_fresh_certificate',blockers=out['blockers'])
    else:out=consume(a.report,a.report_sha)
    print(json.dumps(out,sort_keys=True,separators=(',',':')))

if __name__=='__main__':main()
