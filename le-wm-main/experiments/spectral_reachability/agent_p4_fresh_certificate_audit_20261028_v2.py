"""Self-consistent saved-only P4 freshness/history certificate audit."""
from __future__ import annotations
import argparse, hashlib, io, json, os, zipfile
from pathlib import Path
import numpy as np

NAME='agent_p4_fresh_certificate_audit_20261028_v2'
ROOT=Path(__file__).resolve().parents[2]; CAP=64*1024**2
def need(ok,msg):
    if not ok: raise ValueError(msg)

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

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def json_input(key):
    rel,expected=PINS[key];p=ROOT/rel
    need(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'bounded '+key)
    need(digest(p)==expected,'pinned SHA '+key)
    def pairs(items):
        out={}
        for k,v in items: need(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(p.read_bytes(),object_pairs_hook=pairs)

def certificate_header(key):
    rel,expected=PINS[key];p=ROOT/rel;need(digest(p)==expected,'certificate SHA '+key)
    with zipfile.ZipFile(p) as z:
        names=sorted(x.filename for x in z.infolist())
        expected_names=['future_states.npy','goal.npy','goal_rgb.npy','observed_states.npy','raw_action_history.npy','root_history.npy','witness_commands.npy','witness_logits.npy']
        need(names==expected_names,'construction certificate members '+key)
        arrays={}
        for name in names:
            raw=z.read(name);need(len(raw)<=2*1024**2 and raw[:6]==b'\x93NUMPY','bounded NPY header '+name)
            a=np.load(io.BytesIO(raw),allow_pickle=False)
            arrays[name]={'shape':list(a.shape),'dtype':str(a.dtype)}
    return {'path':str(p),'sha256':expected,'members':names,'arrays':arrays,
            'disposition':'construction_only_not_fresh_or_history_certificate'}

def audit():
    j={k:json_input(k) for k in PINS if not k.startswith('certificate_')}
    root,consumer,cov,excl,pack,prep=[j[k] for k in ('root_manifest','consumer_manifest','coverage','exclusion','sourcepack','source_prepare')]
    need(root['freshness_certified'] is False and root['source_run_authorized'] is False and root.get('history_certificate') is None,'root unexpectedly certified')
    need(consumer['certified_fresh_roots']==[] and consumer['freshness_certified'] is False and consumer['source_run_authorized'] is False and consumer.get('history_certificate') is None,'consumer unexpectedly certified')
    need(cov['coverage_complete'] is False and cov['original_execution_bounds_certified']==0,'history coverage unexpectedly complete')
    need(excl['approval']['approved'] is False and excl['source_run_preconditions_met'] is False and excl['freshness_certified'] is False,'exclusion unexpectedly approved')
    need(pack['sourcepack_complete'] is True and pack['fixed_denominator']==2 and pack['reachable_count']==2 and pack['global_freshness']=='reused_recipe_calibration_not_fresh_not_same_task' and pack['new_independent_confirmation'] is False,'sourcepack freshness contract changed')
    need(prep.get('schema')=='p1_constructive_moving_history_source_20261027_v1.envelope' and isinstance(prep.get('plan_sha256'),str) and len(prep['plan_sha256'])==64,'source prepare envelope schema')
    unresolved=excl.get('unresolved_sources',[]);need(unresolved,'missing historical blocker inventory')
    certs=[certificate_header('certificate_s0'),certificate_header('certificate_s1')]
    bindings={str(ROOT/p):h for p,h in PINS.values()}
    return {'schema':NAME+'.report','status':'blocked_no_legal_fresh_certificate','consume_status':'blocked',
        'certificate_candidates':certs,'certificate_count':2,
        'certificate_disposition':'construction_artifacts_present_but_not_consumable_as_freshness_or_history',
        'sourcepack':{'sha256':PINS['sourcepack'][1],'fixed_denominator':2,'reachable_count':2,'global_freshness':pack['global_freshness'],'new_independent_confirmation':False},
        'freshness':{'root_manifest':False,'consumer_manifest':False,'exclusion_manifest':False,'certified_fresh_roots':[],'source_run_authorized':False},
        'history':{'history_certificate':None,'coverage_complete':False,'original_execution_bounds_certified':0,'unresolved_count':len(unresolved)},
        'blockers':['no certified_fresh_roots','history_certificate is null','coverage_complete=false and original_execution_bounds_certified=0','exclusion approval=false and source_run_preconditions_met=false','sourcepack is reused calibration, not fresh and not same task','new_independent_confirmation=false','reserved-root identity/access/disclosure remains unknown'],
        'unresolved_source_count':len(unresolved),'reserved_roots_opened':False,'approval_created':False,'certificate_created':False,
        'new_neural_calls':0,'new_physics_calls':0,'bindings':bindings,
        'limitations':['metadata absence is not proof of nonexistence','construction arrays are not fresh same-task authority']}

def consume(report,expected):
    p=Path(report).absolute();need(p.resolve()==p and p.is_file() and not p.is_symlink(),'audit report path')
    need(digest(p)==expected,'audit report SHA');raw=json.loads(p.read_bytes());a=audit()
    need(raw==a and a['status']=='blocked_no_legal_fresh_certificate' and a['certificate_created'] is False,'deterministic blocked report')
    return {'schema':NAME+'.consume','status':'blocked_no_legal_fresh_certificate','report_sha256':expected,'certified_fresh_roots':[],'history_certificate':None,'source_run_authorized':False,'approval_created':False,'certificate_created':False,'reserved_roots_opened':False,'blockers':a['blockers']}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('audit','consume'));ap.add_argument('--output');ap.add_argument('--report');ap.add_argument('--report-sha');a=ap.parse_args()
    if a.command=='audit':
        result=audit()
        if a.output:
            p=Path(a.output).absolute();raw=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            need(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and len(raw)<=CAP,'exclusive report path')
            with p.open('xb') as f:need(f.write(raw)==len(raw),'short report write');f.flush();os.fsync(f.fileno())
            result={'output':str(p),'report_sha256':digest(p),'status':result['status'],'blockers':result['blockers']}
    else: result=consume(a.report,a.report_sha)
    print(json.dumps(result,sort_keys=True,separators=(',',':')))
if __name__=='__main__':main()
