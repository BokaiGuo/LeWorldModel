"""Saved-only strong-baseline inventory with protocol-safe author/local split.

This module never imports a simulator or model.  It authenticates three sealed
local reports, keeps author-reported values separate, and refuses direct
comparison when protocol identities differ.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

NAME = 'agent_p2_strong_baseline_saved_summary_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
LOCAL = {
 'kinodynamic': (ROOT/'results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json','d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c'),
 'no_composition': (ROOT/'results/p1_moving_no_composition_baseline_cuda_20261027_v1_consume_main01.json','a05aa97f6c2688d7ae7a1fd8b961a23ccba1f4e45b14d95159cca63c5ba23167'),
 'sampling': (ROOT/'results/p1_sampling_physical_20261027_v1_main01/report.json','c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6'),
}
FLAGS = dict(P1_admission=False,P2_admission=False,scientific_admission=False)

def require(x,msg):
    if not x: raise ValueError(msg)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(path):
    p=Path(path); require(p.is_file() and not p.is_symlink(),'regular saved report'); require(p.stat().st_size<128*1024**2,'bounded report')
    return json.loads(p.read_bytes())
def write(path,value):
    p=Path(path); require(not p.exists(),'exclusive output'); p.write_bytes(canonical(value)+b'\n')
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()

def protocol(name,report):
    c=report.get('contract') or {}
    r=report.get('resources') or report.get('counts') or {}
    if name=='kinodynamic':
        return dict(family='kinodynamic',schema=report['schema'],roots=c.get('roots'),horizon=c.get('horizon'),
            history=c.get('history'),action_support=c.get('support_claim'),algorithm=c.get('algorithm'),
            arms=c.get('arms'),resources=r,equal_total_compute=c.get('equal_total_compute'),
            equal_search_domain=c.get('equal_search_domain'),equal_FLOPs=c.get('equal_FLOPs'),
            physics=report.get('new_physics'),admission=report.get('P1_admission'))
    if name=='no_composition':
        return dict(family='no_composition',schema=report['schema'],roots=len(report.get('rows',[])),horizon=2,
            history=next((x.get('history_semantics') for x in report.get('rows',[]) if x.get('history_semantics')),None),
            action_support='spectral_packet_mapped',algorithm='SVD/VJP bounded solve',arms=sorted({x.get('arm') for x in report.get('rows',[])}),
            resources=r,equal_total_compute=report.get('equal_total_compute_claim'),physics=False,
            admission=report.get('P1_admission'))
    return dict(family='sampling',schema=report['schema'],roots=c.get('roots'),horizon='H2',
        history=c.get('history'),action_support=None,algorithm='CEM/IID sampling',arms=c.get('arms'),resources=r,
        equal_total_compute=c.get('equal_compute'),physics=True,admission=report.get('P1_admission'))

def local_metrics(name,report):
    rows=[]
    for row in report.get('rows',[]):
        out={k:row[k] for k in ('source_slot','root_sha256','arm','status','model_terminal_cost','terminal_cost','attempted_steps','returned_steps','full_horizon','endpoint_rgb_fresh') if k in row}
        if name=='sampling':
            ev=row.get('official_evaluation') or {}; out['official_evaluation']=ev
            out['final_state']=row.get('final_state'); out['selection_sha256']=row.get('selection_sha256')
        if name=='kinodynamic':
            out['result_file']=row.get('result_file'); out['task_id']=row.get('task_id')
        rows.append(out)
    return rows

def load_local():
    out={}
    for name,(path,expected) in LOCAL.items():
        require(sha(path)==expected,'sealed local report drift '+name)
        report=read(path); require(report.get('errors',[])==[],'local report errors '+name)
        require(report.get('P1_admission') is False,'local P1 flag '+name)
        out[name]=dict(path=str(path),report_sha256=expected,schema=report.get('schema'),status=report.get('status'),
                       protocol=protocol(name,report),resources=report.get('resources') or report.get('counts'),
                       local_values=local_metrics(name,report),raw_keys=sorted(report),report=report)
    # Kinodynamic result archives are part of the local witness, not guessed metrics.
    kr=out['kinodynamic']['report']
    for row in kr.get('rows',[]):
        if row.get('result_file'):
            p=Path(out['kinodynamic']['path']).parent/row['result_file']
            require(p.is_file() and not p.is_symlink(),'kinodynamic result archive exists beside report')
            expected_archive=row.get('result_sha256')
            if expected_archive is not None: require(sha(p)==expected_archive,'kinodynamic result archive SHA')
            data=json.loads(gzip.open(p,'rt').read()); require(data.get('source_slot')==row.get('source_slot'),'archive slot identity')
    return out

def author_section(author_path=None):
    if author_path is None:
        return dict(status='unavailable',values=None,source=None,reason='no author-supplied report was bound; local values remain local')
    p=Path(author_path); require(p.is_file(),'author report path'); value=read(p)
    return dict(status='saved_unverified_until_protocol_check',values=value,source=str(p),report_sha256=sha(p))

def compare(a,b):
    pa,pb=a['protocol'],b['protocol']
    if pa != pb:
        return dict(status='protocol_mismatch_no_direct_comparison',protocol_a=pa,protocol_b=pb,delta=None)
    return dict(status='protocol_match_direct_comparison_allowed',protocol_a=pa,protocol_b=pb,
                delta=dict(note='numeric delta intentionally not inferred from heterogeneous row schemas'))

def build(author_path=None):
    local=load_local()
    comparisons={(a,b):compare(local[a],local[b]) for a,b in (('kinodynamic','no_composition'),('kinodynamic','sampling'),('no_composition','sampling'))}
    return dict(schema=NAME+'.summary',status='saved_local_baselines_verified_protocol_separated',
        local=local,author=author_section(author_path),comparisons=comparisons,
        denominators=dict(kinodynamic=len(local['kinodynamic']['local_values']),no_composition=len(local['no_composition']['local_values']),
                          sampling=len(local['sampling']['local_values'])),limitations=['author report unavailable unless explicitly supplied',
          'protocol mismatch blocks direct numeric comparison; no common task/horizon/resource claim is synthesized',
          'P1/P2/scientific admission remain false; saved verification is not scientific evidence'],**FLAGS)

def verify(path,expected,author_path=None):
    require(sha(path)==expected,'summary SHA'); saved=read(path); rebuilt=build(author_path)
    require(saved==rebuilt,'saved summary reconstruction and input closure'); return saved

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build'); p.add_argument('--output',type=Path,required=True); p.add_argument('--author-report',type=Path)
    p=sub.add_parser('verify'); p.add_argument('--summary',type=Path,required=True); p.add_argument('--sha',required=True); p.add_argument('--author-report',type=Path)
    a=ap.parse_args()
    if a.command=='build':
        value=build(a.author_report); write(a.output,value); result={'output':str(a.output.absolute()),'sha256':sha(a.output),'status':value['status']}
    else: result={'status':verify(a.summary,a.sha,a.author_report)['status'],'summary_sha256':a.sha}
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
