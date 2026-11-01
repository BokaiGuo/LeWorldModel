"""Independent saved-only Stage-C composition contract consumer.

Reads the already sealed paired physical directory.  It does not import or
execute a planner, simulator, encoder, producer, or existing composition
consumer.  It reports integrity and descriptive scope only; it never computes
or asserts a composition benefit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image

NAME='agent_p2_stagec_composition_saved_consume_20261028_v1'
EXPECTED_SCHEMA='p2_composition_paired_physical_20261027_v2.report'
PROTOCOL_SCHEMA='p2_composition_paired_physical_20261027_v2.execution'
PHYSICAL_SCHEMA='p2_composition_paired_physical_20261027_v2.physical_before_encoding'
METHODS=('joint_h2_no_composition','conditional_suffix_composition')
ROOTS=4
BRANCHES=8
CAP=64*1024**2

def need(ok,msg):
    if not ok: raise ValueError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(path):
    p=Path(path);need(p.is_file() and not p.is_symlink(),'regular saved file '+str(p))
    need(p.stat().st_size<=CAP,'saved file cap')
    return json.loads(p.read_text())

def strict_rows(report,protocol,physical):
    need(report['schema']==EXPECTED_SCHEMA and protocol['schema']==PROTOCOL_SCHEMA,'schema')
    need(report['status']=='calibration_recorded' and len(report['rows'])==BRANCHES and
         len(report['encodings'])==BRANCHES,'complete eight-row saved report')
    need(physical['schema']==PHYSICAL_SCHEMA and
         len(physical['rows'])==BRANCHES,'paired physical source has eight rows')
    for flag in ('P1_admission','P2_admission','calibration_only','no_neural_predict',
                 'physical_efficacy_verified','fair_total_budget_comparison','root_selection_changed'):
        need(report.get(flag) is (False if flag in ('P1_admission','P2_admission','physical_efficacy_verified','fair_total_budget_comparison','root_selection_changed') else True), 'scope flag '+flag)
    need(report['new_predict_calls']==0 and report['selected_path_env_step_attempts']==80 and
         report['selected_path_env_step_returns']==80 and report['attempted_terminal_encode_samples']==8,'saved totals')
    plans=protocol['plans']; need(len(plans)==BRANCHES,'eight plans')
    keys=[]
    for i,(plan,row,source) in enumerate(zip(plans,report['rows'],physical['rows'])):
        need(plan['root_index'] in range(ROOTS) and plan['method'] in METHODS,'fixed root/method')
        keys.append((plan['root_index'],plan['method']))
        need(row['root_sha256']==plan['root_sha256']==source['root_sha256'] and
             row['method']==plan['method']==source['method'] and row['selection_sha256']==plan['selection_sha256'], 'plan row binding')
        need(row['planned_commands']==10 and row['planned_horizon_tokens']==2 and
             row['attempted_commands']==10 and row['completed_commands']==10 and
             row['actual_endpoint_primitive_steps']==10 and row['full_horizon_completed'] is True,'full H2 endpoint')
        need(row['failures']==[] and source['failures']==[],'failure census')
    need(sorted(keys)==sorted((r,m) for r in range(ROOTS) for m in METHODS),'all root/method cells')
    for enc in report['encodings']:
        need(enc['root_sha256'] in {p['root_sha256'] for p in plans} and enc['method'] in METHODS and
             enc['status']=='completed' and enc['attempted_encode_samples']==1,'encoding row')
    return keys

def artifact_census(directory,report,protocol):
    d=Path(directory).resolve();bound={}
    for name in ('report.json','protocol.json','physical_report.json','notes.txt'):
        bound[name]=sha(d/name)
    for row in report['rows']:
        for field in ('journal_path','terminal_rgb_path'):
            p=Path(row[field]);need(p.parent==d,'artifact must stay in run directory')
            bound[p.name]=sha(p)
        stem=Path(row['journal_path']).name.removesuffix('.events.jsonl')
        for suffix in ('.physical.json','.encoding.json'):
            p=d/(stem+suffix);need(p.is_file(),'missing '+suffix);bound[p.name]=sha(p)
        events=[]
        for line in (d/Path(row['journal_path']).name).read_text().splitlines():
            e=json.loads(line);events.append(e)
        for e in events:
            if 'rgb_path' in e:
                p=d/Path(e['rgb_path']).name;need(p.is_file(),'missing PNG')
                with Image.open(p) as im:need(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'RGB224 PNG')
                need(sha(p)==e['rgb_file_sha256'],'PNG sha');bound[p.name]=sha(p)
        need(len(events)>=21 and events[0]['kind'] in ('initial','attempt'),'journal present')
        obs=[e for e in events if e.get('kind')=='observation']
        need(len(obs)==10 and [e['primitive_step'] for e in obs]==list(range(1,11)),'ten observations')
    # Existing directory may contain notes/source seals; require every regular file be
    # accounted for, preventing a hidden unbound artifact from entering a receipt.
    actual={p.name for p in d.iterdir() if p.is_file()}
    need(actual==set(bound),'exact saved artifact census')
    return bound

def consume(directory,report_sha,protocol_sha=None,physical_sha=None):
    d=Path(directory).resolve();rpath=d/'report.json';ppath=d/'protocol.json';vpath=d/'physical_report.json'
    need(sha(rpath)==report_sha,'external report sha')
    if protocol_sha is not None:need(sha(ppath)==protocol_sha,'external protocol sha')
    if physical_sha is not None:need(sha(vpath)==physical_sha,'external physical sha')
    report,protocol,physical=read(rpath),read(ppath),read(vpath)
    keys=strict_rows(report,protocol,physical);artifacts=artifact_census(d,report,protocol)
    # Deliberately descriptive: no cross-arm difference or benefit field is emitted.
    return dict(schema=NAME+'.consume',status='saved_contract_verified_not_admitted',
        branch_denominator=BRANCHES,root_denominator=ROOTS,methods=list(METHODS),keys=keys,
        full_h2_rows=BRANCHES,selected_public_steps=report['selected_path_env_step_returns'],
        terminal_encodings=report['attempted_terminal_encode_samples'],artifact_count=len(artifacts),
        new_predict_calls=0,neural_calls=0,physics_calls=0,training=False,
        calibration_only=True,P1_admission=False,P2_admission=False,
        physical_efficacy_verified=False,fair_total_budget_comparison=False,
        composition_benefit=None,scientific_claims=[],artifact_sha256=artifacts,
        source_report_sha256=report_sha,source_protocol_sha256=sha(ppath),source_physical_sha256=sha(vpath),
        note='saved integrity and contract only; no composition gain or causal claim')

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--report-sha',required=True)
    p.add_argument('--protocol-sha');p.add_argument('--physical-sha');a=p.parse_args(argv)
    print(json.dumps(consume(a.directory,a.report_sha,a.protocol_sha,a.physical_sha),sort_keys=True))

if __name__=='__main__':main()
