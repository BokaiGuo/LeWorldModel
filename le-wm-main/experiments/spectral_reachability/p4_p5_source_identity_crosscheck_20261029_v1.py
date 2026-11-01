"""Re-extract fixed A-H source identity fields and cross-check saved recovery.

Only named identity fields are decoded from original reports. This is not reset
coverage, original-load provenance, an approval, or a fresh cohort.
"""
import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_freshroot_authority_20261028_v1 as authority
from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as lexical

NAME = 'p4_p5_source_identity_crosscheck_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT/'results/agent_p4_historical_source_identity_20261029_v1_LOCAL01.json'
RECOVERY_SHA = '577580e536598d850cd449fda2a02c314197267bdc41bff9d85692432165e724'
FIELDS = ('root_id','root_sha256','causal_history_action_sha256','causal_history_rgb_sha256','history_rgb_sha256','goal_rgb_sha256')
MISSING = ('episode_id','reset_attempt_id','reset_seed','root_state7','root_snapshot','raw_history_actions','individual_history_rgb_sha256')
need = authority.need


def extract(raw, src):
    sid = src['id']; need(sid in 'ABCDEFGH' and len(sid)==1, 'fixed source id')
    container = 'calibration_roots' if sid in 'ABC' else 'rows'
    projected = lexical.project(BytesIO(raw), [(container,'*',key) for key in FIELDS])
    indices = projected.containers.get((container,), [])
    n = 24 if sid == 'D' else 4
    need(indices == list(range(n)), 'all original selected rows')
    result = []
    for i in indices:
        values = {}
        for key in FIELDS:
            ptr = (container,i,key)
            if ptr not in projected.leaves: continue
            value = projected.leaves[ptr]
            if key == 'root_id': need(type(value) is int and value >= 0, 'root ordinal')
            else: authority.hash_value(value)
            values[key] = dict(path=src['path'],sha256=src['sha256'],pointer=lexical.pointer(ptr),value=value)
        need(all(k in values for k in ('root_id','root_sha256','goal_rgb_sha256')) and
             any(k in values for k in ('history_rgb_sha256','causal_history_rgb_sha256')), 'required declared identity')
        result.append(dict(row_pointer=lexical.pointer((container,i)),fields=values))
    return result


def join(src, historical, extracted):
    need(historical['source_id'] == src['id'] and historical['report'] == dict(path=src['path'],sha256=src['sha256']), 'original report join')
    old = historical['selected_rows']; need(len(old)==len(extracted), 'saved row count')
    for o, new in zip(old,extracted,strict=True):
        need(o['row_pointer'] == new['row_pointer'] and o['ordinal_is_not_episode_id'] is True and
             o['source_episode_identity'] is None and o['reset_attempt_identity'] is None and
             o['physical_state_identity'] is None and o['history_hash_unit'] == 'whole_stack_not_individual_frames' and
             o['missing'] == list(MISSING), 'identity semantics')
        fields = o['fields']; need(type(fields) is list, 'saved identity fields')
        got = {}
        for ref in fields:
            need(type(ref) is dict and set(ref)=={'path','sha256','pointer','value'}, 'identity reference shape')
            key = ref['pointer'].rsplit('/',1)[-1]
            need(key in FIELDS and key not in got, 'unique allowed identity')
            got[key] = ref
        need(got == new['fields'], 'independent raw-source identity pointer join')


def build():
    reader = authority.Reader()
    catalog = authority.parse(reader.read(str(authority.CATALOG),authority.CATALOG_SHA,authority.CAP))
    authority.requests(catalog)
    previous = authority.parse(reader.read(str(RECOVERY),RECOVERY_SHA,8*1024**2))
    old = previous['sources']; need(len(old)==8 and [x['source_id'] for x in old]==list('ABCDEFGH'), 'saved roster')
    sources = []
    for src, saved in zip(catalog['sources'],old,strict=True):
        raw = reader.read(src['path'],src['sha256'],4*1024**2)
        entries = extract(raw,src); join(src,saved,entries)
        sources.append(dict(source_id=src['id'],original_report=dict(path=src['path'],sha256=src['sha256']),rows=entries))
    for path, data in reader.cache.items():
        p = Path(path); need(p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(data).digest(), 'input drift')
    return dict(schema=NAME+'.report', status='original_declared_identity_crosschecked',
        recovery=dict(path=str(RECOVERY),sha256=RECOVERY_SHA), catalog=dict(path=str(authority.CATALOG),sha256=authority.CATALOG_SHA),
        sources=sources, source_count=8, selected_rows=sum(len(x['rows']) for x in sources),
        scope='selected rows only; source report bytes hashed, only identity fields decoded',
        outcome_values_decoded=False, original_load_attested=False, attempt_coverage_complete=False,
        episode_mapping_known=False, historical_scope_approved=False, freshness_certified=False,
        P4_admission=False, P5_admission=False, neural_calls=0, physics_calls=0)


def raw(value): return authority.canonical(value)+b'\n'


def write(path, value):
    p=Path(path); data=raw(value)
    need(p.is_absolute() and p==p.resolve() and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and
         not p.exists() and not p.is_symlink() and len(data)<=8*1024**2, 'exclusive bounded output')
    with p.open('xb') as f:
        need(f.write(data)==len(data), 'full write'); f.flush(); os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return hashlib.sha256(data).hexdigest()


def verify(path, sha):
    p=Path(path); authority.hash_value(sha)
    need(p.is_absolute() and p==p.resolve() and p.is_file() and not p.is_symlink() and p.stat().st_size<=8*1024**2, 'canonical result')
    data=p.read_bytes(); need(hashlib.sha256(data).hexdigest()==sha, 'result hash')
    need(data==raw(build()), 'full reconstruction from original source identities')
    need(hashlib.sha256(p.read_bytes()).hexdigest()==sha, 'stable result')
    return dict(verified=True, selected_rows=52, original_load_attested=False, freshness_certified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('extract'); p.add_argument('--output',required=True)
    p=subs.add_parser('verify'); p.add_argument('--report',required=True); p.add_argument('--report-sha256',required=True)
    a=parser.parse_args()
    if a.command=='extract':
        value=build(); result=dict(path=a.output,sha256=write(a.output,value),selected_rows=value['selected_rows'],freshness_certified=False)
    else: result=verify(a.report,a.report_sha256)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__': main()
