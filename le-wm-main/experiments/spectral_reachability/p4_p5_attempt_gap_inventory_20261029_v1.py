"""Bounded A-H direct-directory inventory and pinned reset evidence gaps.

Names only in original output directories; no file discovery outside those
directories, no outcome decoding, and no attestation of historical absence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_freshroot_authority_20261028_v1 as a

ROOT=Path(__file__).resolve().parents[2]
NAME='p4_p5_attempt_gap_inventory_20261029_v1'
RECOVERY=ROOT/'results/agent_p4_historical_source_identity_20261029_v1_LOCAL01.json'
RECOVERY_SHA='577580e536598d850cd449fda2a02c314197267bdc41bff9d85692432165e724'
CROSS=ROOT/'results/p4_p5_source_identity_crosscheck_20261029_v1_MAIN01.json'
CROSS_SHA='f0543af7a653d86aa99ec23703646af74337cb7bd69f9ea1869b0a724fb644a9'
RESOLUTION=ROOT/'results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json'
RESOLUTION_SHA='032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00'
need=a.need


def inventory(directory):
    p=Path(directory)
    need(p.is_absolute() and p==p.resolve() and p.is_dir() and not p.is_symlink(), 'canonical original directory')
    rows=[]
    with os.scandir(p) as it:
        for x in it:
            need(not x.is_symlink(), 'original direct entry symlink')
            kind='file' if x.is_file(follow_symlinks=False) else 'directory' if x.is_dir(follow_symlinks=False) else 'other'
            rows.append(dict(name=x.name,kind=kind))
            need(len(rows)<=1000, 'bounded direct directory entries')
    return sorted(rows,key=lambda x:x['name'])


def build():
    reader=a.Reader()
    catalog=a.parse(reader.read(str(a.CATALOG),a.CATALOG_SHA,a.CAP)); a.requests(catalog)
    hist=a.parse(reader.read(str(RECOVERY),RECOVERY_SHA,8*1024**2))
    cross=a.parse(reader.read(str(CROSS),CROSS_SHA,8*1024**2))
    resolution=a.parse(reader.read(str(RESOLUTION),RESOLUTION_SHA,32*1024**2))
    need(cross['source_count']==8 and cross['selected_rows']==52 and cross['freshness_certified'] is False, 'fixed identity crosscheck')
    old=hist['sources']; seen=cross['sources']; dispatch=hist['dispatch_load_closure']
    need(len(old)==len(seen)==8 and len(dispatch)==10, 'fixed historical denominators')
    calls={d['call_id']:d for d in catalog['recovery']['dispatches']}
    need(len(calls)==10 and len({x['call_id'] for x in dispatch})==10, 'unique original calls')
    rows=[]; dirs=[]
    for source,saved,checked in zip(catalog['sources'],old,seen,strict=True):
        sid=source['id']; ref=dict(path=source['path'],sha256=source['sha256'])
        need(saved['source_id']==checked['source_id']==sid and saved['report']==checked['original_report']==ref and
             len(saved['selected_rows'])==len(checked['rows']), 'source and identity join')
        directory=Path(source['path']).parent; contents=inventory(directory);dirs.append((directory,contents))
        need(any(x==dict(name='report.json',kind='file') for x in contents), 'original report present')
        relevant=[]
        for d in dispatch:
            original=calls.get(d['call_id']); need(original is not None, 'known dispatch only')
            if d['source_id']!=sid:continue
            need(original['source_id']==sid and d['attempts_including_rejected_failed_restarts'] is None and
                 d['original_load_closure_complete'] is False and all(x['original_load_attested'] is False and
                 x['original_loaded_path'] is None and x['original_loaded_sha256'] is None for x in d['roles']), 'unattested run evidence')
            relevant.append(dict(call_id=d['call_id'],catalog_pointer=d['catalog_pointer'],
                original_dispatch=d['dispatch_intent'],original_ledger='not_supplied_in_pinned_recovery',
                original_load='not_attested_in_pinned_recovery',
                role_gaps=[dict(role=x['role'],candidate_bytes_available=x['candidate_bytes_available'],
                                original_loaded_path=None,original_loaded_sha256=None) for x in d['roles']]))
        need(relevant, 'source dispatch')
        rows.append(dict(source_id=sid,report=ref,selected_rows=len(checked['rows']),directory=str(directory),
            direct_entries=contents,discovery_scope='direct entry names and types only; file contents and nested directories not scanned',
            selected_mapping='no_episode_or_attempt_mapping_in_pinned_recovery',calls=relevant))
    bridges=resolution['physical_identity_bridges'];need(len(bridges)==4, 'fixed prior known-root bridge roster')
    known=[]
    for i,b in enumerate(bridges):
        meta=b['metadata']; journal=b['evidence']['journal']
        known.append(dict(resolution_pointer='/physical_identity_bridges/'+str(i),record_id=meta['record_id'],
            maps_to=meta['maps_to'],root_seed=meta['root_seed'],reset_seed=meta['reset_seed'],
            root_state_sha256=meta['root_state_sha256'],root_snapshot_sha256=meta['root_snapshot_sha256'],
            source_journal=journal,scope='prior saved bridge; not a mapping to A-H attempt ledger'))
    for p,prev in dirs: need(inventory(p)==prev, 'direct directory changed during inventory')
    for path,data in reader.cache.items():
        p=Path(path);need(p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(data).digest(), 'pinned input changed')
    return dict(schema=NAME+'.inventory',status='bounded_sources_evidence_gaps',sources=rows,known_root_bridges=known,
        pinned_inputs=[dict(path=str(p),sha256=h) for p,h in ((a.CATALOG,a.CATALOG_SHA),(RECOVERY,RECOVERY_SHA),(CROSS,CROSS_SHA),(RESOLUTION,RESOLUTION_SHA))],
        source_count=8,dispatch_count=10,selected_rows=52,bridge_records=4,
        external_requirements=['original attempt ledger including failures, rejected, unselected, replay and restart OR attested original loaded closure plus exhaustive envelope',
            'selected row to original episode/reset identity and observed state/history',
            'bounded relevant historical scope and post-cutoff delta with external MAIN approval',
            'P5 prospective confirmation protocol and authorized actual candidate identities'],
        scope='eight source output directories direct entries plus four pinned metadata files; no recursive or global scan',
        absence_claim='no proof that original ledger or other assets never existed; file name inventory is not semantic classification',
        original_attempts=None,original_load_attested=False,coverage_approved=False,freshness_certified=False,
        P4_admission=False,P5_admission=False,outcome_values_decoded=False,neural_calls=0,physics_calls=0)


def raw(v):return a.canonical(v)+b'\n'


def write(path,v):
    p=Path(path);data=raw(v)
    need(p.is_absolute() and p==p.resolve() and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and
         not p.exists() and not p.is_symlink() and len(data)<=8*1024**2,'exclusive output')
    with p.open('xb') as f:need(f.write(data)==len(data),'full output');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return hashlib.sha256(data).hexdigest()


def verify(path,sha):
    p=Path(path);a.hash_value(sha)
    need(p.is_absolute() and p==p.resolve() and p.is_file() and not p.is_symlink() and p.stat().st_size<=8*1024**2,'result path')
    data=p.read_bytes();need(hashlib.sha256(data).hexdigest()==sha and data==raw(build()),'result hash and exact fresh reconstruction')
    need(hashlib.sha256(p.read_bytes()).hexdigest()==sha,'stable result')
    return dict(verified=True,freshness_certified=False,original_attempts=None)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    x=s.add_parser('build');x.add_argument('--output',required=True)
    x=s.add_parser('verify');x.add_argument('--report',required=True);x.add_argument('--report-sha256',required=True)
    q=p.parse_args()
    if q.command=='build':
        v=build();result=dict(path=q.output,sha256=write(q.output,v),freshness_certified=False)
    else:result=verify(q.report,q.report_sha256)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
