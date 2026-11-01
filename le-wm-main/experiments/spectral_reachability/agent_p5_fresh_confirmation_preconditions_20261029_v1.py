"""Pinned saved-only P5 freshness readiness, not a new freshness authority."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

NAME='agent_p5_fresh_confirmation_preconditions_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CAP=4*1024**2
FLOOR=200*1024**2
AUTHORITY=ROOT/'experiments/spectral_reachability/agent_freshroot_authority_20261028_v1.py'
PINS={
    'authority_source':(AUTHORITY,'f7a30fcad18483e1848170acdd1fe4bd96e7d08b38804b18f1cf2cf1087870e7'),
    'protocol':(ROOT.parent/'docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md','42f1bcbf517dfd46b8ba05525663d82efac32dbb5fb24f87a8ca572bef507a2f'),
    'preregistration':(ROOT.parent/'docs/chat_freshroots_20261027_preregistration.json','79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e'),
    'fixed_slot_consumer':(ROOT/'results/chat_freshroots_20261027_consumer_manifest_v2.json','2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7'),
    'exclusions':(ROOT/'results/moving_root_exclusion_manifest_20261027_v1_inventory8_main01.json','a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060'),
    'reset_catalog':(ROOT/'results/agent_freshroots_resume_20261027_coverage.json','00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb'),
}
SEEDS=[dict(slot=i,root_seed=203027101+i,reset_seed=203027201+i,witness_sign=1 if i%2==0 else -1) for i in range(4)]


def need(ok,message):
    if not ok:raise ValueError(message)


def raw(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reuse():
    # Authenticate the existing minimal authority implementation BEFORE import.
    p,h=PINS['authority_source']
    need(p==AUTHORITY and not any(x.is_symlink() for x in (p,*p.parents)) and sha(p)==h,'pinned existing authority code')
    from experiments.spectral_reachability import agent_freshroot_authority_20261028_v1 as authority
    need(Path(authority.__file__).resolve()==AUTHORITY,'no substitute authority module')
    return authority


def resolve(reader=None):
    a=reuse();reader=a.Reader() if reader is None else reader
    saved={role:reader.read(str(p),h,CAP) for role,(p,h) in PINS.items()}
    reg=a.parse(saved['preregistration']);c=a.parse(saved['fixed_slot_consumer']);e=a.parse(saved['exclusions'])
    request=a.requests(a.parse(saved['reset_catalog']))
    need(reg['schema']=='chat_freshroots_20261027.preregistration.v1' and
        reg['protocol_sha256']==PINS['protocol'][1] and reg['candidate_order']==SEEDS and
        reg['sampling_denominator']==4 and reg['attempt_per_slot']==reg['proposal_per_slot']==1 and
        reg['replacement_or_retry'] is False and reg['new_outcomes_opened'] is False,'original fixed seed/order/no-outcome contract')
    need(c['schema']=='chat_freshroots_20261027.consumer_manifest.v2' and c['sampling_denominator']==4 and
        [{k:x[k] for k in SEEDS[0]} for x in c['candidates']]==SEEDS and
        c['coverage_certified'] is c['freshness_certified'] is c['source_run_authorized'] is False and
        c['certified_fresh_roots']==[] and all(x['actual_root_identity'] is None and x['freshness_certified'] is False for x in c['candidates']),
        'pinned blocked snapshot only; changed/claimed-ready cohort needs existing authority review, not this resolver')
    need(e['schema']=='moving_fresh_source_exclusions.v1' and e['source_run_preconditions_met'] is False and
        e['freshness_certified'] is False and e['approval']['approved'] is False and e['unresolved_sources'] and
        e['protocol']==dict(path=str(PINS['protocol'][0]),sha256=PINS['protocol'][1]),'existing unresolved exclusion snapshot')
    upstream=[]
    for source in request['sources']:
        upstream.append(dict(component='original_reset_closure',status='blocked',source_id=source['source_id'],
            original_report=source['report'],inherited_refs=source['inherited_refs'],dispatches=source['dispatches'],
            exact_required_fields=['call_id','reset_evidence','origin_citation','attempts_complete','restarts_complete'],
            accepted_routes=source['required_routes'],
            local_reuse='agent_freshroot_authority_20261028_v1 request / validate',
            owner='original run custodian; MAIN provenance review',
            cannot_recover_from='current source, final selected count, known-root replay, absent search hits'))
    upstream.extend([
        dict(component='historical_scope_delta_and_identity',status='blocked',
            required_new_file_role='MAIN exact path+SHA inventory delta since old cutoff, including all failed/unselected/reserved sources; source-bound state7/pose/snapshot/RGB3/raw10 bridges and original reset scope',
            reuse='moving_root_exclusion_manifest_20261027_v1; chat_freshroots_20261027_consumer_v2.assess',
            old_content_sha256=e['coverage_content_sha256'],
            known_unresolved_count=len(e['unresolved_sources']),old_unresolved=list(c['unresolved_dependencies']),
            reason='old metadata census is not coverage; no implicit follow or deletion of unresolved refs'),
        dict(component='external_scope_approval',status='blocked',
            required_new_file_role='MAIN_coverage_approval JSON for newly complete exclusion content, external path+SHA',
            exact_required_fields=['schema','authority','approved','coverage_scope','cutoff_utc','manifest_content_sha256'],
            reason='approval must bind complete current content; self-signed resolver output cannot approve; reserved spans must be excluded without outcomes'),
        dict(component='P5_prospective_protocol',status='blocked',
            required_new_file_role='MAIN pre-outcome P5 protocol path+SHA: fixed cohort N/order/seeds/distribution, root/source/episode exclusions, frozen model/task/observation/action/history/label ABI, all arms/counters/caps, failure and stopping rules',
            reason='old four-slot protocol is LeWM CLS192 moving source design, not a JEPA/factorial P5 confirmation contract; no automatic transfer of four seeds or scientific gates'),
        dict(component='actual_candidate_identity',status='blocked',
            required_new_file_role='future authorized source prepare plus full attempt/reset/load ledger and per-slot state7/pose/snapshot/RGB3/raw10 identity; historical and intra-batch exclusion receipt before witness/outcome dispatch',
            reason='no actual roots supplied for old four slots; distinct seeds, 164 branches and two known roots are not independent samples; no collection authorized here'),
    ])
    for path,b in reader.cache.items():
        if reader.provider is None:need(hashlib.sha256(b).hexdigest()==sha(path),'final unchanged pinned input bytes')
    return dict(schema=NAME+'.resolution',status='blocked_no_certified_fresh_cohort',
        inputs={role:dict(path=str(p),sha256=h) for role,(p,h) in PINS.items()},
        reusable_components=[dict(component='original_contract',status='ready',scope='fixed metadata contract only'),
            dict(component='reset_request_schema',status='ready',scope='8 sources/16 refs/10 known calls; not proof these exhaust history'),
            dict(component='fixed_slot_identity_consumer',status='ready',scope='historical/intra-batch checks; cannot certify missing coverage')],
        old_design_slots=SEEDS,old_design_denominator=4,P5_confirmation_denominator=None,
        certified_fresh_roots=[],certified_fresh_count=0,upstream_requests=upstream,
        request_semantic_sha256=a.sha(a.canonical(request)),
        source_run_authorized=False,reserved_outcomes_opened=False,neural_calls=0,physics_calls=0,
        P1_admission=False,P5_admission=False,
        limitations=['pinned asset set only, not proof no other project assets exist',
            'no scan or automatic source/receipt follow; no original load bytes inferred',
            'ready component is reusable code/metadata only, never a fresh cohort',
            'new authority or cohort requires existing resolver/MAIN review, not a flag here'])


def write_new(path,v):
    p=Path(path).absolute();b=raw(v)
    need(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and
        not any(x.is_symlink() for x in (p,*p.parents)) and len(b)<=CAP and
        shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive own bounded output/floor')
    with p.open('xb') as f:need(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(status=v['status'],path=str(p),sha256=sha(p))


def verify(path,expected):
    a=reuse();v=a.parse(a.Reader().read(str(Path(path).absolute()),expected,CAP))
    need(raw(v)==raw(resolve()),'whole fixed preconditions resolution reproduced')
    return dict(status='blocked_preconditions_reproduced',resolution_sha256=expected,certified_fresh_count=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('resolve');q.add_argument('--output')
    q=s.add_parser('verify');q.add_argument('--resolution',required=True);q.add_argument('--resolution-sha',required=True)
    q=p.parse_args()
    if q.command=='resolve':
        v=resolve();v=write_new(q.output,v) if q.output else v
    else:v=verify(q.resolution,q.resolution_sha)
    print(raw(v).decode(),end='')


if __name__=='__main__':main()
