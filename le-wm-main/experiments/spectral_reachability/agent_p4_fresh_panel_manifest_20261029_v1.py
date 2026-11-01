"""Outcome-blind fixed-four panel handoff, not root collection or science admission.

Reuse frozen history projection/coverage and all-pairs root assessment. Extend
only the source-episode and post-cutoff exposure boundary. No model/env imports.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import moving_root_exclusion_manifest_20261027_v1 as exclusion
from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as resolution
from experiments.spectral_reachability import chat_freshroots_20261027_consumer_v2 as previous

NAME='agent_p4_fresh_panel_manifest_20261029_v1'
ROOT=exclusion.ROOT
REG=ROOT.parent/'docs/chat_freshroots_20261027_preregistration.json'
REG_SHA='79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e'
CAP=8*1024**2
FILE_CAP=128*1024**2
TOTAL_CAP=512*1024**2
FLOOR=200*1024**2
require=exclusion.require
canonical=exclusion.canonical
digest=exclusion.digest
FIELDS=('root_seed','reset_seed','root_goal_sha256','root_state','root_snapshot_sha256',
        'history_pixels_sha256','raw_history_sha256')
ALIASES={k:{k} for k in FIELDS}
ALIASES['root_goal_sha256'].add('root_sha256')
ALIASES['root_state'].add('state')
ALIASES['history_pixels_sha256'].add('history_frame_pixel_sha256')
ALIASES['reset_seed_span']={'reset_seed_span','span'}
SCOPE=('failed','unselected','reserved','restarted','all_seen_outcomes','all_source_episodes')


def decode(payload):
    require(len(payload)<=CAP,'metadata8MiB cap')
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate metadata key');out[k]=v
        return out
    value=json.loads(payload,object_pairs_hook=pairs);canonical(value);return value


def path_value(value):
    require(type(value) is str,'literal path')
    p=Path(value)
    require(p.is_absolute() and '..' not in p.parts and str(p)==value and p.is_relative_to(ROOT.parent)
        and not any(c in value for c in '*?[]'),'explicit project path, no glob')
    return p


def descriptor(value):
    exclusion.fields(value,'path sha256');path_value(value['path']);exclusion.hash_value(value['sha256'])
    return value


class Reader:
    """Explicit paths only; RAM provider is tests, never a production switch."""
    def __init__(self,provider=None):self.provider=provider;self.bindings={};self.total=0
    def read(self,desc,cap=CAP):
        descriptor(desc);path=path_value(desc['path'])
        if self.provider is None:
            require(path.resolve()==path and path.is_file() and not path.is_symlink()
                and path.stat().st_size<=cap,'canonical bounded regular file')
            with path.open('rb') as f:data=f.read(cap+1)
        else:data=self.provider(str(path))
        require(type(data) is bytes and len(data)<=cap and hashlib.sha256(data).hexdigest()==desc['sha256'],'external byte SHA/cap')
        if str(path) not in self.bindings:self.total+=len(data)
        require(self.total<=TOTAL_CAP,'512MiB unique read cap')
        require(str(path) not in self.bindings or self.bindings[str(path)]==desc['sha256'],'conflicting source SHA')
        self.bindings[str(path)]=desc['sha256'];return data
    def json(self,desc):return decode(self.read(desc))
    def check(self):
        for p,h in list(self.bindings.items()):self.read(dict(path=p,sha256=h),FILE_CAP)


def utc(value):
    require(type(value) is str,'literal UTC cutoff')
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.utcoffset() is not None and result.utcoffset().total_seconds()==0,'UTC cutoff timezone')
    return result


def fixed_candidates(reg):
    expected=[dict(slot=i,root_seed=203027101+i,reset_seed=203027201+i,witness_sign=1 if i%2==0 else -1) for i in range(4)]
    require(reg['candidate_order']==expected and reg['sampling_denominator']==4 and reg['proposal_per_slot']==1
        and reg['attempt_per_slot']==1 and reg['replacement_or_retry'] is False,'original fixed four, one attempt, no replacement')
    return expected


def proof_value(reader,proof,field,expected):
    exclusion.fields(proof,'file pointer');descriptor(proof['file'])
    pointer=proof['pointer'];require(type(pointer) is str and pointer.startswith('/'),'literal JSON pointer')
    decoded=[x.replace('~1','/').replace('~0','~') for x in pointer[1:].split('/')]
    parts=tuple(int(x) if x.isdecimal() else x for x in decoded)
    allowed=ALIASES.get(field,{'episode_id'})
    require(parts[-1] in allowed and len(parts)<=32,'identity-only pointer leaf; no outcome/cost/difficulty')
    # Only the named identity is decoded; all native outcomes are lexically skipped.
    raw=reader.read(proof['file'],FILE_CAP)
    p=resolution.project(BytesIO(raw),[parts,parts+('*',)])
    value=p.leaves.get(parts) if parts in p.leaves else exclusion.values(p,parts)
    require(canonical(value)==canonical(expected),'source-bound actual identity at '+pointer)


def normalize(record,reader):
    exclusion.fields(record,'record_id root_seed reset_seed root_goal_sha256 root_state root_snapshot_sha256 history_pixels_sha256 raw_history_sha256 source_episode evidence')
    require(type(record['record_id']) is str and 0<len(record['record_id'])<=200,'record identity')
    require(type(record['evidence']) is dict and set(record['evidence'])=={k for k in FIELDS if record[k] is not None},'proof for every non-null identity, no invented metadata')
    out=deepcopy(record);missing=[]
    for key in FIELDS:
        value=record[key]
        if value is None:missing.append(key);continue
        if key in ('root_seed','reset_seed'):exclusion.integer(value)
        elif key=='root_state':
            require(type(value) is list and len(value)==7 and all(type(x) in (int,float) for x in value),'literal finite state7')
            out.update(exclusion.state_identity(value))
        elif key=='history_pixels_sha256':
            require(type(value) is list and len(value)==3,'complete RGB3 identity')
            for h in value:exclusion.hash_value(h)
        else:exclusion.hash_value(value)
        proof_value(reader,record['evidence'][key],key,value)
    episode=record['source_episode']
    if episode is None:missing.append('source_episode')
    else:
        exclusion.fields(episode,'source episode_id pointer');descriptor(episode['source'])
        require(type(episode['episode_id']) is str and 0<len(episode['episode_id'])<=200,'original source episode_id')
        proof_value(reader,dict(file=episode['source'],pointer=episode['pointer']),'episode_id',episode['episode_id'])
    out['missing']=missing;return out


def episode_equal(a,b):
    if a is None or b is None:return False
    return a['episode_id']==b['episode_id'] and (a['source']['path']==b['source']['path'] or a['source']['sha256']==b['source']['sha256'])


def assess(candidates,actuals,history,seeds,spans,root_goals):
    """All pairs; conflicts retain BOTH slots. Never choose a surviving subset."""
    require(len(candidates)==len(actuals)==4,'all four fixed slots')
    # Original assessor owns pose/hash/RGB3 alias thresholds; no new thresholds.
    complete_history=[r for r in history if r.get('root_state') is not None]
    inputs=[dict(slot=i,**({k:r.get(k) for k in previous.ACTUAL_FIELDS if k!='slot'} if r is not None else dict(root_state=None))) for i,r in enumerate(actuals)]
    rows=previous.assess(complete_history,inputs,exclusion.state_identity,exclusion.same_pose)
    for i,(row,candidate,actual) in enumerate(zip(rows,candidates,actuals,strict=True)):
        hits=row['duplicate_evidence'];missing=['actual_identity_not_supplied'] if actual is None else list(actual['missing'])
        for field in ('root_seed','reset_seed'):
            seed=candidate[field]
            if seed in seeds:hits.append(dict(kind='seed',field=field,seed=seed,reason='used_or_reserved_cross_namespace'))
            for lo,hi in spans:
                if lo<=seed<hi:hits.append(dict(kind='span',field=field,span=[lo,hi]))
        if actual is not None:
            require(all(actual[k]==candidate[k] for k in ('root_seed','reset_seed')),'actual slot is original preregistered candidate')
            if actual['root_goal_sha256'] in root_goals:hits.append(dict(kind='historical_root_goal',sha256=actual['root_goal_sha256']))
            # Keep hash-only old records too: absence of state never means unused.
            for j,old in enumerate(history):
                reasons=[k for k in ('root_goal_sha256','root_state_sha256','root_snapshot_sha256','history_pixels_sha256') if actual.get(k) is not None and old.get(k)==actual[k]]
                if reasons:hits.append(dict(kind='historical_metadata',index=j,record_id=old.get('record_id',old.get('record_ref')),reasons=reasons))
                if episode_equal(actual['source_episode'],old.get('source_episode')):hits.append(dict(kind='historical_source_episode',index=j))
            for j,other in enumerate(actuals):
                if j!=i and other is not None and episode_equal(actual['source_episode'],other['source_episode']):hits.append(dict(kind='batch_source_episode',slot=j))
        row.update(candidate=candidate,actual_root_identity=actual,missing=sorted(set(missing+row['missing'])),
            status='excluded_duplicate' if hits else 'blocked_missing_actual_identity' if missing else 'metadata_disjoint_pending_external_authority',
            metadata_disjoint=not hits and not missing)
    return rows


def template():
    """Actual existing references, absent current inputs remain null, never SHA stubs."""
    def pin(rel,h):return dict(path=str(ROOT/rel),sha256=h)
    return dict(schema=NAME+'.request',preregistration=dict(path=str(REG),sha256=REG_SHA),
        legacy_manifest=pin('results/chat_freshroots_20261027_root_manifest.json','3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838'),
        legacy_consumer=pin('results/chat_freshroots_20261027_consumer_manifest_v2.json','2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7'),
        history_request=None,exposure_inventory=None,actual_roots=None,panel_approval=None)


def required_coverage_scope(request):
    # Existing coverage_scope is extensible; do not invent a second history signer.
    return ['panel_legacy_manifest_sha256:'+request['legacy_manifest']['sha256'],
        'panel_legacy_consumer_sha256:'+request['legacy_consumer']['sha256'],
        *['panel_scope:'+x for x in SCOPE]]


def build(request,reader=None,request_file=None):
    reader=Reader() if reader is None else reader
    if request_file is not None:
        require(canonical(reader.json(request_file))==canonical(request),'entire external request bytes/semantic binding')
    exclusion.fields(request,'schema preregistration legacy_manifest legacy_consumer history_request exposure_inventory actual_roots panel_approval')
    require(request['schema']==NAME+'.request','panel request schema')
    require(request['preregistration']==dict(path=str(REG),sha256=REG_SHA),'frozen preregistration authority')
    for key in ('legacy_manifest','legacy_consumer'):
        require(request[key]==template()[key],'sealed legacy exclusions cannot be replaced by a same-schema empty index')
    reg=reader.json(request['preregistration']);candidates=fixed_candidates(reg)
    base=reader.json(request['legacy_manifest']);prior=reader.json(request['legacy_consumer'])
    require(base['schema']=='chat_freshroots_20261027.root_manifest.v1' and prior['schema']=='chat_freshroots_20261027.consumer_manifest.v2'
        and prior['sampling_denominator']==4 and prior['candidates']==base['candidates'],'existing fixed-slot metadata authority, no substituted panel')
    require(any(x['path']==request['legacy_manifest']['path'] and x['sha256']==request['legacy_manifest']['sha256'] for x in prior['bindings']),'consumer binds entire original manifest')
    history=deepcopy(prior['historical_exclusion_index']);blockers=list(base['unresolved_dependencies'])
    seeds={x['seed'] for x in base['used_or_reserved_seed_evidence']}
    spans=[x['half_open'] for x in base['half_open_exclusion_spans']]
    goals={x['root_goal_sha256'] for x in base['root_goal_identity_index']}
    authority=None
    if request['history_request'] is not None:
        q=reader.json(request['history_request'])
        require(q['protocol']==dict(path=str(exclusion.PROTOCOL),sha256=reg['protocol_sha256']),'same preregistered exclusion protocol')
        descriptors=[q['protocol'],*q['evidence_files'],*([q['approval_file']] if q['approval_file'] else [])]
        by_path={}
        for e in descriptors:
            require(e['path'] not in by_path or by_path[e['path']]==e['sha256'],'conflicting history descriptor SHA')
            by_path[e['path']]=e['sha256']
        authority=exclusion.build(q,read_bytes=lambda p:reader.read(dict(path=p,sha256=by_path[p]),exclusion.FILE_CAP))
        history+=authority['observed_roots'];seeds.update(authority['used_or_reserved_root_seeds']+authority['used_or_reserved_reset_seeds'])
        spans+=authority['reserved_reset_spans'];goals.update(authority['root_goal_sha256s'])
        if authority['source_run_preconditions_met'] and set(required_coverage_scope(request))<=set(authority['approval']['coverage_scope']):
            # Exact original MAIN coverage approval is the only historical disposition.
            blockers=[]
        else:blockers.append('existing_history_validator_or_explicit_legacy_disposition_blocked:'+str(authority['unresolved_sources']))
    else:blockers.append('current_MAIN_approved_history_request_not_supplied')
    exposure=None
    if request['exposure_inventory'] is not None:
        exposure=reader.json(request['exposure_inventory'])
        exclusion.fields(exposure,'schema cutoff_utc project_root includes complete unknowns records reserved_reset_spans')
        require(exposure['schema']==exclusion.NAME+'.identity_metadata' and exposure['project_root']==str(ROOT),'existing identity_metadata extension, exact project')
        utc(exposure['cutoff_utc']);require(type(exposure['complete']) is bool and type(exposure['unknowns']) is list,'explicit exposure scope')
        require(exposure['includes']==list(SCOPE),'all seen/failed/unselected/reserved/restart/source episodes scope')
        require(type(exposure['records']) is list and len(exposure['records'])<=4096,'bounded complete exposure inventory')
        seen=[normalize(r,reader) for r in exposure['records']]
        require(len({r['record_id'] for r in seen})==len(seen),'unique exposure records')
        history+=seen
        require(type(exposure['reserved_reset_spans']) is list and len(exposure['reserved_reset_spans'])<=4096,'bounded reserved spans')
        for span in exposure['reserved_reset_spans']:
            exclusion.fields(span,'span evidence')
            require(type(span['span']) is list and len(span['span'])==2,'half-open reset span')
            lo,hi=span['span'];exclusion.integer(lo);exclusion.integer(hi);require(lo<hi,'positive half-open span')
            proof_value(reader,span['evidence'],'reset_seed_span',span['span']);spans.append([lo,hi])
        for r in seen:
            seeds.update(r[k] for k in ('root_seed','reset_seed') if r[k] is not None)
            if r['root_goal_sha256'] is not None:goals.add(r['root_goal_sha256'])
            if r['missing']:blockers.append('seen_identity_incomplete:'+r['record_id']+':'+','.join(r['missing']))
        if exposure['complete'] is not True or exposure['unknowns']:blockers.append('exposure_coverage_unknown:'+str(exposure['unknowns']))
        if authority is None or not any(e['path']==request['exposure_inventory']['path'] and e['sha256']==request['exposure_inventory']['sha256'] and e['projected'] for e in authority['evidence_files']):blockers.append('exposure_inventory_not_bound_by_original_history_approval')
        if authority is not None and exposure['cutoff_utc']!=authority['cutoff_utc']:blockers.append('history_exposure_cutoff_mismatch')
    else:blockers.append('all_seen_root_and_source_episode_inventory_missing')
    actuals=[None]*4
    if request['actual_roots'] is not None:
        payload=reader.json(request['actual_roots']);exclusion.fields(payload,'schema slots')
        require(payload['schema']==NAME+'.actual_roots' and type(payload['slots']) is list and len(payload['slots'])==4,'full four actual identities or null')
        actuals=[None if r is None else normalize(r,reader) for r in payload['slots']]
        require(len({r['record_id'] for r in actuals if r is not None})==sum(r is not None for r in actuals),'unique actual record ids')
    rows=assess(candidates,actuals,history,seeds,spans,goals)
    if any(not r['metadata_disjoint'] for r in rows):blockers.append('one_or_more_fixed_slots_missing_or_excluded_no_replacement')
    content=dict(schema=NAME+'.panel_content',request={k:v for k,v in request.items() if k!='panel_approval'},
        fixed_denominator=4,candidates=candidates,rows=rows,historical_records=len(history),
        history_content_sha256=None if authority is None else authority['coverage_content_sha256'],
        history_approval=None if authority is None else authority['approval'],
        exposure_cutoff_utc=None if exposure is None else exposure['cutoff_utc'],
        used_or_reserved_seeds=sorted(seeds),reserved_reset_spans=sorted(set(tuple(s) for s in spans)),
        unresolved=sorted(set(blockers)),source_sha256=own_hashes(),
        selection='original slot order, no outcome/cost/difficulty, no replacement, one attempt',
        certification_scope='source-bound metadata disjointness only; NOT actual physical freshness or disclosure proof')
    content_hash=digest(content);accepted=False;approval_error=None
    if request['panel_approval'] is not None:
        a=reader.json(request['panel_approval'])
        exclusion.fields(a,'schema authority approved panel_content_sha256 cutoff_utc outcome_blind_selection')
        require(a['schema']==NAME+'.MAIN_panel_approval' and a['authority']=='MAIN' and a['approved'] is True
            and a['panel_content_sha256']==content_hash and a['cutoff_utc']==content['exposure_cutoff_utc']
            and a['outcome_blind_selection'] is True,'external MAIN exact panel metadata attestation, no future hash')
        accepted=not content['unresolved']
        if not accepted:approval_error='approval_cannot_erase_unresolved_or_missing_or_duplicate'
    reader.check()
    return dict(schema=NAME+'.manifest',request=request,request_file=request_file,content=content,panel_content_sha256=content_hash,
        bindings=reader.bindings,metadata_panel_certified=accepted,
        status='metadata_panel_accepted_science_still_closed' if accepted else 'blocked_planning_artifact',
        approval_error=approval_error,certified_fresh_roots=[],freshness_certified=False,source_run_authorized=False,
        scientific_admission=False,P1_admission=False,P4_admission=False,new_root_evidence=False,
        actual_neural_calls=0,actual_physics_calls=0)


def own_hashes():
    files=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        Path(exclusion.__file__).resolve(),Path(exclusion.lexical.__file__).resolve(),Path(resolution.__file__).resolve(),Path(previous.__file__).resolve()]
    return {str(p):exclusion.file_sha(p) for p in files}


def write_new(path,value):
    p=path_value(str(path));raw=canonical(value)
    require(p.name.startswith(NAME+'_') and p.parent.resolve()==p.parent and not p.exists() and not p.is_symlink()
        and len(raw)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(raw),'exclusive bounded own output, 200MiB floor')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'full metadata write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=exclusion.file_sha(p),bytes=len(raw),status=value.get('status','request_draft'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('draft','build','verify'))
    for key in ('request','request-sha','manifest','manifest-sha','output'):parser.add_argument('--'+key)
    a=parser.parse_args(argv);reader=Reader()
    if a.command=='draft':value=template()
    elif a.command=='build':
        desc=dict(path=a.request,sha256=a.request_sha)
        value=build(reader.json(desc),reader,request_file=desc)
    else:
        saved=reader.json(dict(path=a.manifest,sha256=a.manifest_sha))
        now=build(saved['request'],request_file=saved['request_file'])
        require(canonical(now)==canonical(saved),'entire manifest saved reconstruction')
        value=dict(status='saved_metadata_verified_NOT_fresh',metadata_panel_certified=now['metadata_panel_certified'],scientific_admission=False)
    print(canonical(write_new(a.output,value) if a.output else value).decode(),end='')


if __name__=='__main__':main()
