"""Merged historical exclusion index and fail-closed fixed-slot consumer.

No producer/model imports; only pinned metadata. This version cannot certify
historical coverage or authorize a source run, even with apparently new poses.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/chat_freshroots_20261027_root_manifest.json'
PREFIX=ROOT/'results/chat_freshroots_20261027_prefix_identity_supplement.json'
SCOPE=ROOT.parent/'docs/chat_freshroots_20261027_parent_scope.json'
PINS={
    str(BASE):'3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838',
    str(PREFIX):'2cd81f5741fbef9fea0eae62b913e9ad73b09fa8c9d02e525876d6854a91d7e9',
    str(SCOPE):'beb4231eb06fcf33de151de4bbb853f5900d1be2621b72a688ecf04e230dcf61',
    str(ROOT/'experiments/spectral_reachability/chat_freshroots_20261027_closure.py'):
        'd408e2dbab9a56d1daa1024910895eeb485dabab35ed2dac95ead328cac37b4a',
}


def require(ok,reason):
    if not ok:raise ValueError(reason)


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()


def digest(raw):return hashlib.sha256(raw).hexdigest()


def bound_read(path,sha):
    p=Path(path)
    require(p.is_absolute() and p.resolve()==p and p.is_relative_to(ROOT.parent),'literal project file')
    require(p.stat().st_size<=8*1024**2,'file cap')
    with p.open('rb') as stream:raw=stream.read(8*1024**2+1)
    require(len(raw)<=8*1024**2 and digest(raw)==sha,'file SHA/cap mismatch')
    return raw


def parent_prefixes(raw,adapter,project,identity):
    require(adapter in ('branch_prefix','probe_prefix'),'known adapter')
    prefix=('rows','*','branches','*')
    if adapter=='probe_prefix':prefix+=('probes','*')
    patterns=[('rows','*','root_sha256'),('rows','*','branches','*','candidate_id'),
              prefix+('prefix_state','*'),prefix+('prefix_rgb_sha256',)]
    from io import BytesIO
    p=project(BytesIO(raw),patterns)
    result=[]
    for path,value in p.leaves.items():
        if path[-2:]!=('prefix_state',0):continue
        container=path[:-2]
        state=[p.leaves.get(container+('prefix_state',i)) for i in range(7)]
        require({k[-1] for k in p.leaves if k[:-1]==container+('prefix_state',)}==set(range(7)),
                'exact parent prefix state7')
        require(all(type(x) in (int,float) for x in state),'literal state7')
        meta=identity(state)
        root=p.leaves.get(('rows',path[1],'root_sha256'))
        candidate=p.leaves.get(('rows',path[1],'branches',path[3],'candidate_id'))
        rgb=p.leaves.get(container+('prefix_rgb_sha256',))
        for h in (root,rgb):require(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'parent identity SHA')
        require(type(candidate) is str and bool(candidate),'explicit branch ID')
        result.append(dict(state_sha256=meta['root_state_sha256'],prefix_rgb_sha256=rgb,
            ancestor_root_goal_sha256=root,candidate_id=candidate,
            pointer='/'+'/'.join(map(str,container)),relation='explicit_report_hierarchy_not_physical_root_equivalence'))
    return result


def join_parents(records,parents):
    """All parents retained; no name/index based matching or winner selection."""
    grouped={}
    for parent in parents:
        key=(parent['state_sha256'],parent['prefix_rgb_sha256'],
             parent['ancestor_root_goal_sha256'],parent['candidate_id'])
        grouped.setdefault(key,[]).append(parent)
    groups=[]; lookup={}
    for key,proofs in sorted(grouped.items()):
        index=len(groups)
        groups.append(dict(group_id=index,state_sha256=key[0],prefix_rgb_sha256=key[1],
            ancestor_root_goal_sha256=key[2],candidate_id=key[3],evidence=proofs,
            physical_root_equivalence_proved=False))
        lookup.setdefault(key[:2],[]).append(index)
    mappings=[]
    for i,record in enumerate(records):
        hits=lookup.get((record['root_state_sha256'],record['prefix_rgb_sha256']),[])
        mappings.append(dict(record_index=i,parent_group_ids=hits,
            status='explicit_ancestor_branch_associated' if len(hits)==1 else
                   'ambiguous_ancestor_mapping' if hits else 'parent_mapping_missing',
            original_sampling_state_equivalence_proved=False))
    return groups,mappings


ACTUAL_FIELDS={'slot','root_state','root_goal_sha256','root_snapshot_sha256',
               'history_pixels_sha256','raw_history_sha256'}


def assess(index,actuals,identity,same_pose):
    """Fixed four slots; reject duplicates without looking at task outcomes.

    History and batch collision checks are all-pairs and retain both colliding
    batch slots. Hash/pose difference is never a freshness certification.
    """
    require(type(actuals) is list and len(actuals)==4,'fixed four actual slots required')
    require([x.get('slot') for x in actuals]==list(range(4)) and
            all(type(x['slot']) is int for x in actuals),'fixed slot order')
    normalized=[]
    for actual in actuals:
        require(set(actual)<=ACTUAL_FIELDS,'outcome/unrecognized input field forbidden')
        state=actual.get('root_state')
        if state is None:normalized.append(None);continue
        require(type(state) is list and all(type(x) in (int,float) for x in state),'literal state')
        meta=identity(state)
        for key in ('root_goal_sha256','root_snapshot_sha256','raw_history_sha256'):
            value=actual.get(key)
            require(value is None or (type(value) is str and len(value)==64 and
                    all(c in '0123456789abcdef' for c in value)),'actual SHA')
        pixels=actual.get('history_pixels_sha256')
        require(pixels is None or (type(pixels) is list and len(pixels)==3 and all(
                type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in pixels)),
                'actual RGB3 SHA')
        normalized.append(dict(**actual,**meta))
    rows=[]
    for i,actual in enumerate(normalized):
        hits=[]; missing=[]
        if actual is None:missing.append('actual_root_state_not_supplied')
        else:
            for key in ('root_goal_sha256','root_snapshot_sha256','history_pixels_sha256','raw_history_sha256'):
                if actual.get(key) is None:missing.append(key)
            for j,old in enumerate(index):
                reasons=[]
                if same_pose(actual['physical_pose'],old['physical_pose']):reasons.append('physical_pose_alias')
                for key in ('root_goal_sha256','root_state_sha256','root_snapshot_sha256','history_pixels_sha256'):
                    if actual.get(key) is not None and old.get(key) is not None and actual[key]==old[key]:reasons.append(key)
                if reasons:hits.append(dict(kind='historical',index=j,reasons=reasons))
            for j,other in enumerate(normalized):
                if j==i or other is None:continue
                reasons=[]
                if same_pose(actual['physical_pose'],other['physical_pose']):reasons.append('physical_pose_alias')
                for key in ('root_goal_sha256','root_state_sha256','root_snapshot_sha256','history_pixels_sha256'):
                    if actual.get(key) is not None and other.get(key) is not None and actual[key]==other[key]:reasons.append(key)
                if reasons:hits.append(dict(kind='batch',slot=j,reasons=reasons))
        rows.append(dict(slot=i,identity=actual,duplicate_evidence=hits,missing=missing,
            status='excluded_duplicate' if hits else 'blocked_missing_actual_identity' if missing else
                   'no_known_duplicate_still_blocked_by_coverage',
            coverage_certified=False,freshness_certified=False,source_run_authorized=False))
    return rows


def run():
    saved={p:bound_read(p,h) for p,h in PINS.items()}
    from experiments.spectral_reachability import chat_freshroots_20261027_closure as c
    for rel,expected in c.PINS.items():
        if rel.startswith('experiments/'):bound_read(ROOT/rel,expected)
    from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as r
    base=c.strict_json(saved[str(BASE)]);supp=c.strict_json(saved[str(PREFIX)]);scope=c.strict_json(saved[str(SCOPE)])
    parents=[]
    for desc in scope['reports']:
        raw=bound_read(desc['path'],desc['sha256'])
        for parent in parent_prefixes(raw,desc['adapter'],r.project,r.v.state_identity):
            parent['evidence_path']=desc['path'];parent['evidence_sha256']=desc['sha256'];parents.append(parent)
    groups,mappings=join_parents(supp['records'],parents)
    index=[]
    for bridge in base['observed_historical_roots']:
        index.append(dict(**bridge['metadata'],evidence=bridge['evidence']))
    for i,record in enumerate(supp['records']):
        index.append(dict(**record,parent_mapping=mappings[i]))
    assessment=assess(index,[dict(slot=i,root_state=None) for i in range(4)],r.v.state_identity,r.v.same_pose)
    return dict(schema='chat_freshroots_20261027.consumer_manifest.v2',
        bindings=[dict(path=p,sha256=h) for p,h in PINS.items()],
        consumer_sha256=digest(Path(__file__).read_bytes()),parent_report_bindings=scope['reports'],
        historical_exclusion_index=index,parent_identity_groups=groups,prefix_parent_mappings=mappings,
        summary=dict(historical_records=len(index),parent_report_prefix_records=len(parents),
                     prefix_records=len(mappings),associated=sum(x['status']=='explicit_ancestor_branch_associated' for x in mappings),
                     ambiguous=sum(x['status']=='ambiguous_ancestor_mapping' for x in mappings),
                     missing=sum(x['status']=='parent_mapping_missing' for x in mappings)),
        candidates=base['candidates'],sampling_denominator=4,candidate_identity_assessment=assessment,
        certified_fresh_roots=[],coverage_certified=False,freshness_certified=False,source_run_authorized=False,
        original_reset_sources=8,original_reset_blocking_refs=16,old_unknowns_discharged=0,
        unresolved_dependencies=base['unresolved_dependencies'],
        additional_required_evidence='ancestor branch mapping is not original sampled-state equality, reset/seed or RGB3/raw10/snapshot attestation',
        neural_calls=0,physics_calls=0)


if __name__=='__main__':print(canonical(run()).decode(),end='')
