"""Explicit saved-prefix projection supplement, never a coverage approval.

Only stdout. No directory discovery, producer import, simulation or neural calls.
"""
from __future__ import annotations
import hashlib
from io import BytesIO
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT/'results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json'
CAPTURE_SHA = 'e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591'
SCOPE = ROOT.parent/'docs/chat_freshroots_20261027_prefix_scope.json'
BASE = ROOT/'results/chat_freshroots_20261027_root_manifest.json'
BASE_SHA = '3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838'
FILE_CAP = 8*1024**2
TOTAL_CAP = 64*1024**2


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()


def read_bound(path, expected):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and '..' not in p.parts
            and p.is_relative_to(ROOT.parent), 'literal project path')
    require(p.stat().st_size <= FILE_CAP, 'file cap')
    with p.open('rb') as stream:
        raw = stream.read(FILE_CAP+1)
    require(len(raw) <= FILE_CAP and sha(raw) == expected, 'file SHA/cap mismatch')
    return raw


def project_prefix(raw, projector):
    p = projector(BytesIO(raw), [('prefix_state','*'),('prefix_rgb_sha256',)])
    p.limit = FILE_CAP
    p.walk(); p.whitespace()
    require(p.peek() is None, 'trailing input')
    state = [p.leaves.get(('prefix_state', i)) for i in range(7)]
    require(set(k for k in p.leaves if k[:1] == ('prefix_state',)) ==
            {('prefix_state', i) for i in range(7)}, 'exact prefix state7 required')
    require(all(type(x) in (int,float) for x in state), 'literal numeric state7')
    rgb = p.leaves.get(('prefix_rgb_sha256',))
    require(type(rgb) is str and len(rgb)==64 and all(x in '0123456789abcdef' for x in rgb),
            'prefix RGB SHA required')
    return state, rgb


def alias_groups(records, same_pose):
    """Conservative connected components, not independent samples."""
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    edges = []
    for i, a in enumerate(records):
        for j in range(i):
            b = records[j]
            if same_pose(a['physical_pose'], b['physical_pose']):
                parents[find(i)] = find(j)
                edges.append([j,i])
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i),[]).append(i)
    return sorted(groups.values(),key=lambda g:g[0]), edges


def run():
    # Source identities are reused only after verifying the previously sealed pins.
    read_bound(ROOT/'experiments/spectral_reachability/chat_freshroots_20261027_closure.py',
               'd408e2dbab9a56d1daa1024910895eeb485dabab35ed2dac95ead328cac37b4a')
    from experiments.spectral_reachability import chat_freshroots_20261027_closure as c
    source_pins = {k:v for k,v in c.PINS.items() if k.startswith('experiments/')}
    for rel, expected in source_pins.items():
        read_bound(ROOT/rel, expected)
    from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as r
    raw = read_bound(CAPTURE,CAPTURE_SHA); capture = c.strict_json(raw)
    scope_raw = read_bound(SCOPE,'dc2d13ac58b0392aba24253739e151000408704b37d24fe5c77a8810722a0c3b')
    require(len(scope_raw)<16384,'scope cap')
    scope = c.strict_json(scope_raw)
    require(scope['capture_sha256']==CAPTURE_SHA and scope['expected_file_denominator']==80,
            'scope binding')
    base_raw = read_bound(BASE,BASE_SHA); base = c.strict_json(base_raw)
    selected = [x for x in capture['history_file_requests'] if 'prefix_state' in x['top_level_keys']]
    require(len(selected)==80 and len({x['path'] for x in selected})==80, 'fixed prefix denominator')
    records=[]; failures=[]; used=0
    for row in selected:
        evidence=dict(path=row['path'],sha256=row['sha256'],pointer='/prefix_state')
        try:
            require(used+Path(row['path']).stat().st_size<=TOTAL_CAP,'total prefix byte cap')
            data=read_bound(row['path'],row['sha256']); used+=len(data)
            require(used<=TOTAL_CAP,'total prefix byte cap')
            state,rgb=project_prefix(data,r.v.Projection)
            identity=r.v.state_identity(state)
            records.append(dict(evidence=evidence,record_type='reported_historical_observed_prefix',
                state7=state,**identity,prefix_rgb_sha256=rgb,
                root_seed=None,reset_seed=None,root_snapshot_sha256=None,raw_history_sha256=None,
                original_sampling_root_equivalence_proved=False,
                unresolved=['sampling-root/episode binding','seed/reset attempt mapping',
                            'snapshot and full RGB3/raw10 history','historical scope approval']))
        except (ValueError,KeyError,TypeError,OSError) as exc:
            failures.append(dict(evidence=evidence,error=str(exc)))
    groups,edges=alias_groups(records,r.v.same_pose)
    matches=[]
    for i,record in enumerate(records):
        for bridge in base['observed_historical_roots']:
            meta=bridge['metadata']
            if r.v.same_pose(record['physical_pose'],meta['physical_pose']):
                matches.append(dict(record_index=i,known_record_id=meta['record_id'],
                                    relation='numeric_pose_alias_only_not_lineage_proof'))
    return dict(schema='chat_freshroots_20261027.prefix_identity_supplement.v1',
        capture_binding=dict(path=str(CAPTURE),sha256=CAPTURE_SHA),
        base_manifest_binding=dict(path=str(BASE),sha256=BASE_SHA),
        scope_binding=dict(path=str(SCOPE),sha256=sha(scope_raw)),
        consumer_binding=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).read_bytes())),
        source_pins=[dict(path=str(ROOT/k),sha256=v) for k,v in source_pins.items()],
        file_denominator=80,parsed_records=len(records),failed_records=failures,
        records=records,pose_alias_components=groups,pose_alias_edges=edges,
        known_historical_pose_matches=matches,bytes_read_prefix_files=used,
        interpretation='Additional exclusion identities only. Pose groups are not independent roots; RGB hash is one declared prefix image, not RGB3 or independently decoded PNG.',
        old_unknowns_discharged=0,historical_reset_sources=8,historical_reset_blocking_refs=16,
        coverage_certified=False,freshness_certified=False,source_run_authorized=False,
        sampling_denominator=4,candidates=base['candidates'],neural_calls=0,physics_calls=0)


if __name__=='__main__':
    print(canonical(run()).decode(),end='')
