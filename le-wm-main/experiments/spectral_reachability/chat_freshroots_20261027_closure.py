"""Bounded saved-only closure. Stdout is metadata, never an execution permit.

Run from le-wm-main with python -B -m experiments.spectral_reachability.
chat_freshroots_20261027_closure. No output files are opened by this consumer.
"""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTER = ROOT.parent
PREFIX = 'chat_freshroots_20261027_'
PINS = {
    'results/moving_authority_inventory_bridge_20261027_v1.inventory.json':
        '9fc0218a8c39b194abac363ca59bb7877323bdcf882e8aefd54a1c9effb8bd9a',
    'results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json':
        'e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591',
    'experiments/spectral_reachability/root_exclusion_authority_resolution_20261027_v1.py':
        '41474b63621545c04dc9ece6e81fc8bf9dd1c712e5ced5fee992c85d25450c04',
    'experiments/spectral_reachability/moving_root_exclusion_manifest_20261027_v1.py':
        'e6521ae535bdf26b6544868bf3c9018761332ee3b339936475ea54707c0db79c',
    'experiments/spectral_reachability/p1_conditional_source_metadata_20261027_v1.py':
        '13e5b65c149f56d7a0285d262f264dfe933b55a8366de0223e6724aedb3dacde',
    'experiments/spectral_reachability/moving_authority_identity_link_closure_20261027_v1.py':
        '5e16f51d3cd69a5826803c750e26637a931a923e60233c9c6c0c00924ee4ac18',
    'experiments/spectral_reachability/moving_historical_reset_quarantine_20261027_v1.py':
        '7df166c45b2285d44d2cc53d5815c91d6893f0a0080e180675deb7c276d004d8',
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound_read(path, expected, cap=8*1024**2):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and p.resolve() == p
            and p.is_relative_to(OUTER), 'canonical project file only')
    require(p.stat().st_size <= cap, 'input cap')
    with p.open('rb') as stream:
        raw = stream.read(cap+1)
    require(len(raw) <= cap and digest(raw) == expected, 'input size/SHA mismatch: '+str(p))
    return raw


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def strict_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_plan(plan):
    require(plan['schema'] == 'chat_freshroots_20261027.preregistration.v1', 'plan schema')
    expected = [dict(slot=i, root_seed=203027101+i, reset_seed=203027201+i,
                     witness_sign=1 if i % 2 == 0 else -1) for i in range(4)]
    require(plan['candidate_order'] == expected, 'fixed candidate order changed')
    require(all(type(r[k]) is int for r in plan['candidate_order'] for k in r), 'literal ints')
    require(type(plan['sampling_denominator']) is int and plan['sampling_denominator'] == 4,
            'fixed denominator')
    require(plan['replacement_or_retry'] is False and plan['new_outcomes_opened'] is False,
            'no retries/outcome selection')


def seed_conflicts(candidate, seeds, spans):
    """Cross both namespaces; half-open inputs only, constant work per span."""
    hits = []
    for field in ('root_seed', 'reset_seed'):
        value = candidate[field]
        require(type(value) is int and 0 <= value < 2**63, 'literal seed')
        for row in seeds:
            require(type(row['seed']) is int, 'literal historical seed')
            if value == row['seed']:
                hits.append(dict(candidate_field=field, seed=value, evidence=row))
        for row in spans:
            lo, hi = row['half_open']
            require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= 2**63,
                    'half-open span')
            if lo <= value < hi:
                hits.append(dict(candidate_field=field, seed=value, evidence=row))
    return hits


def audit_witnesses(graph):
    """Independent verification of every emitted path/SHA directed witness."""
    nodes = {s['path']: s for s in graph['sources']}
    require(len(nodes) == len(graph['sources']), 'duplicate graph node')
    edges = {e['id']: e for e in graph['edges']}
    require(len(edges) == len(graph['edges']), 'duplicate edge ID')
    count = 0
    for source in graph['sources']:
        for assoc in source['associations']:
            cursor = source['path']; visited = {cursor}
            require(bool(assoc['witness_edge_ids']), 'empty witness')
            for edge_id in assoc['witness_edge_ids']:
                edge = edges[edge_id]
                require(edge['status'] == 'bound' and edge['source_path'] == cursor
                        and edge['source_sha256'] == nodes[cursor]['sha256'],
                        'discontinuous/unbound witness')
                target = nodes[edge['target_path']]
                require(edge['target_sha256'] == target['sha256'], 'witness SHA')
                pointer = edge['pointer']
                parts = pointer.split('/')
                key = parts[-1].replace('~1', '/').replace('~0', '~')
                require(key == edge['target_path'], 'pointer path binding')
                cursor = target['path']
                require(cursor not in visited, 'cycle'); visited.add(cursor)
            require(cursor == assoc['target_path'] and
                    nodes[cursor]['sha256'] == assoc['target_sha256'], 'witness terminal')
            require(bool(nodes[cursor]['direct_identity_projections']), 'no terminal identity')
            count += 1
    return count


def run():
    # Verify exact source bytes before importing the existing read-only consumers.
    inputs = []
    for rel, sha in PINS.items():
        bound_read(ROOT/rel, sha)
        inputs.append(dict(path=str(ROOT/rel), sha256=sha))
    from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as resolver
    from experiments.spectral_reachability import moving_authority_identity_link_closure_20261027_v1 as links
    from experiments.spectral_reachability import moving_historical_reset_quarantine_20261027_v1 as quarantine

    plan_path = OUTER/'docs/chat_freshroots_20261027_preregistration.json'
    plan_raw = bound_read(plan_path, '79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e')
    require(len(plan_raw) < 16384, 'plan cap')
    plan = strict_json(plan_raw); validate_plan(plan)
    protocol = OUTER/'docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md'
    bound_read(protocol, plan['protocol_sha256'])
    inventory_rel = 'results/moving_authority_inventory_bridge_20261027_v1.inventory.json'
    inventory = strict_json(bound_read(ROOT/inventory_rel, PINS[inventory_rel]))
    require(inventory['schema'] == resolver.NAME+'.inventory' and len(inventory['files']) == 618,
            'exact inventory contract')
    require(len({f['path'] for f in inventory['files']}) == 618, 'duplicate inventory')
    reader = resolver.Reader()
    capture = resolver.generate(reader, inventory['files'])
    capture['inventory_binding'] = dict(path=str(ROOT/inventory_rel), sha256=PINS[inventory_rel])
    # Recompute, do not merely quote the saved graph or its summary.
    graph = links.generate(capture)
    verified_chains = audit_witnesses(graph)
    q = quarantine.build()
    seeds = capture['stable_identity_projections']['seed_projections'][:]
    for bridge in capture['bridge_payload']['records']:
        for kind in ('root_seed', 'reset_seed'):
            seeds.append(dict(kind=kind, seed=bridge[kind],
                              evidence_pointer='/bridge_payload/records/'+bridge['record_id']))
    spans = []
    # Prepare reservations are projected from their original bounded envelopes.
    for sid, rel, sha in resolver.PINNED:
        if not sid.endswith('prepare'):
            continue
        p = resolver.unpack_prepare(reader.read(ROOT/rel, sha))
        for pointer, value in p.leaves.items():
            if pointer[-1:] in [('root_seed',), ('reset_seed',)] and type(value) is int:
                seeds.append(dict(kind=pointer[-1], seed=value, path=str(ROOT/rel),
                                  sha256=sha, pointer=resolver.pointer(pointer)))
            if pointer[:2] == ('exclusions', 'seed_families') and pointer[-1] == 'seed':
                require(type(value) is int, 'literal inherited seed')
                seeds.append(dict(kind='inherited_seed_family', seed=value, path=str(ROOT/rel),
                                  sha256=sha, pointer=resolver.pointer(pointer)))
            if pointer[:2] == ('exclusions', 'reset_spans') and pointer[-2:] == ('span', 0):
                hi = p.leaves.get(pointer[:-1]+(1,))
                spans.append(dict(half_open=[value, hi], path=str(ROOT/rel), sha256=sha,
                                  pointer=resolver.pointer(pointer[:-1]), certified=False,
                                  kind='inherited_reserved_reset_span_not_coverage_attestation'))
            if len(pointer) >= 2 and pointer[-2:] == ('reset_seed_span', 0):
                hi = p.leaves.get(pointer[:-1]+(1,))
                spans.append(dict(half_open=[value, hi], path=str(ROOT/rel), sha256=sha,
                                  pointer=resolver.pointer(pointer[:-1]), certified=False,
                                  kind='prepare_reserved_span'))
    for row in q['rows']:
        for envelope in row['conditional_envelopes']:
            spans.append(dict(half_open=[envelope['start'], envelope['end']+1],
                              source_id=row['source_id'], path=str(ROOT/row['path']),
                              sha256=row['sha256'], kind=envelope['kind'], certified=False,
                              authority='conditional_code_family_not_execution_attested'))
    candidates = [dict(**c, known_seed_conflicts=seed_conflicts(c, seeds, spans),
                       status='blocked_pending_coverage_and_actual_identity',
                       actual_root_identity=None, freshness_certified=False,
                       source_run_authorized=False) for c in plan['candidate_order']]
    sources = []
    for s in graph['sources']:
        sources.append(dict(path=s['path'], sha256=s['sha256'],
                            direct_identity=s['direct_identity_projections'],
                            associations=s['associations'], inherited_refs=s['inherited_refs'],
                            blockers=s['original_blockers'], unknown=s['unknown'],
                            required_evidence=('source-bound root/state/episode mapping; upstream association is insufficient'
                                               if not s['direct_identity_projections'] else
                                               'complete attempts/reservations and MAIN scope adjudication still required')))
    result = dict(schema='chat_freshroots_20261027.root_manifest.v1',
                  status='metadata_closure_recomputed_freshness_not_certified',
                  preregistration=dict(path=str(plan_path), sha256=digest(plan_raw)),
                  inputs=inputs, protocol_binding=capture['protocol_binding'],
                  source_sha256=digest(Path(__file__).read_bytes()),
                  inventory_scope='sealed 618-path snapshot only; not current complete history',
                  summary=capture['summary'], graph_summary=graph['summary'],
                  independently_checked_witness_chains=verified_chains,
                  evidence_census=sources, graph_edges=graph['edges'],
                  root_goal_identity_index=capture['stable_identity_projections']['root_goal_groups'],
                  observed_historical_roots=capture['physical_identity_bridges'],
                  physical_mapping_errors=capture['physical_mapping_errors'],
                  used_or_reserved_seed_evidence=seeds, half_open_exclusion_spans=spans,
                  historical_reset_evidence=q,
                  authority_requests=capture['authority_requests'],
                  candidates=candidates, sampling_denominator=4,
                  certified_fresh_root_count=0, source_run_authorized=False,
                  coverage_certified=False, freshness_certified=False, P1_admission=False,
                  actual_neural_calls=0, actual_physics_calls=0,
                  unresolved_dependencies=[
                      '8 original runs / 16 refs: report->executed producer+root factory+env closure or complete reset ledger, including failed/unselected attempts',
                      'each evidence_census missing identity: explicit source-bound state/episode bridge, not inferred from graph lineage',
                      'original_four authority: exact family scope remains unverified, not relabelled from family_hint',
                      'MAIN external exact-content-SHA coverage approval for all families/reservations and failure sources',
                      'explicit delta inventory for post-snapshot MAIN activity; 618 is not current full coverage',
                      'future fixed four actual roots: historical and intra-batch pose/snapshot/RGB3 exclusion before witness dispatch',
                      'P1 conditional task admission is a separate MAIN obligation; no model/control experiment performed here'
                  ])
    # Time-of-check protection for all current evidence and imported source bytes.
    for row in reader.files.values():
        require(resolver.v.file_sha(Path(row['path'])) == row['sha256'], 'evidence changed during closure')
    for rel, sha in PINS.items():
        bound_read(ROOT/rel, sha)
    require(plan_path.read_bytes() == plan_raw, 'plan changed during closure')
    result['verified_evidence_files'] = sorted(reader.files.values(), key=lambda x:x['path'])
    require(len(canonical(result)) < 8*1024**2, 'output cap')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk-index', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=250000)
    args = parser.parse_args()
    require(args.chunk_index >= 0 and 1 <= args.chunk_size <= 250000, 'bounded stdout chunk')
    raw = canonical(run())
    start = args.chunk_index * args.chunk_size
    print(canonical(dict(full_sha256=digest(raw), full_bytes=len(raw), index=args.chunk_index,
                         chunk=raw.decode()[start:start+args.chunk_size])).decode(), end='')
