"""Saved-capture-only input lineage closure; never physical root equivalence.

Reads one explicitly SHA-bound metadata report, not its referenced files.
Cycles/conflicts suppress associations for every source reaching them.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re

from experiments.spectral_reachability import p1_conditional_source_metadata_20261027_v1 as lexical

NAME = 'moving_authority_identity_link_closure_20261027_v1'
INPUT_SCHEMA = 'root_exclusion_authority_resolution_20261027_v1.resolution_request'
ROOT = Path(__file__).resolve().parents[2]
RELATION = 'explicit_input_binding_not_physical_root_equivalence'
RESET_REASON = 'historical_reset_attempt_bound_not_attested'
INPUT_CAP = 8 * 1024**2
OUTPUT_CAP = 16 * 1024**2
NODE_CAP = 1024
EDGE_CAP = 8192
ASSOCIATION_CAP = 40000
IDENTITIES = {'root_seed', 'reset_seed', 'root_sha256', 'root_goal_sha256',
              'task_id', 'episode_id', 'blind_sha256'}
ROOT_IDENTITIES = {'root_seed', 'reset_seed', 'root_sha256', 'root_goal_sha256'}
BINDINGS = {('bindings',), ('plans_bindings',), ('protocol', 'bindings'), ('source_bindings',)}
PATTERNS = [('schema',), ('summary', '*'), ('inventory_binding', '*')]
PATTERNS += [(k,) for k in ('MAIN_coverage_approval', 'freshness_certified', 'P1_admission')]
PATTERNS += [('authority_requests', '*', k) for k in ('family', 'status', 'remaining')]
PATTERNS += [('authority_requests', '*', 'inventory_paths', '*')]
PATTERNS += [('history_file_requests', '*', k) for k in
             ('path', 'sha256', 'status', 'hash_authority', 'inventory_family')]
PATTERNS += [('history_file_requests', '*', k, '*') for k in ('blockers',)]
for container, fields in (
    ('projection', ('pointer', 'value')), ('root_identity_projections', ('pointer', 'value')),
    ('input_lineage_links', ('pointer', 'target_path', 'target_sha256', 'relation')),
    ('inherited_refs', ('parent', 'pointer', 'historical_sha256', 'original_reason')),
):
    PATTERNS += [('history_file_requests', '*', container, '*', k) for k in fields]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def valid_hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def tokens(pointer):
    require(type(pointer) is str and pointer.startswith('/') and len(pointer) <= 4096,
            'literal bounded JSON pointer required')
    require(re.search(r'~(?![01])', pointer) is None, 'invalid JSON pointer escape')
    return tuple(x.replace('~1', '/').replace('~0', '~') for x in pointer[1:].split('/'))


def literal_path(value):
    require(type(value) is str and 0 < len(value) <= 4096, 'literal path required')
    p = Path(value)
    require(p.is_absolute() and '..' not in p.parts and str(p) == value and
            p.is_relative_to(ROOT) and p != ROOT, 'literal project path required')


class CaptureProjection(lexical.Projection):
    """Decode only the metadata allowlist; unrelated payloads are opaque."""

    def __init__(self, raw):
        super().__init__(BytesIO(raw), INPUT_CAP)
        self.visited = 0

    def walk(self, path=()):
        self.visited += 1
        require(self.visited <= 100000 and len(path) <= 16, 'projection structure cap')
        self.whitespace()
        if not any(lexical.matches(p, path, True) for p in PATTERNS):
            self.skip()
            return None
        ch = self.peek()
        if ch == 123:
            self.get(); self.whitespace()
            out = {}; seen = set()
            if self.peek() == 125:
                self.get(); return out
            while True:
                self.whitespace(); key = self.string(True)
                require(key not in seen and len(seen) < 256, 'duplicate metadata key/key cap')
                seen.add(key); self.whitespace()
                require(self.get() == 58, 'JSON colon')
                child = path + (key,)
                value = self.walk(child)
                if any(lexical.matches(p, child, True) for p in PATTERNS):
                    out[key] = value
                self.whitespace(); sep = self.get()
                if sep == 125:
                    return out
                require(sep == 44, 'JSON object separator')
        if ch == 91:
            self.get(); self.whitespace(); out = []
            if self.peek() == 93:
                self.get(); return out
            while True:
                require(len(out) < 40000, 'metadata array cap')
                out.append(self.walk(path + (len(out),)))
                self.whitespace(); sep = self.get()
                if sep == 93:
                    return out
                require(sep == 44, 'JSON array separator')
        require(any(lexical.matches(p, path) for p in PATTERNS), 'metadata container required')
        if ch == 34:
            value = self.string(True)
            require(len(value) <= 4096, 'metadata string cap')
            return value
        scalar = bytearray()
        while self.peek() not in (None, 32, 9, 10, 13, 44, 93, 125):
            scalar.append(self.get())
            require(len(scalar) <= 1024, 'metadata scalar cap')
        value = json.loads(scalar)
        require(value is None or type(value) in (bool, int, float), 'metadata scalar')
        canonical(value)  # Reject nonfinite values in decoded metadata.
        return value


def projection_map(items):
    require(type(items) is list and len(items) <= 40000, 'projection list required')
    result = {}
    for item in items:
        require(type(item) is dict and set(item) == {'pointer', 'value'}, 'projection fields')
        pointer = item['pointer']; tokens(pointer)
        require(pointer not in result, 'duplicate/conflicting projection pointer')
        result[pointer] = item['value']
    return result


def binding_path(pointer):
    parts = tokens(pointer)
    return parts[-1] if parts[:-1] in BINDINGS and parts[-1].startswith('/') else None


def validate_nodes(report):
    rows = report['history_file_requests']
    require(type(rows) is list and 0 < len(rows) <= NODE_CAP, 'source count cap')
    nodes = {}; maps = {}
    reset_count = 0
    for row in rows:
        require(type(row) is dict and {'path', 'sha256', 'projection', 'root_identity_projections',
                'input_lineage_links', 'inherited_refs', 'blockers', 'status', 'hash_authority',
                'inventory_family'} <= set(row), 'missing source metadata')
        path = row['path']; literal_path(path)
        require(path not in nodes, 'duplicate/conflicting source path')
        require(valid_hash(row['sha256']), 'captured source SHA required')
        require(row['status'] == 'metadata_captured_not_coverage_certified', 'uncaptured/blocked source')
        for k in ('blockers', 'inherited_refs', 'input_lineage_links'):
            require(type(row[k]) is list, 'metadata list required')
        require(all(type(x) is str for x in row['blockers']), 'original blocker strings required')
        pm = projection_map(row['projection'])
        im = projection_map(row['root_identity_projections'])
        for pointer, value in im.items():
            field = tokens(pointer)[-1]
            require(field in IDENTITIES and pointer in pm and canonical(pm[pointer]) == canonical(value),
                    'identity must match captured projection')
            if field in ('root_seed', 'reset_seed'):
                require(type(value) is int and 0 <= value < 2**63, 'literal nonnegative identity seed')
            elif field.endswith('sha256'):
                require(valid_hash(value), 'identity SHA required')
            else:
                require((type(value) is str and bool(value)) or type(value) is int, 'identity scalar required')
        # Prevent an omitted identity list from silently changing graph semantics.
        require(set(im) == {p for p in pm if tokens(p)[-1] in IDENTITIES}, 'incomplete identity projection list')
        for ref in row['inherited_refs']:
            require(type(ref) is dict and set(ref) == {'parent', 'pointer', 'historical_sha256', 'original_reason'},
                    'inherited reference fields')
            require(type(ref['parent']) is str and type(ref['original_reason']) is str, 'inherited reference strings')
            tokens(ref['pointer'])
            historical = ref['historical_sha256']
            require(historical is None or historical == row['sha256'], 'historical SHA conflict')
            reset_count += ref['original_reason'] == RESET_REASON
        nodes[path] = row; maps[path] = pm
    require(type(report['summary']) is dict and
            type(report['summary'].get('inherited_reset_bound_unattested_refs')) is int and
            report['summary']['inherited_reset_bound_unattested_refs'] == reset_count,
            'reset-bound blocker count mismatch')
    return nodes, maps, reset_count


def build_edges(nodes, maps):
    edges = []; adjacency = {path: [] for path in nodes}; conflicts = set()
    for path, row in nodes.items():
        seen = set()
        for link in row['input_lineage_links']:
            require(type(link) is dict and set(link) == {'pointer', 'target_path', 'target_sha256', 'relation'},
                    'input link fields')
            require(len(edges) < EDGE_CAP, 'edge cap')
            signature = canonical(link)
            require(signature not in seen, 'duplicate input link')
            seen.add(signature)
            pointer = link['pointer']; bound = binding_path(pointer)
            target = link['target_path']; literal_path(target)
            edge = dict(id=len(edges), source_path=path, source_sha256=row['sha256'], **link)
            reason = None; status = 'conflict'
            if link['relation'] != RELATION:
                reason = 'unsupported_relation'
            elif pointer not in maps[path]:
                reason = 'missing_projection_for_edge'
            elif not valid_hash(link['target_sha256']):
                reason = 'missing_or_invalid_target_sha'
            elif bound is None:
                reason = 'not_explicit_path_sha_binding'
            elif maps[path][pointer] != link['target_sha256']:
                reason = 'projection_sha_conflict'
            elif bound != target:
                reason = 'path_alias_not_explicit_same_path'; status = 'unknown'
            elif target not in nodes:
                reason = 'target_not_captured'; status = 'unknown'
            elif nodes[target]['sha256'] != link['target_sha256']:
                reason = 'target_sha_conflict'
            else:
                status = 'bound'; adjacency[path].append(edge)
            edge.update(status=status, reason=reason)
            if status == 'conflict':
                conflicts.add(path)
            edges.append(edge)
    return edges, adjacency, conflicts


def reachable(start, adjacency):
    """Shortest explicit witness per target, without recursion or path explosion."""
    previous = {start: None}; queue = deque([start])
    while queue:
        for edge in adjacency[queue.popleft()]:
            target = edge['target_path']
            if target not in previous:
                previous[target] = edge; queue.append(target)
    indegree = {path: 0 for path in previous}
    for path in previous:
        for edge in adjacency[path]:
            indegree[edge['target_path']] += 1
    queue = deque(p for p, n in indegree.items() if n == 0); removed = 0
    while queue:
        path = queue.popleft(); removed += 1
        for edge in adjacency[path]:
            target = edge['target_path']; indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return previous, removed != len(previous)


def generate(report):
    require(report['schema'] == INPUT_SCHEMA, 'capture schema')
    for flag in ('MAIN_coverage_approval', 'freshness_certified', 'P1_admission'):
        require(report[flag] is False, 'no inherited self-approval allowed')
    nodes, maps, reset_count = validate_nodes(report)
    edges, adjacency, conflicts = build_edges(nodes, maps)
    unlinked_by_source = {}
    for path in nodes:
        bound_pointers = {e['pointer'] for e in adjacency[path]}
        unlinked = []
        for pointer, value in maps[path].items():
            bound = binding_path(pointer)
            if bound is not None and valid_hash(value) and pointer not in bound_pointers:
                unlinked.append(dict(pointer=pointer, bound_path=bound, sha256=value,
                                     reason='edge_not_captured' if bound in nodes else 'target_not_captured'))
        unlinked_by_source[path] = unlinked
    sources = []; association_count = 0
    for path, row in nodes.items():
        previous, cyclic = reachable(path, adjacency)
        poisoned = bool(set(previous) & conflicts)
        unknown = []; associations = []
        if cyclic:
            unknown.append('reachable_cycle')
        if poisoned:
            unknown.append('reachable_conflict')
        if not cyclic and not poisoned:
            for target in sorted(set(previous) - {path}):
                if not nodes[target]['root_identity_projections']:
                    continue
                chain = []; cursor = target
                while cursor != path:
                    edge = previous[cursor]; chain.append(edge['id']); cursor = edge['source_path']
                association_count += 1
                require(association_count <= ASSOCIATION_CAP, 'association cap')
                associations.append(dict(target_path=target, target_sha256=nodes[target]['sha256'],
                                         witness_edge_ids=chain[::-1], relation='upstream_contains_identity_metadata_only'))
        if not associations:
            unknown.append('no_bound_upstream_identity')
        unlinked = unlinked_by_source[path]
        if unlinked:
            unknown.append('unlinked_bindings_remain')
        if any(unlinked_by_source[p] for p in previous):
            unknown.append('reachable_unlinked_bindings')
        if any(e['status'] != 'bound' and e['source_path'] in previous for e in edges):
            unknown.append('reachable_unresolved_edges')
        sources.append(dict(path=path, sha256=row['sha256'], inventory_family=row['inventory_family'],
                            direct_identity_projections=row['root_identity_projections'],
                            original_blockers=row['blockers'], inherited_refs=row['inherited_refs'],
                            associations=associations, unlinked_bindings=unlinked, unknown=unknown,
                            status=('blocked_graph' if cyclic or poisoned else
                                    'upstream_metadata_associated_not_physical_identity' if associations else 'unknown_upstream_identity')))
    return dict(schema=NAME+'.metadata_link_closure', edges=edges, sources=sources,
                authority_requests=report['authority_requests'], original_summary=report['summary'],
                inventory_binding=report['inventory_binding'],
                summary=dict(sources=len(nodes), edges=len(edges), bound_edges=sum(e['status']=='bound' for e in edges),
                             conflict_edges=sum(e['status']=='conflict' for e in edges),
                             unknown_edges=sum(e['status']=='unknown' for e in edges),
                             associations=association_count,
                             sources_with_upstream_identity_association=sum(bool(s['associations']) for s in sources),
                             sources_without_upstream_identity_association=sum(not s['associations'] for s in sources),
                             graph_blocked_sources=sum(s['status']=='blocked_graph' for s in sources),
                             direct_identity_sources=sum(bool(s['direct_identity_projections']) for s in sources),
                             inherited_refs_preserved=sum(len(s['inherited_refs']) for s in sources),
                             inherited_refs_without_direct_identity=sum(len(s['inherited_refs']) for s in sources
                                 if not s['direct_identity_projections']),
                             inherited_refs_without_direct_but_associated=sum(len(s['inherited_refs']) for s in sources
                                 if not s['direct_identity_projections'] and s['associations']),
                             inherited_refs_without_direct_and_without_upstream=sum(len(s['inherited_refs']) for s in sources
                                 if not s['direct_identity_projections'] and not s['associations']),
                             preserved_reset_bound_unattested_refs=reset_count),
                physical_root_equivalence_proved=False, reset_attempt_bound_certified=False,
                MAIN_coverage_approval=False, freshness_certified=False, P1_admission=False,
                historical_unknowns_discharged=False,
                note='Directed input provenance only. Identity values remain attributed to their own source/pointer; upstream evidence is not the consumer physical root, reset bound, completeness or freshness.')


def convert(raw, expected_sha256):
    require(type(raw) is bytes and len(raw) <= INPUT_CAP, 'capture byte cap')
    require(valid_hash(expected_sha256) and hashlib.sha256(raw).hexdigest() == expected_sha256, 'capture SHA mismatch')
    parser = CaptureProjection(raw); report = parser.walk(); parser.whitespace()
    require(parser.peek() is None, 'trailing capture JSON')
    try:
        result = generate(report)
    except (KeyError, TypeError) as exc:
        raise ValueError('missing/malformed capture metadata') from exc
    result['capture_binding'] = dict(sha256=expected_sha256)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    with Path(args.input).open('rb') as stream:
        raw = stream.read(INPUT_CAP + 1)
    result = convert(raw, args.input_sha256)
    result['capture_binding']['path'] = str(Path(args.input).absolute())
    output = canonical(result)
    require(len(output) <= OUTPUT_CAP, 'closure output cap')
    with Path(args.output).open('xb') as stream:
        stream.write(output); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(path=str(Path(args.output).absolute()), sha256=hashlib.sha256(output).hexdigest(),
                         summary=result['summary']), sort_keys=True))


if __name__ == '__main__':
    main()
