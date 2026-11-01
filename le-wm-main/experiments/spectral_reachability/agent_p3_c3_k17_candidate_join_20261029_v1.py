"""Offline source seal and fixture-only outer/lifecycle join; no dispatch authority.

The lifecycle is a *post-execution projection*, never a record of dispatch time.
External manifest SHA is required: hashes without an independent anchor prove
only internal consistency, not authenticity or native provenance.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer
from experiments.spectral_reachability import agent_p3_c3_k17_native_store_20261029_v1 as native_store
from experiments.spectral_reachability import agent_p3_c3_k17_lifecycle_20261029_v1 as lifecycle
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport

SCHEMA = 'agent_p3_c3_k17_candidate_join_20261029_v1'


def sha(path):
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise ValueError('missing or symlink artifact')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def normalize(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('object array source')
        a = np.ascontiguousarray(value)
        return {'array_dtype': a.dtype.str, 'shape': list(a.shape), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, dict):
        entries = [[normalize(k), normalize(v)] for k, v in value.items()]
        entries.sort(key=lambda entry: transport.canonical(entry[0]))
        return {'map_entries': entries}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError('unsupported source type: ' + type(value).__name__)


def snapshot(data, sources):
    if len(sources) != 6:
        raise ValueError('six source slots')
    for (item, _, _), identity in zip(sources, outer.ORDER, strict=True):
        if (item['slot'], item['root'], item['mode'], item['K']) != (*identity, 17):
            raise ValueError('source identity')
    return {'schema': SCHEMA + '.sources', 'data': normalize(data), 'sources': normalize(sources)}


def write(path, obj):
    raw = transport.canonical(obj) + b'\n'
    with Path(path).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path):
    sha(path)
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if raw != transport.canonical(value) + b'\n':
        raise ValueError('noncanonical artifact')
    return value


def run_fixture(directory, factory):
    """Exercise the join with fixtures only. Never call with a real environment."""
    path = Path(directory)
    if path.exists() or path.is_symlink():
        raise ValueError('exclusive candidate directory')
    path.mkdir()
    data, sources = outer.material()
    write(path/'sources.json', snapshot(data, sources))
    store = native_store.NativeStore(path/'outer')
    result = outer.execute_material(data, sources, store, factory)
    write(path/'outer_result.json', result)
    rows = result['rows']

    def project(identity, _):
        row = rows[identity['index']]
        if row['status'] == 'engineering_failed':
            raise ValueError('outer engineering failure; see outer result')
        if row['status'] not in ('horizon_completed', 'native_stopped'):
            raise ValueError('unexpected projected status')
        return {'status': row['status'], 'outer_index': identity['index']}

    lifecycle.run(path/'lifecycle', lambda: None, project)
    files = ('sources.json', 'outer_result.json', 'outer/journal.jsonl',
             'lifecycle/report.json', 'lifecycle/completion.json')
    manifest = {'schema': SCHEMA + '.manifest', 'files': {name: sha(path/name) for name in files},
                'fixture_only': True, 'lifecycle_is_posthoc_projection': True}
    write(path/'manifest.json', manifest)
    return {'manifest_sha256': sha(path/'manifest.json')}


def consume(directory, manifest_sha256, *, verify_sources=False):
    """Read-only independent consistency check; optional current source rebuild."""
    path = Path(directory)
    if sha(path/'manifest.json') != manifest_sha256:
        raise ValueError('external manifest SHA')
    manifest = read(path/'manifest.json')
    names = ('sources.json', 'outer_result.json', 'outer/journal.jsonl',
             'lifecycle/report.json', 'lifecycle/completion.json')
    if (manifest.get('schema') != SCHEMA + '.manifest' or manifest.get('fixture_only') is not True
            or manifest.get('lifecycle_is_posthoc_projection') is not True
            or set(manifest.get('files', {})) != set(names)):
        raise ValueError('manifest contract')
    for name in names:
        if sha(path/name) != manifest['files'][name]:
            raise ValueError('artifact SHA: ' + name)
    source = read(path/'sources.json')
    if verify_sources and source != snapshot(*outer.material()):
        raise ValueError('source snapshot mismatch')
    result = read(path/'outer_result.json')
    rows = result['rows']
    if len(rows) != 6 or result['fixed_denominator'] != 6 or result['physical_release'] is not False or result['production_authorized'] is not False:
        raise ValueError('outer contract')
    projected = lifecycle.consume(path/'lifecycle', manifest['files']['lifecycle/report.json'], manifest['files']['lifecycle/completion.json'])
    events = transport.read_journal(path/'outer'/'journal.jsonl')
    closed = [e['payload']['row'] for e in events if e['kind'] == 'branch_closed']
    if closed != rows[:len(closed)]:
        raise ValueError('outer branch records')
    failed = False
    for i, (row, projection, ident) in enumerate(zip(rows, projected['rows'], outer.ORDER, strict=True)):
        if (row['index'], row['slot'], row['root_slot'], row['mode']) != (i, *ident):
            raise ValueError('outer slot identity')
        status = row['status']
        if status != projection['status'] or (failed and status != 'not_started'):
            raise ValueError('projection mismatch')
        if status == 'engineering_failed':
            failed = True
        if status == 'not_started' and not failed:
            raise ValueError('unexplained unstarted slot')
        if status not in ('horizon_completed', 'native_stopped', 'engineering_failed', 'not_started'):
            raise ValueError('outer status')
    if len(closed) != sum(r['status'] != 'not_started' for r in rows):
        raise ValueError('missing branch close')
    capture = native_store.inspect_capture_journal(path/'outer')
    return {'schema': SCHEMA + '.consume', 'rows': rows, 'lifecycle_verified': True,
            'source_snapshot_matches_current': bool(verify_sources), 'capture_hashes': capture,
            'native_provenance_verified': False, 'physical_release': False, 'equal_compute': False,
            'limitations': 'fixture-only posthoc projection; no independent native action, resource, guard, physical origin or outcome audit'}
