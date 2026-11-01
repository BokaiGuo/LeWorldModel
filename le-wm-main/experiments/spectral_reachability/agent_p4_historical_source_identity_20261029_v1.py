"""Fixed A--H identity recovery, not an original-load or freshness certificate.

Original reports are lexically projected. Historical Python is parsed, never
imported/executed; fixed Git blobs are read, not checked out. No discovery.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import defaultdict
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import zlib

from experiments.spectral_reachability import agent_freshroot_authority_20261028_v1 as a
from experiments.spectral_reachability import root_exclusion_authority_resolution_20261027_v1 as r

NAME = 'agent_p4_historical_source_identity_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
AUTHORITY = ('results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json',
             '032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00')
CATALOG = ('results/agent_freshroots_resume_20261027_coverage.json', a.CATALOG_SHA)
DELTA = ('results/agent_freshroots_resume_20261027_next_delta.json',
         '5ef07ab7815fff0c392feef59309c0116596e95d541eb95413d1d966a8b2a29b')
HSEAL = ('results/pusht_d2p_p1_root_anchored_v3_physical_calibration_20261026_cuda/prerun_seal.json',
         '4e024c0ded185e0bb58c6ceaeb33a0ab1d784165a962995c28f6216d439ef81e')
ARCHIVE = ('results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/prerun_source_seal/manifest.json',
           '0f0f5b8b8a525990dcffd2393ada1a49270dd50a3d4865800343cfd9b8a39aef')
POSTRUN = 'results/pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda/postrun_implementation/'
POSTRUN_FILES = (
 ('preflight_pusht_d2p_p0.py', '5215f0b49b772d8def14997996774bfd49db614b52b9f615c164271591f7b83b'),
 ('preflight_pusht_d2p_p1_reconfiguration.py', '80b9674063360a2f51d2e146c830c29a5399bb53b9086b177f3b3f2dc3b0d5e4'),
 ('run_pusht_d2p_p1_physical_calibration.py', '7aa9f156484ec3ab483ce3f8308466f60b09a512232f21cd2bcd02f25944f114'))
DEPENDENCIES = (
 ('root_exclusion_authority_resolution_20261027_v1.py', '41474b63621545c04dc9ece6e81fc8bf9dd1c712e5ced5fee992c85d25450c04'),
 ('moving_root_exclusion_manifest_20261027_v1.py', 'e6521ae535bdf26b6544868bf3c9018761332ee3b339936475ea54707c0db79c'),
 ('p1_conditional_source_metadata_20261027_v1.py', '13e5b65c149f56d7a0285d262f264dfe933b55a8366de0223e6724aedb3dacde'),
 ('agent_freshroot_authority_20261028_v1.py', 'f7a30fcad18483e1848170acdd1fe4bd96e7d08b38804b18f1cf2cf1087870e7'))
FIELDS = ('root_id', 'root_sha256', 'causal_history_action_sha256',
          'causal_history_rgb_sha256', 'history_rgb_sha256', 'goal_rgb_sha256')
ABSENT = ('episode_id', 'reset_attempt_id', 'reset_seed', 'root_state7',
          'root_snapshot', 'raw_history_actions', 'individual_history_rgb_sha256')
REPORT_CAP = 4*1024**2
OUTPUT_CAP = 8*1024**2
FLOOR = 200*1024**2
need = a.need
sha = a.sha
encode = a.canonical


def citation(path, digest, pointer, value):
    return dict(path=str(path), sha256=digest, pointer=pointer, value=value)


def identity_projection(raw, source):
    """Only identity/config ancestry values; no J, compilation, costs or labels."""
    container = 'calibration_roots' if source['id'] in 'ABC' else 'rows'
    config = [('schema',), ('inputs', 'checkpoint_weights'),
              ('inputs', 'checkpoint_weights_sha256'), ('upstream_p0', 'path'),
              ('upstream_p0', 'sha256')]
    # The pinned catalog specifies the exact config allowlist; /protocol/outcome
    # here contains reserved seed/count settings, NOT measured outcomes.
    config += [tuple(x['pointer'][1:].split('/')) for x in source['protocol_projection']]
    patterns = config + [(container, '*', k) for k in FIELDS]
    p = r.project(BytesIO(raw), patterns)
    indices = p.containers.get((container,), [])
    need(type(indices) is list and indices == list(range(len(indices))), 'contiguous selected-row array')
    expected = 24 if source['id'] == 'D' else 4
    need(len(indices) == expected, 'fixed report selected-row denominator')
    rows = []
    for i in sorted(indices):
        fields = []
        for key in FIELDS:
            path = (container, i, key)
            if path not in p.leaves:
                continue
            val = p.leaves[path]
            if key == 'root_id':
                need(type(val) is int and val >= 0, 'selected-row ordinal')
            else:
                a.hash_value(val)
            fields.append(citation(source['path'], source['sha256'], r.pointer(path), val))
        values = {x['pointer'].split('/')[-1]: x['value'] for x in fields}
        for key in ('root_id', 'root_sha256', 'goal_rgb_sha256'):
            need(key in values, 'required declared identity: '+key)
        need('history_rgb_sha256' in values or 'causal_history_rgb_sha256' in values,
             'whole history stack identity')
        rows.append(dict(row_pointer=r.pointer((container, i)), fields=fields,
            source_episode_identity=None, reset_attempt_identity=None,
            physical_state_identity=None, missing=list(ABSENT),
            ordinal_is_not_episode_id=True, history_hash_unit='whole_stack_not_individual_frames'))
    projected_config = [citation(source['path'], source['sha256'], r.pointer(k), v)
                        for k, v in sorted(p.leaves.items(), key=lambda x:r.pointer(x[0]))
                        if k[0] != container]
    for item in source['protocol_projection']:
        path = tuple(item['pointer'][1:].split('/'))
        need(path in p.leaves and p.leaves[path] == item['value'], 'catalog config join')
    return dict(selected_rows=rows, config_and_input_ancestry=projected_config,
                episode_mapping=None, original_attempt_count=None)


def static_python(raw):
    """Byte-bound AST facts, not an executed import graph or a reset bound."""
    text = raw.decode('utf-8'); tree = ast.parse(text)
    imports = []; calls = []; definitions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(dict(line=node.lineno, expression=ast.get_source_segment(text, node)))
        if isinstance(node, ast.Call):
            callee = ast.unparse(node.func)
            if any(t in callee.lower() for t in ('reset', 'collect', 'root_fingerprint')):
                calls.append(dict(line=node.lineno, expression=ast.get_source_segment(text, node)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                t in node.name for t in ('root', 'reset', 'fingerprint')):
            definitions.append(dict(name=node.name, first_line=node.lineno,
                last_line=node.end_lineno, text=ast.get_source_segment(text, node)))
    return dict(import_declarations=sorted(imports, key=lambda x:x['line']),
                reset_collector_identity_calls=sorted(calls, key=lambda x:x['line']),
                identity_related_definitions=sorted(definitions, key=lambda x:x['first_line']),
                executed=False, original_load_attested=False)


def selected_refs(authority, catalog):
    refs = [x for x in authority['history_file_requests'] if any(
        y['original_reason'] == 'historical_reset_attempt_bound_not_attested'
        for y in x['inherited_refs'])]
    need(len(refs) == 8 and sum(len(x['inherited_refs']) for x in refs) == 16,
         'exact eight source/sixteen inherited refs')
    by_path = {x['path']: x for x in refs}
    need(len(by_path) == 8 and [s['id'] for s in catalog['sources']] == list('ABCDEFGH'),
         'fixed A-H source order and unique paths')
    for s in catalog['sources']:
        need(s['path'] in by_path, 'source/request path join')
        row = by_path[s['path']]
        need(row['sha256'] == s['sha256'], 'source/request SHA join')
        need(sorted((x['parent'], x['historical_sha256']) for x in row['inherited_refs']) ==
             sorted((x['parent'], x['sha256']) for x in s['references']), 'inherited ref join')
    return by_path


def process_receipts(raw):
    """Use already-sealed saved disposition only; don't rescan logs/delta paths."""
    e = a.parse(raw)
    need(e['schema'] == 'agent_freshroots_resume_20261027_next.lossless-envelope.v1', 'delta envelope')
    z = zlib.decompressobj(31)
    decoded = z.decompress(base64.b64decode(e['gzip_base64'], validate=True), 16*1024**2+1)
    need(z.eof and not z.unused_data and not z.unconsumed_tail and len(decoded) <= 16*1024**2,
         'bounded single delta gzip')
    need(len(decoded) == e['semantic_bytes'] and sha(decoded) == e['semantic_sha256'], 'delta payload SHA')
    prefix = ('disposition', 'sources', '*', 'dispatch_status', '*')
    p = r.project(BytesIO(decoded), [prefix+(k,) for k in
        ('dispatch_call_id', 'dispatch_line', 'state', 'exit_code')])
    grouped = defaultdict(dict)
    for path, value in p.leaves.items():
        grouped[path[:-1]][path[-1]] = value
    return {v['dispatch_call_id']: dict(fields=v, envelope_payload_sha256=e['semantic_sha256'],
        envelope_pointer=r.pointer(path)) for path, v in grouped.items()}


def git_blob(descriptor):
    # Only the four literal catalog entries can reach this helper from build.
    need(descriptor.startswith('git:d26e4bff84f31138adf5d1bda809bccafd9be10a:le-wm-main/experiments/'),
         'fixed historical Git object')
    result = subprocess.run(['git', 'show', descriptor[4:]], cwd=ROOT,
                            check=True, capture_output=True, timeout=15)
    need(len(result.stdout) <= REPORT_CAP, 'Git blob cap')
    return result.stdout


def source_candidates(reader, catalog, blob_reader=git_blob):
    candidates = []; missing = []

    def add(path, expected, origin, original_path, manifest=None, pointer=None, python=True):
        try:
            raw = reader.read(path, expected)
        except FileNotFoundError:
            missing.append(dict(path=str(path), sha256=expected, origin=origin,
                                reason='explicit_source_snapshot_missing'))
            return
        item = dict(path=str(path), sha256=expected, bytes=len(raw), origin=origin,
                    original_path=original_path, original_load_attested=False,
                    original_dispatch_binding=None, manifest=manifest, pointer=pointer)
        if python:
            item['ast'] = static_python(raw)
        candidates.append(item)

    for name, digest in POSTRUN_FILES:
        add(str(ROOT/(POSTRUN+name)), digest, 'E_postrun_not_original_load',
            'experiments/spectral_reachability/'+name)
    for desc, key, label in ((HSEAL, 'inputs', 'H_downstream_prerun_not_H_model_load'),
                             (ARCHIVE, 'files', 'later_program_archive_not_A_H_load')):
        raw = reader.read(str(ROOT/desc[0]), desc[1]); manifest = a.parse(raw)
        need(len(manifest[key]) == 7, 'seven fixed sealed entries')
        for i, entry in enumerate(manifest[key]):
            if key == 'inputs':
                path = Path(entry['snapshot']); original = entry['path']
            else:
                path = (ROOT/desc[0]).parent/entry['archive']; original = entry['source']
            # Manifest-bound snapshot reads only, never follow original live imports.
            need(path.is_relative_to((ROOT/desc[0]).parent) and '..' not in path.parts,
                 'snapshot confined to explicit seal directory')
            add(str(path), entry['sha256'], label, original,
                dict(path=str(ROOT/desc[0]), sha256=desc[1]), '/'+key+'/'+str(i),
                python=not original.endswith('.json'))
    need(len(catalog['historical_chain']) == 4, 'four historical factory blobs')
    for entry in catalog['historical_chain']:
        try:
            raw = blob_reader(entry['path'])
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(dict(path=entry['path'], sha256=entry['sha256'],
                                reason='fixed_Git_blob_unavailable')); continue
        need(sha(raw) == entry['sha256'], 'historical Git blob SHA')
        candidates.append(dict(path=entry['path'], sha256=entry['sha256'], bytes=len(raw),
            original_path=entry['path'].split(':le-wm-main/')[-1],
            origin='historical_factory_family_not_original_load', ast=static_python(raw),
            original_load_attested=False, original_dispatch_binding=None))
    return candidates, missing


def closure_rows(catalog, candidates, receipts):
    result = []
    factories = [x['path'].split(':le-wm-main/')[-1] for x in catalog['historical_chain']]
    for i, d in enumerate(catalog['recovery']['dispatches']):
        roles = []
        for role in sorted(a.load_roles(d)):
            expected = None; path = None
            if role == 'producer':
                expected = d['recovered_producer_sha256']; path = d['producer']
            elif role == 'p0_dependency':
                expected = d['replayed_P0_dependency_sha256']
                path = 'experiments/spectral_reachability/preflight_pusht_d2p_p0.py'
            elif role == 'shell_launcher':
                expected = d['shell_recovered_sha256']
                path = d['command'].split()[1]
            elif role.startswith('factory_'):
                path = factories[int(role[-1])-1]
            available = [dict(path=c['path'], sha256=c['sha256'], origin=c['origin'])
                         for c in candidates if path is not None and
                         c['original_path'].endswith(path) and
                         (expected is None or c['sha256'] == expected)]
            roles.append(dict(role=role, declared_path=path, recovered_candidate_sha256=expected,
                available_byte_candidates=available, original_loaded_path=None,
                original_loaded_sha256=None, original_load_attested=False,
                missing=['original_process_load_binding'],
                candidate_bytes_available=bool(available)))
        result.append(dict(source_id=d['source_id'], call_id=d['call_id'],
            catalog_pointer='/recovery/dispatches/'+str(i),
            dispatch_intent={k:d[k] for k in ('command', 'producer', 'log', 'line', 'timestamp')},
            recovered_static_sites=d['recovered_producer_metadata'],
            saved_process_disposition=receipts.get(d['call_id']),
            process_return_is_not_reset_or_load_attestation=True, roles=roles,
            attempts_including_rejected_failed_restarts=None,
            original_load_closure_complete=False))
    need(len(result) == 10, 'ten original dispatch intents')
    return result


def build():
    reader = r.Reader()
    def read(desc):
        return reader.read(str(ROOT/desc[0]), desc[1])
    for filename, digest in DEPENDENCIES:
        read(('experiments/spectral_reachability/'+filename, digest))
    authority = a.parse(read(AUTHORITY)); catalog = a.parse(read(CATALOG))
    a.requests(catalog)  # Existing exact eight-source/ten-dispatch contract.
    refs = selected_refs(authority, catalog)
    parent_prepares = {}
    for sid, path, digest in r.PINNED:
        if sid in ('reachable_prepare', 'moving_prepare'):
            projected = r.unpack_prepare(read((path, digest)))
            parent_prepares[sid] = (projected, dict(path=str(ROOT/path), sha256=digest))
    receipts = process_receipts(read(DELTA))
    candidates, missing = source_candidates(reader, catalog)
    sources = []; groups = defaultdict(list)
    for source in catalog['sources']:
        raw = reader.read(source['path'], source['sha256'])
        need(len(raw) <= REPORT_CAP, 'original report cap')
        row = identity_projection(raw, source)
        row.update(source_id=source['id'], report=dict(path=source['path'], sha256=source['sha256']),
                   inherited_refs=refs[source['path']]['inherited_refs'],
                   inherited_reset_bound_status='unknown_not_cleared')
        row['inherited_refs'] = []
        for ref in refs[source['path']]['inherited_refs']:
            projected, parent = parent_prepares[ref['parent']]
            matches = [v for p, v in projected.leaves.items() if r.pointer(p) == ref['pointer']]
            need(matches == [source['path']] and projected.leaves.get(
                ('exclusions', 'files', source['path'])) == source['sha256'], 'original prepare ref bytes join')
            row['inherited_refs'].append(dict(**ref, parent_evidence=parent,
                pointer_domain='decompressed_original_prepare_payload',
                resolution_status='source_bytes_joined_reset_history_unknown'))
        sources.append(row)
        for selected in row['selected_rows']:
            root = next(f for f in selected['fields'] if f['pointer'].endswith('/root_sha256'))
            groups[root['value']].append(dict(source_id=source['id'], **root))
    implementations = []
    for filename in ('experiments/spectral_reachability/'+NAME+'.py',
                     'tests/test_'+NAME+'.py', 'innerdocs/'+NAME+'.md'):
        p = ROOT/filename; raw = reader.read(str(p))
        implementations.append(dict(path=str(p), sha256=sha(raw)))
    return dict(schema=NAME+'.report', status='source_identity_metadata_recovered_history_unknown',
        sources=sources, dispatch_load_closure=closure_rows(catalog, candidates, receipts),
        byte_verified_source_candidates=candidates, missing_snapshot_files=missing,
        declared_hash_groups=[dict(root_sha256=h, occurrences=rows)
                              for h, rows in sorted(groups.items())],
        summary=dict(original_sources=8, inherited_references=16, dispatch_intents=10,
                     selected_identity_rows=sum(len(s['selected_rows']) for s in sources),
                     distinct_declared_root_hashes=len(groups), physical_unique_roots=None,
                     original_load_closures_attested=0, inherited_reset_refs_cleared=0),
        exact_remaining_inputs=[
            'original attempt/episode-to-selected-row mapping with state/history identities',
            'exhaustive reset ledger including unselected/rejected/failed/restarted attempts OR original-loaded complete closure plus exhaustive envelope',
            'original producer/P0/four factories/installed environment/wrappers/shell run-bound bytes',
            'independent complete historical machine/worktree/time/unsaved-failure authority and subsequent deltas'],
        global_history_complete=None, MAIN_coverage_approval=False, freshness_certified=False,
        P1_admitted=False, scientific_admission=False, source_run_authorized=False,
        neural_calls=0, physics_calls=0, outcome_values_decoded=False,
        implementation_bindings=implementations, evidence_files=sorted(reader.files.values(), key=lambda x:x['path']))


def write_new(path, value):
    path = Path(path).absolute()
    need(path.parent.resolve() == ROOT/'results' and path.name.startswith(NAME+'_'),
         'exclusive new result stem in results/')
    raw = encode(value)+b'\n'; need(len(raw) <= OUTPUT_CAP, 'output cap')
    need(shutil.disk_usage(path.parent).free >= FLOOR+len(raw), '200MiB disk floor')
    with path.open('xb') as out:
        out.write(raw)
    return sha(raw)


def verify(path, expected):
    a.hash_value(expected); path = Path(path)
    need(path.stat().st_size <= OUTPUT_CAP, 'saved output cap')
    raw = path.read_bytes(); need(sha(raw) == expected, 'saved report SHA')
    need(raw == encode(build())+b'\n', 'full saved metadata rebuild')
    return dict(status='all_fixed_metadata_rebuilt_not_freshness', report_sha256=expected,
                P1_admitted=False, freshness_certified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('recover'); q.add_argument('--output', required=True)
    q = sub.add_parser('verify'); q.add_argument('--report', required=True); q.add_argument('--report-sha256', required=True)
    args = p.parse_args()
    if args.command == 'recover':
        value = build(); print(encode(dict(path=args.output, sha256=write_new(args.output, value),
                                           summary=value['summary'], status=value['status'])).decode())
    else:
        print(encode(verify(args.report, args.report_sha256)).decode())


if __name__ == '__main__':
    main()
