"""Bounded, saved-only index; not a history certificate or freshness decision."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability.moving_root_exclusion_manifest_20261027_v1 import state_identity

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
INPUTS = {
    'identity': ('agent_p4_historical_source_identity_20261029_v1_LOCAL01.json', '577580e536598d850cd449fda2a02c314197267bdc41bff9d85692432165e724'),
    'prefix': ('chat_freshroots_20261027_prefix_identity_supplement.json', '2cd81f5741fbef9fea0eae62b913e9ad73b09fa8c9d02e525876d6854a91d7e9'),
    'quarantine': ('moving_historical_reset_quarantine_20261027_v1_saved01.json', 'fe27d07b749c8a518faee2667f2c256c97fd94ea80d5ae25347560ec72bf3b70'),
    'coverage': ('agent_freshroots_resume_20261027_coverage.json', '00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb'),
}


def read_checked(path, expected):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(RESULTS):
        raise ValueError('noncanonical or out-of-scope path')
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('size or SHA mismatch: ' + str(path))
    return json.loads(raw)


def pointer(obj, ref):
    if not ref.startswith('/'):
        raise ValueError('invalid pointer')
    for token in ref[1:].split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        obj = obj[int(token)] if isinstance(obj, list) else obj[token]
    return obj


def verify_dispatch_log(coverage):
    dispatches = coverage['recovery']['dispatches']
    binding = coverage['log_bindings'][0]
    if len(dispatches) != 10 or any(d['log'] != binding['path'] for d in dispatches):
        raise ValueError('unexpected dispatch scope')
    wanted = {d['line']: d for d in dispatches}
    if len(wanted) != 10:
        raise ValueError('duplicate dispatch line')
    found, digest, consumed = [], hashlib.sha256(), 0
    with open(binding['path'], 'rb') as stream:
        line_no = 0
        while consumed < binding['bytes']:
            line = stream.readline(min(8 * 1024 * 1024 + 1, binding['bytes'] - consumed))
            if not line or len(line) > 8 * 1024 * 1024 or not line.endswith(b'\n'):
                raise ValueError('truncated or oversized log line')
            consumed += len(line)
            line_no += 1
            digest.update(line)
            if line_no in wanted:
                expected = wanted[line_no]
                obj = json.loads(line)
                if obj.get('payload', {}).get('call_id') != expected['call_id'] or obj.get('timestamp') != expected['timestamp']:
                    raise ValueError('dispatch log mismatch')
                found.append({'source_id': expected['source_id'], 'call_id': expected['call_id'], 'line': line_no,
                              'line_sha256': hashlib.sha256(line).hexdigest(), 'timestamp': expected['timestamp'],
                              'intent_only_not_execution_or_reset_ledger': True})
    if digest.hexdigest() != binding['sha256'] or len(found) != 10:
        raise ValueError('original log prefix SHA or denominator mismatch')
    return {'path': binding['path'], 'bytes': binding['bytes'], 'sha256': binding['sha256'], 'dispatches': found}


def build():
    inputs = {k: read_checked(RESULTS / name, sha) for k, (name, sha) in INPUTS.items()}
    identity, prefix, quarantine = (inputs[k] for k in ('identity', 'prefix', 'quarantine'))
    if len(identity['sources']) != 8 or len(prefix['records']) != 80 or len(quarantine['rows']) != 8:
        raise ValueError('input denominator changed')
    cache = {}

    def source(ref):
        key = (ref['path'], ref['sha256'])
        if key not in cache:
            cache[key] = read_checked(*key)
        return pointer(cache[key], ref['pointer'])

    observed = []
    for record in prefix['records']:
        state = source(record['evidence'])
        computed = state_identity(state)
        if list(state) != record['state7'] or any(computed[k] != record[k] for k in ('root_state_sha256', 'physical_pose_key', 'physical_pose')):
            raise ValueError('original prefix state mismatch')
        observed.append({k: record[k] for k in ('evidence', 'state7', 'root_state_sha256', 'physical_pose', 'physical_pose_key', 'unresolved')})

    selected = []
    for item in identity['sources']:
        for row in item['selected_rows']:
            fields = row['fields']
            for field in fields:
                if source(field) != field['value']:
                    raise ValueError('original selected field mismatch')
            selected.append({'source_id': item['source_id'], 'row_pointer': row['row_pointer'], 'fields': fields,
                             'physical_state_identity': None, 'reset_attempt_identity': None})
    if len(selected) != 52:
        raise ValueError('selected denominator changed')
    conditional = []
    for item in quarantine['rows']:
        for env in item['conditional_envelopes']:
            if env['certified'] is not False or env['classification'] != 'conditional_code_family_envelope':
                raise ValueError('unexpected envelope authority')
            conditional.append({'source_id': item['source_id'], 'report_path': item['path'], 'report_sha256': item['sha256'],
                                'kind': env['kind'], 'start_inclusive': env['start'], 'end_inclusive': env['end'],
                                'assumptions': env['assumptions'], 'unknown': env['unknown'], 'certified': False})
    dispatch_log = verify_dispatch_log(inputs['coverage'])
    return {'schema': 'codex_d2p_historical_exclusion_index_20261029.v1',
            'inputs': {k: {'path': str(RESULTS / n), 'sha256': h} for k, (n, h) in INPUTS.items()},
            'counts': {'original_files_verified': len(cache), 'observed_prefix_records': len(observed),
                       'selected_records': len(selected), 'conditional_envelopes': len(conditional)},
            'observed_prefix_records': observed, 'selected_declared_records': selected,
            'conditional_seed_envelopes': conditional,
            'original_dispatch_log': dispatch_log,
            'limits': ['observed prefixes are not necessarily sampling roots', 'selected hashes include goal and are not physical pose identities',
                       'conditional seed envelopes are not attested attempt ledgers', 'historical scope and unselected or failed attempts remain unknown'],
            'history_complete': False, 'freshness_certified': False, 'source_run_authorized': False,
            'original_reset_bounds_cleared': 0, 'neural_calls': 0, 'physics_calls': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = (json.dumps(build(), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({'path': str(args.output.resolve()), 'sha256': hashlib.sha256(data).hexdigest()}))


if __name__ == '__main__':
    main()
