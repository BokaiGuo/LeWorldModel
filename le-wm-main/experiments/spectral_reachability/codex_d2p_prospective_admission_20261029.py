"""Conservative metadata screen; never authorizes collection or certifies freshness."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from experiments.spectral_reachability.moving_root_exclusion_manifest_20261027_v1 import same_pose, state_identity

PROJECT = Path(__file__).resolve().parents[3]
INPUTS = {
    'prereg': ('docs/chat_freshroots_20261027_preregistration.json', '79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e'),
    'index': ('le-wm-main/results/codex_d2p_historical_exclusion_index_20261029_LOCAL01.json', 'ac2544512d033ec14fd07808c3abe2dfd96676bb6b82ec89569f498f58256f98'),
    'manifest': ('le-wm-main/results/chat_freshroots_20261027_root_manifest.json', '3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838'),
}
IDENTITY_FIELDS = {'state7', 'snapshot_sha256', 'rgb3_sha256', 'raw_history_sha256', 'root_goal_sha256'}


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key')
        obj[key] = value
    return obj


def read(path, expected):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_file():
        raise ValueError('noncanonical input path')
    raw = path.read_bytes()
    if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('input size or sha256 mismatch')
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def hash_check(value):
    if not isinstance(value, str) or len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
        raise ValueError('expected lowercase SHA256')


def validate_candidate(value, slot):
    if value is None:
        return None
    if set(value) != {'slot', 'root_seed', 'reset_seed', 'identity'}:
        raise ValueError('candidate fields must exclude outcome and selection data')
    for k in ('slot', 'root_seed', 'reset_seed'):
        if type(value[k]) is not int or value[k] != slot[k]:
            raise ValueError('candidate slot or seed does not match preregistration')
    identity = value['identity']
    if not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS:
        raise ValueError('exact identity fields required')
    state = identity['state7']
    if not isinstance(state, list) or len(state) != 7 or any(type(x) not in (int, float) or not math.isfinite(x) for x in state):
        raise ValueError('finite numeric state7 required')
    for field in ('snapshot_sha256', 'raw_history_sha256'):
        hash_check(identity[field])
    if identity['root_goal_sha256'] is not None:
        hash_check(identity['root_goal_sha256'])
    if not isinstance(identity['rgb3_sha256'], list) or len(identity['rgb3_sha256']) != 3:
        raise ValueError('three RGB hashes required')
    for value_hash in identity['rgb3_sha256']:
        hash_check(value_hash)
    return {**identity, **state_identity(state)}


def screen(prereg, index, manifest, candidates):
    expected = prereg['candidate_order']
    if prereg['sampling_denominator'] != 4 or len(expected) != 4 or prereg['replacement_or_retry'] is not False:
        raise ValueError('unexpected preregistration')
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError('four slots including nulls required')
    identities = [validate_candidate(candidate, slot) for candidate, slot in zip(candidates, expected)]
    historical = [r['physical_pose'] for r in index['observed_prefix_records']]
    roots = [r['metadata'] for r in manifest['observed_historical_roots']]
    historical += [r['physical_pose'] for r in roots if r.get('physical_pose') is not None]
    used = {r['seed'] for r in manifest['used_or_reserved_seed_evidence']}
    spans = [s['half_open'] for s in manifest['half_open_exclusion_spans']]
    envelopes = index['conditional_seed_envelopes']
    root_goals = {r['root_goal_sha256'] for r in manifest['root_goal_identity_index']}
    rows = []
    for i, (slot, identity) in enumerate(zip(expected, identities)):
        reasons = []
        for seed in (slot['root_seed'], slot['reset_seed']):
            if seed in used or any(lo <= seed < hi for lo, hi in spans) or any(e['start_inclusive'] <= seed <= e['end_inclusive'] for e in envelopes):
                reasons.append('known_or_conditional_seed_conflict')
        if identity is not None:
            if any(same_pose(identity['physical_pose'], p) for p in historical):
                reasons.append('known_historical_pose_collision')
            if any(other is not None and j != i and same_pose(identity['physical_pose'], other['physical_pose']) for j, other in enumerate(identities)):
                reasons.append('batch_pose_collision')
            if any(identity['snapshot_sha256'] == r.get('root_snapshot_sha256') for r in roots):
                reasons.append('known_snapshot_collision')
            if any(identity['rgb3_sha256'] == r.get('history_pixels_sha256') for r in roots):
                reasons.append('known_rgb3_collision')
            if identity['root_goal_sha256'] is not None and identity['root_goal_sha256'] in root_goals:
                reasons.append('known_root_goal_collision')
            if any(other is not None and j != i and (identity['snapshot_sha256'] == other['snapshot_sha256'] or identity['rgb3_sha256'] == other['rgb3_sha256']) for j, other in enumerate(identities)):
                reasons.append('batch_snapshot_or_rgb3_collision')
        rows.append({'slot': i, 'root_seed': slot['root_seed'], 'reset_seed': slot['reset_seed'],
                     'identity_provided': identity is not None, 'collision_reasons': sorted(set(reasons)),
                     'screening_status': 'blocked_collision' if reasons else ('blocked_no_actual_identity' if identity is None else 'no_known_collision_unverified'),
                     'admitted': False, 'missing': ['independent_pre_witness_identity_and_attempt_provenance', 'complete_historical_scope_and_reset_authority']})
    return {'schema': 'codex_d2p_prospective_admission_20261029.v1', 'sampling_denominator': 4,
            'rows': rows, 'historical_index_is_complete': False, 'freshness_certified': False,
            'source_run_authorized': False, 'witness_dispatch_authorized': False,
            'candidate_identity_input_is_untrusted': True, 'neural_calls': 0, 'physics_calls': 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', type=Path)
    p.add_argument('--candidates-sha256')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if (a.candidates is None) != (a.candidates_sha256 is None):
        p.error('candidate path and SHA must be supplied together')
    inputs = {k: read(PROJECT / path, sha) for k, (path, sha) in INPUTS.items()}
    candidates = [None] * 4
    if a.candidates:
        doc = read(a.candidates, a.candidates_sha256)
        if set(doc) != {'schema', 'slots'} or doc['schema'] != 'codex_d2p_prospective_identity.v1':
            raise ValueError('strict candidate input schema')
        candidates = doc['slots']
    report = screen(inputs['prereg'], inputs['index'], inputs['manifest'], candidates)
    report['input_bindings'] = {k: {'path': str(PROJECT / path), 'sha256': sha} for k, (path, sha) in INPUTS.items()}
    report['candidate_binding'] = None if a.candidates is None else {'path': str(a.candidates), 'sha256': a.candidates_sha256}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with a.output.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({'path': str(a.output.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}))


if __name__ == '__main__':
    main()
