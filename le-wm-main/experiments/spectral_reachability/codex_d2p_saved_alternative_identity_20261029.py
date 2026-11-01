"""Verify known saved source identities only; never collect or certify freshness."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability.root_exclusion_authority_resolution_20261027_v1 import journal_roots
from experiments.spectral_reachability.moving_root_exclusion_manifest_20261027_v1 import state_identity, same_pose

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'results/chat_freshroots_20261027_root_manifest.json'
MANIFEST_SHA = '3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838'
SOURCE_SHA = {
    'reachable_source': '9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9',
    'moving_source': 'c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6',
}


def load(path, sha, cap=8 * 1024 * 1024):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(ROOT / 'results'):
        raise ValueError('path outside fixed saved results')
    raw = path.read_bytes()
    if len(raw) > cap or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('file SHA/size mismatch: ' + str(path))
    return raw


def audit():
    manifest = json.loads(load(MANIFEST, MANIFEST_SHA))
    records = manifest['observed_historical_roots']
    if len(records) != 4:
        raise ValueError('four recorded identities required')
    observations, actions, sourcepacks = {}, {}, {}
    for family in ('reachable_source', 'moving_source'):
        group = [r for r in records if r['metadata']['record_id'].startswith(family + '_root')]
        if len(group) != 2:
            raise ValueError('two recorded source rows required')
        journal = group[0]['evidence']['journal']
        if any(r['evidence']['journal'] != journal for r in group):
            raise ValueError('journal mismatch')
        path = Path(journal['path'])
        src = json.loads(load(path.parent / 'sourcepack.json', SOURCE_SHA[family]))
        if src['sourcepack_complete'] is not True or src['journal_chain_tail'] != group[0]['evidence']['journal_tail']:
            raise ValueError('sourcepack/tail not complete')
        raw = load(path, journal['sha256'])
        if len(raw) != journal['bytes']:
            raise ValueError('journal length mismatch')
        observations[family], actions[family] = journal_roots(raw, src['journal_chain_tail'])
        sourcepacks[family] = {'path': str(path.parent / 'sourcepack.json'), 'sha256': SOURCE_SHA[family],
                               'global_freshness': src['global_freshness'],
                               'new_independent_confirmation': src['new_independent_confirmation'],
                               'journal': journal, 'journal_chain_tail': src['journal_chain_tail']}
    rows = []
    for record in records:
        m, e = record['metadata'], record['evidence']
        family, slot_str = m['record_id'].rsplit('_root', 1)
        slot = int(slot_str)
        checkpoints = []
        for step, declared in zip((0, 5, 10), e['checkpoints'], strict=True):
            observed = observations[family][slot, step]
            if observed != declared:
                raise ValueError('journal checkpoint does not match manifest')
            png = Path(e['journal']['path']).parent / observed['png']
            raw = load(png, observed['png_sha256'], 1024 * 1024)
            with Image.open(io.BytesIO(raw)) as image:
                pixels = np.asarray(image)
            if pixels.dtype != np.uint8 or pixels.shape != (224, 224, 3):
                raise ValueError('unexpected PNG shape or dtype')
            if hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest() != observed['pixel_sha256']:
                raise ValueError('PNG pixel hash mismatch')
            checkpoints.append({'step': step, 'journal_index': observed['journal_index'], 'event_sha256': observed['event_sha256'],
                                'path': str(png), 'file_sha256': observed['png_sha256'], 'pixel_sha256': observed['pixel_sha256']})
        identity = state_identity(observed['state'])
        if any(identity[k] != m[k] for k in ('root_state_sha256', 'physical_pose_key', 'physical_pose')):
            raise ValueError('state/pose mismatch')
        snap = hashlib.sha256(json.dumps(observed['snapshot'], sort_keys=True, allow_nan=False).encode()).hexdigest()
        action_rows = [actions[family][slot, step] for step in range(1, 11)]
        action = np.asarray([a['raw'] for a in action_rows], dtype='<f4')
        if not all(a['completed'] for a in action_rows) or hashlib.sha256(action.tobytes()).hexdigest() != m['raw_history_sha256']:
            raise ValueError('raw action history mismatch')
        if snap != m['root_snapshot_sha256'] or [x['pixel_sha256'] for x in checkpoints] != m['history_pixels_sha256']:
            raise ValueError('snapshot/history mismatch')
        rows.append({'record_id': m['record_id'], 'root_seed': m['root_seed'], 'reset_seed': m['reset_seed'],
                     'root_state': m['root_state'], 'physical_pose': m['physical_pose'],
                     'root_state_sha256': m['root_state_sha256'], 'physical_pose_key': m['physical_pose_key'],
                     'root_snapshot_sha256': m['root_snapshot_sha256'], 'history_pixels_sha256': m['history_pixels_sha256'],
                     'raw_history_sha256': m['raw_history_sha256'], 'checkpoints': checkpoints})
    pairs = [{'left': a['record_id'], 'right': b['record_id'], 'same_seed_pair': (a['root_seed'], a['reset_seed']) == (b['root_seed'], b['reset_seed']),
              'same_pose': bool(same_pose(a['physical_pose'], b['physical_pose']))}
             for i, a in enumerate(rows) for b in rows[i + 1:]]
    return {'schema': 'codex_d2p_saved_alternative_identity_20261029.v1',
            'manifest': {'path': str(MANIFEST), 'sha256': MANIFEST_SHA}, 'sources': sourcepacks, 'rows': rows, 'pairs': pairs,
            'independent_fresh_cohort_proved': False, 'history_complete': False, 'new_source_authorized': False,
            'reserved_p4_roots_opened': False, 'neural_calls': 0, 'physics_calls': 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = (json.dumps(audit(), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as out:
        out.write(result)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({'path': str(args.output.resolve()), 'sha256': hashlib.sha256(result).hexdigest()}))


if __name__ == '__main__':
    main()
