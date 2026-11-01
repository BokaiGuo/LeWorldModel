"""Fail closed on reuse of any live training/validation root in repair ancestry.

This is a provenance check, not learned-model evidence.  It reads the actual
SHA-bound pool seeds (not just a claimed interval) before any P1 root opens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def admit(checkpoint: Path, seeds: list[int]) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('evaluation root seeds must be nonempty and unique')
    current = checkpoint.resolve()
    visited: set[Path] = set()
    pools: list[dict[str, Any]] = []
    while (current / 'repair_manifest.json').is_file():
        if current in visited:
            raise ValueError('cyclic repair checkpoint ancestry')
        visited.add(current)
        manifest = json.loads((current / 'repair_manifest.json').read_text())
        pool = manifest.get('runtime_pool') or manifest.get('pool')
        mode = manifest.get('training_contract', {}).get('mode')
        if pool is None and (mode in {'runtime_pool', 'same_root_ranking_pool'} or manifest.get('status') in {'trained_runtime_aligned_repair', 'trained_same_root_ranking_repair'}):
            raise ValueError('live repair ancestry lacks a root pool')
        if pool is not None:
            path = Path(pool['path']).resolve()
            receipt_path = Path(pool['receipt']).resolve()
            if _sha(path) != pool['sha256'] or _sha(receipt_path) != pool['receipt_sha256']:
                raise ValueError('repair pool bytes/receipt changed')
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('status') != 'pass' or receipt.get('pool', {}).get('sha256') != pool['sha256']:
                raise ValueError('repair pool receipt is not admissible')
            with h5py.File(path, 'r') as handle:
                values = np.asarray(handle['root_seed'], dtype=np.int64)
            if values.ndim != 1 or len(set(values.tolist())) != len(values):
                raise ValueError('repair pool seeds are malformed or duplicate')
            overlap = sorted(set(seeds) & set(values.tolist()))
            if overlap:
                raise ValueError(f'evaluation/training root seed overlap: {overlap}')
            pools.append({'checkpoint': str(current), 'pool': str(path), 'sha256': pool['sha256'], 'pool_root_count': len(values), 'evaluation_overlap': []})
        base = manifest.get('base_checkpoint')
        if not base:
            raise ValueError('repair checkpoint lacks base ancestry')
        current = Path(base).resolve()
        if not (current / 'weights.pt').is_file():
            raise ValueError('repair base checkpoint is unavailable')
        if manifest.get('base_weights_sha256') != _sha(current / 'weights.pt'):
            raise ValueError('repair base checkpoint weights changed')
    if not (current / 'weights.pt').is_file():
        raise ValueError('terminal base checkpoint is unavailable')
    return {'schema': 'cube_d2p_repair_root_admission.v1', 'status': 'pass', 'checkpoint': str(checkpoint.resolve()), 'evaluation_seeds': seeds, 'checked_live_pools': pools, 'claim_boundary': 'Seed exclusion from all SHA-bound live repair pools, including validation. Not proof of generalization or frozen D2P efficacy.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--roots', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.roots < 1 or args.output.exists():
        raise ValueError('positive root count and new admission output required')
    result = admit(args.checkpoint, list(range(args.seed, args.seed + args.roots)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
