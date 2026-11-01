import hashlib
import json

import h5py
import pytest

from experiments.spectral_reachability.cube_d2p_repair_root_admission import admit


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, key='runtime_pool'):
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'weights.pt').write_bytes(b'base test bytes')
    checkpoint = tmp_path / 'repair'
    checkpoint.mkdir()
    pool = tmp_path / 'pool.h5'
    with h5py.File(pool, 'w') as handle:
        handle.create_dataset('root_seed', data=[202613501, 202614000, 202614524])
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'status': 'pass', 'pool': {'sha256': _hash(pool)}}))
    (checkpoint / 'repair_manifest.json').write_text(json.dumps({
        'base_checkpoint': str(base), 'base_weights_sha256': _hash(base / 'weights.pt'),
        key: {'path': str(pool), 'receipt': str(receipt), 'sha256': _hash(pool), 'receipt_sha256': _hash(receipt)},
    }))
    return checkpoint


@pytest.mark.parametrize('key', ['runtime_pool', 'pool'])
def test_rejects_v5_overlapping_root_regression(tmp_path, key):
    checkpoint = _fixture(tmp_path, key=key)
    with pytest.raises(ValueError, match='overlap'):
        admit(checkpoint, list(range(202614000, 202614008)))


def test_admits_new_roots_and_reports_checked_pool(tmp_path):
    checkpoint = _fixture(tmp_path)
    result = admit(checkpoint, list(range(202618000, 202618008)))
    assert result['status'] == 'pass'
    assert result['checked_live_pools'][0]['pool_root_count'] == 3


def test_rejects_pool_tamper(tmp_path):
    checkpoint = _fixture(tmp_path)
    with h5py.File(tmp_path / 'pool.h5', 'r+') as handle:
        handle['root_seed'][0] = 202618000
    with pytest.raises(ValueError, match='changed'):
        admit(checkpoint, [202618000])
