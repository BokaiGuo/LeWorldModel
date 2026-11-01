import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_native_store_20261029_v1 as m


def test_real_frame_and_journal_chain(tmp_path):
    store = m.NativeStore(tmp_path / 'run')
    a = np.zeros((224, 224, 3), dtype=np.uint8)
    digest = store.frame('actual.png', a)
    store.emit('fresh_observation', receipt={'png': 'actual.png', 'file_sha256': digest})
    assert m.inspect_capture_journal(store.path)['observation_pngs'] == 1
    (store.path / 'actual.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='PNG hash'):
        m.inspect_capture_journal(store.path)


def test_invalid_array_saved_to_failure_reserve(tmp_path):
    store = m.NativeStore(tmp_path / 'run')
    with pytest.raises(ValueError, match='preserved'):
        store.arrays('bad_invalid.npz', raw_render=np.zeros((3, 4)))
    data = (store.path / 'failure.json').read_bytes()
    assert json.loads(data)['requested_name'] == 'bad_invalid.npz'


def test_oversize_and_duplicate_reject(tmp_path):
    store = m.NativeStore(tmp_path / 'run')
    with pytest.raises(ValueError, match='uint8'):
        store.frame('bad.png', np.zeros((3, 4)))
    a = np.zeros((224, 224, 3), dtype=np.uint8)
    store.frame('a.png', a)
    with pytest.raises(ValueError, match='quota'):
        store.frame('a.png', a)
    with pytest.raises(ValueError, match='store fault'):
        store.check()
