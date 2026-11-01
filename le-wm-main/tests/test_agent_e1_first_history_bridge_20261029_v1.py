import hashlib
import io

import numpy as np
import pytest

from experiments.spectral_reachability import agent_e1_first_history_bridge_20261029_v1 as bridge


def sha(x):
    return hashlib.sha256(x).hexdigest()


def put(path, value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    data = stream.getvalue()
    path.write_bytes(data)
    return sha(data)


def test_actual_saved_subset():
    result = bridge.inspect()
    assert len(result['rows']) == 4
    assert all(x['target_physical_time'] is None for x in result['rows'])
    assert not result['history3_delivery_executed'] and result['scaler_authority'] is None


def test_history_requires_exact_last_frame_and_keeps_no_authority(tmp_path):
    z = np.arange(576, dtype=np.float32).reshape(1, 3, 192)
    a = np.arange(20, dtype=np.float32).reshape(1, 2, 10)
    zp, ap = tmp_path / 'z.npy', tmp_path / 'a.npy'
    zh, ah = put(zp, z), put(ap, a)
    result = bridge.check_history(zp, zh, ap, ah, sha(z[:, -1, :].tobytes()))
    assert result['shape_and_last_frame_match'] and not result['admission']
    with pytest.raises(ValueError):
        bridge.check_history(zp, zh, ap, ah, '0' * 64)
    with pytest.raises(ValueError):
        bridge.check_history(zp, '0' * 64, ap, ah, sha(z[:, -1, :].tobytes()))


@pytest.mark.parametrize('kind', ['shape', 'dtype', 'nan', 'trailing'])
def test_array_rejects_invalid(tmp_path, kind):
    x = np.zeros((1, 3, 192), dtype=np.float32)
    if kind == 'shape': x = x[:, :1]
    if kind == 'dtype': x = x.astype(np.float64)
    if kind == 'nan': x[0, 0, 0] = np.nan
    path = tmp_path / 'array.npy'
    h = put(path, x)
    if kind == 'trailing':
        path.write_bytes(path.read_bytes() + b'x')
        h = sha(path.read_bytes())
    with pytest.raises(ValueError):
        bridge.array(path, h, (1, 3, 192))
