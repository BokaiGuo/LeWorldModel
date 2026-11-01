import hashlib
import io

import numpy as np
import pytest

from experiments.spectral_reachability import e3_cell03_encoder_parity_inputs_codex_20261030_v1 as m


def test_compare_exact_signed_zero_and_changed_values():
    a = {role: np.zeros(m.SHAPES[role], np.float32) for role in m.ROLES}
    b = {role: value.copy() for role, value in a.items()}
    b['latent'][0] = -0.
    b['encoder_hidden'].flat[1] += np.float32(.25)
    result = m.compare(a, b)
    assert not result['latent']['bitwise_equal']
    assert result['latent']['max_abs_error'] == 0
    assert result['encoder_hidden']['max_abs_error'] == .25
    assert result['projector_input']['bitwise_equal']


def test_compare_rejects_missing_shape_dtype_and_nan():
    a = {role: np.zeros(m.SHAPES[role], np.float32) for role in m.ROLES}
    with pytest.raises(ValueError, match='roles'):
        m.compare(a, {})
    for value in (np.zeros(192, np.float64), np.zeros(3, np.float32),
                  np.full(192, np.nan, np.float32)):
        b = dict(a, latent=value)
        with pytest.raises(ValueError):
            m.compare(a, b)


def test_saved_array_hash_and_trailing_bytes(tmp_path):
    stream = io.BytesIO()
    np.save(stream, np.ones(2, dtype=np.float32), allow_pickle=False)
    data = stream.getvalue()
    path = tmp_path / 'a.npy'
    path.write_bytes(data)
    descriptor = dict(file='a.npy', sha256=hashlib.sha256(data).hexdigest(),
                      dtype='<f4', shape=[2])
    assert m.checked_array(tmp_path, descriptor).tolist() == [1., 1.]
    path.write_bytes(data + b'x')
    with pytest.raises(ValueError, match='SHA'):
        m.checked_array(tmp_path, descriptor)
    descriptor['sha256'] = hashlib.sha256(data + b'x').hexdigest()
    with pytest.raises(ValueError, match='encoding'):
        m.checked_array(tmp_path, descriptor)
    descriptor['file'] = '../a.npy'
    with pytest.raises(ValueError, match='nonflat'):
        m.checked_array(tmp_path, descriptor)
