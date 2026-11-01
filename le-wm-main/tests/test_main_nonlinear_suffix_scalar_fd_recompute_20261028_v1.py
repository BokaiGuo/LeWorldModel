import base64
import hashlib
import numpy as np
import pytest
from experiments.spectral_reachability import main_nonlinear_suffix_scalar_fd_recompute_20261028_v1 as m


def blob(x):
    data = np.asarray([x], dtype='<f4').tobytes()
    return dict(shape=[1], dtype='<f4', sha256=hashlib.sha256(data).hexdigest(),
                data=base64.b64encode(data).decode())


def test_original_native_loss_bytes():
    assert m.loss(blob(.125)) == .125


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'sha', 'negative', 'nan', 'extra'])
def test_false_loss_rejected(bad):
    b = blob(1)
    if bad == 'shape': b['shape'] = []
    elif bad == 'dtype': b['dtype'] = '<f8'
    elif bad == 'sha': b['sha256'] = '0'*64
    elif bad == 'negative': b = blob(-1)
    elif bad == 'nan': b = blob(float('nan'))
    else: b['extra'] = 1
    with pytest.raises(ValueError): m.loss(b)


def test_complete_vector_and_zero_norm():
    r = m.vector_metrics([1.]*10, np.ones(10))
    assert r['relative_L2'] == 0 and r['cosine'] == pytest.approx(1)
    r = m.vector_metrics([0.]*10, np.zeros(10))
    assert r['relative_L2'] is r['cosine'] is None


def test_missing_pair_never_summarizes_survivors():
    assert m.vector_metrics([None]+[1.]*9, np.ones(10)) is None
    with pytest.raises(ValueError): m.vector_metrics([1.]*9, np.ones(10))
