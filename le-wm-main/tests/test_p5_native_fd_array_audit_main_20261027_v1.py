import numpy as np
import pytest
from experiments.spectral_reachability import p5_native_fd_array_audit_main_20261027_v1 as a


@pytest.mark.parametrize('epsilon', [.01, .005])
def test_central_uses_returned_float32_values(epsilon):
    plus = np.array([1.1, 2.2], np.float32)
    minus = np.array([.9, 1.8], np.float32)
    assert np.array_equal(a.central(plus, minus, epsilon), (plus.astype(float)-minus.astype(float))/(2*epsilon))


@pytest.mark.parametrize('plus,minus,epsilon', [
    (np.zeros(2), np.zeros(2), .01),
    (np.zeros(2, np.float32), np.zeros(3, np.float32), .01),
    (np.zeros(2, np.float32), np.zeros(2, np.float32), .1),
    (np.array([np.nan], np.float32), np.zeros(1, np.float32), .01),
])
def test_central_rejects_invalid(plus, minus, epsilon):
    with pytest.raises(ValueError):
        a.central(plus, minus, epsilon)


def test_measure_zero_not_false_perfect_match():
    m = a.measures(np.zeros(2), np.zeros(2))
    assert m['relative_error_to_jvp'] is None and m['cosine'] is None


def test_measure_perfect_and_opposite():
    v = np.array([1., -2.])
    assert a.measures(v, v)['relative_error_to_jvp'] == 0
    assert np.isclose(a.measures(v, -v)['cosine'], -1)


def test_duplicate_json_rejected(tmp_path):
    p = tmp_path/'bad.json'
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError):
        a.read(p)
