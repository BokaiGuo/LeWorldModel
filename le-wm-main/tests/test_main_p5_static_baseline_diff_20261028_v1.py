import numpy as np
import pytest
from experiments.spectral_reachability import main_p5_static_baseline_diff_20261028_v1 as m


def test_equal():
    a = np.zeros((224,224,3), np.uint8)
    assert m.differences(a,a) == dict(changed_pixels=0,bbox=None,max_abs=0,pixels=[])


def test_signed_difference_no_uint8_wrap():
    a = np.zeros((224,224,3), np.uint8)
    b = a.copy(); b[5,7] = 255
    d = m.differences(a,b)
    assert d['max_abs'] == 255 and d['bbox'] == [7,5,7,5] and d['changed_pixels'] == 1


@pytest.mark.parametrize('array', [np.zeros((1,3),np.uint8),np.zeros((224,224,3),float)])
def test_bad_abi(array):
    with pytest.raises(ValueError):
        m.differences(array,np.zeros((224,224,3),np.uint8))


def test_actual_saved_failure_full_denominator():
    r = m.analyze()
    assert len(r['rows']) == 5
    assert r['rows'][0]['npy_bytes_equal'] is True
    assert r['rows'][1]['diff']['changed_pixels'] == 3
    assert r['rows'][1]['diff']['bbox'] == [88,120,88,122]
    assert all(not x['compared'] and x['diff'] is None for x in r['rows'][2:])
    assert r['scientific_admission'] is False and r['threshold_relaxed'] is False
