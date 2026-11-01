import numpy as np
import pytest
from experiments.spectral_reachability import main_raw_anchor_factorial_20261028_v1 as m


def test_pure_common_offset_is_repaired_only_by_oracle_anchor():
    one = np.ones(192)
    f = m.factorial(3*one, 4*one, -2*one, -one, 3*one, -2*one, 0*one, .02)
    assert f['signs']['MM'] == 'opposite'
    assert f['signs']['AM'] == 'same'
    assert f['baseline_response_effect'] == 0 and f['interaction'] == 0


def test_common_offset_subtraction_cannot_fix_reversed_response():
    one = np.ones(192)
    f = m.factorial(one, 2*one, one, 0*one, one, one, 0*one, .02)
    assert f['signs']['MM'] == f['signs']['AM'] == 'opposite'
    assert f['baseline_anchor_effect'] == f['interaction'] == 0


def test_zero_real_response_remains_tie_under_all_substitutions():
    one = np.ones(192)
    f = m.factorial(one, 2*one, one, one, one, one, 0*one, .002)
    assert set(f['signs'].values()) == {'actual_tie'}


def test_random_factorial_identity_and_no_hidden_axis_collapse():
    rng = np.random.default_rng(281028)
    for _ in range(20):
        f = m.factorial(*rng.normal(size=(7, 192)), .002)
        s = f['secants']
        assert s['MM']-s['AA'] == pytest.approx(f['baseline_anchor_effect']+
            f['baseline_response_effect']+f['interaction'], abs=1e-9)
        assert abs(f['interaction_identity_residual']) < 1e-9


@pytest.mark.parametrize('span', [0, -1, float('nan')])
def test_bad_span_rejected(span):
    with pytest.raises(ValueError):
        m.factorial(*np.zeros((7,192)), span)


def test_nonzero_actual_with_zero_substituted_response_is_not_success():
    assert m.label(0., 1.) == 'substituted_tie'


def test_dimension_or_nonfinite_is_rejected():
    bad = np.zeros((7, 191))
    with pytest.raises(ValueError):
        m.factorial(*bad, .02)
    bad = np.zeros((7, 192)); bad[2, 0] = float('nan')
    with pytest.raises(ValueError):
        m.factorial(*bad, .02)
