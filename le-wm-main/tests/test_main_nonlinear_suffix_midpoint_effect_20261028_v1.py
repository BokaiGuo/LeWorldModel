import numpy as np
import pytest

from experiments.spectral_reachability import main_nonlinear_suffix_midpoint_effect_20261028_v1 as m


def v(x):
    return np.full(192, x, np.float32)


def test_exact_prediction_has_zero_error_without_new_control_benefit():
    r = m.compare(v(3), v(1), v(1), v(0))
    assert r['imagined_endpoint_MSE'] == 4
    assert r['observed_mid_endpoint_MSE'] == 0
    assert r['signed_endpoint_terms'] == dict(cross=-8, response_square=4)
    assert r['endpoint_MSE_delta'] == -4
    assert r['absolute_Q_error_delta'] == -8
    assert not r['physical_benefit_claim']
    assert not r['ranking_repair_claim']


def test_observation_can_worsen_prediction_and_is_not_clamped():
    r = m.compare(v(1), v(3), v(0), v(0))
    assert r['endpoint_MSE_worse']
    assert r['endpoint_MSE_delta'] == 8
    assert r['absolute_Q_error_worse']


def test_endpoint_error_and_task_cost_error_are_different():
    r = m.compare(v(-1), v(0.5), v(1), v(0))
    assert r['endpoint_MSE_better']
    assert r['absolute_Q_error_worse']
    assert r['actual_Q'] == r['imagined_predicted_Q'] == 1


def test_identical_input_is_exact_tie():
    r = m.compare(v(1), v(1), v(0), v(2))
    assert r['endpoint_MSE_delta'] == r['absolute_Q_error_delta'] == 0
    assert not r['prediction_changed']
    assert not r['endpoint_MSE_better'] and not r['endpoint_MSE_worse']


@pytest.mark.parametrize('bad', [np.zeros(191), np.full(192, np.nan),
                               np.full(192, np.inf), np.full(192, 0.1, np.float64),
                               np.full(192, 1e100)])
def test_foreign_invalid_or_rounded_coordinates_refused(bad):
    with pytest.raises(ValueError):
        m.compare(bad, v(0), v(0), v(0))


def rows():
    return [dict(index=i, root_slot=i//2, condition=('off', 'on')[i%2],
                 **m.compare(v(1), v(0), v(0), v(0))) for i in range(4)]


def test_fixed_denominator():
    r = m.summarize(rows())
    assert r['endpoint_counts'] == dict(better=4, worse=0)
    assert r['protocol']['known_roots'] == 2
    assert not r['scientific_admission']


@pytest.mark.parametrize('which', ['drop', 'order', 'duplicate', 'benefit'])
def test_missing_reordered_duplicate_or_false_claim_refused(which):
    rs = rows()
    if which == 'drop':
        rs.pop()
    elif which == 'order':
        rs.reverse()
    elif which == 'duplicate':
        rs[3]['condition'] = 'off'
    else:
        rs[0]['physical_benefit_claim'] = True
    with pytest.raises(ValueError):
        m.summarize(rs)
