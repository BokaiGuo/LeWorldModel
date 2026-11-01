import numpy as np
import pytest
from experiments.spectral_reachability import main_observed_fd_scalar_cause_20261028_v1 as m


def test_quadratic_identity_random_rectangular_metric():
    rng = np.random.default_rng(2810)
    for _ in range(40):
        v = [rng.normal(size=7).tolist() for i in range(5)]
        r = m.scalar_cause(*v, rng.normal(size=(3, 7)).tolist(), .02)
        assert abs(r['scalar_identity_residual']) < 1e-10


def test_common_offset_can_reverse_goal_gradient_with_identical_response():
    r = m.scalar_cause([1.], [3.], [-3.], [-1.], [0.], [[1.]], 2.)
    assert r['strict_direction_reversal']
    assert r['terms']['response'] == r['terms']['interaction'] == 0
    assert r['terms']['midpoint_bias'] == 8


def test_correct_midpoint_does_not_imply_correct_response():
    r = m.scalar_cause([1.], [3.], [3.], [1.], [0.], [[1.]], 2.)
    assert r['strict_direction_reversal']
    assert r['terms']['midpoint_bias'] == r['terms']['interaction'] == 0
    assert r['terms']['response'] == 8


def test_zero_physical_response_is_not_reversal():
    r = m.scalar_cause([1.], [3.], [2.], [2.], [0.], [[1.]], 2.)
    assert r['actual_secant_norm'] == r['actual_scalar_fd'] == 0
    assert not r['strict_direction_reversal']


@pytest.mark.parametrize('span', [0, -1, True, float('nan')])
def test_invalid_span(span):
    with pytest.raises(ValueError):
        m.scalar_cause([1.], [3.], [3.], [1.], [0.], [[1.]], span)


def test_bool_and_nonfinite_arrays_rejected():
    for value in ([True], [float('inf')], ['1']):
        with pytest.raises(ValueError):
            m.array(value)


def test_saved_actual_reports_all32_no_neural():
    r = m.build()
    assert r['comparisons'] == 32 and len(r['rows']) == 8
    assert r['new_neural_calls'] == r['new_physics_calls'] == 0
    assert r['P1_admission'] is False
