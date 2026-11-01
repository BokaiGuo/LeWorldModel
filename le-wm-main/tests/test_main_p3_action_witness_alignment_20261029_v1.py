from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import main_p3_action_witness_alignment_20261029_v1 as m


def zeros(n):
    return np.zeros((n, 2), np.float32).tolist()


def test_full_same_action_H3_is_possible_despite_feedback():
    a = m.align(zeros(15), zeros(20), 5, 3)
    assert a['exact_full_action_match'] is True and a['first_differing_control_step'] is None


def test_feedback_action_change_is_not_endpoint_fidelity():
    e = zeros(20); e[10][0] = .5
    a = m.align(zeros(15), e, 5, 3)
    assert a['full_action_witness_observed'] and a['exact_full_action_match'] is False
    assert a['first_differing_control_step'] == 11


def test_late_H3_no_terminal_substitute():
    a = m.align(zeros(15), zeros(20), 15, 3)
    assert a['compared_steps'] == 5 and a['missing_steps'] == 10
    assert a['exact_full_action_match'] is None


def test_earlystop_keeps_missing():
    a = m.align(zeros(5), zeros(8), 5, 1)
    assert a['missing_steps'] == 2 and a['exact_full_action_match'] is None


def test_signed_zero_is_real_action_bit_difference():
    e = zeros(20); e[5][0] = -0.
    assert m.align(zeros(5), e, 5, 1)['first_differing_control_step'] == 6


@pytest.mark.parametrize('value', [True, .1, float('nan'), 2.])
def test_illegal_raw_values(value):
    e = zeros(20); e[5][0] = value
    with pytest.raises(ValueError):
        m.align(zeros(5), e, 5, 1)


@pytest.mark.parametrize('b,h,n', [(0,1,5),(5,2,10),(10,3,8),(15,1,21)])
def test_shape_time_mutations(b,h,n):
    with pytest.raises(ValueError):
        m.align(zeros(5*h), zeros(n), b, h)


def test_inputs_unchanged():
    p, e = zeros(15), zeros(20); before = deepcopy((p,e))
    m.align(p,e,5,3)
    assert (p,e) == before
