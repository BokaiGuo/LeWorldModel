import numpy as np

from experiments.audit_live_reacher_shared_candidate_ranking import (
    _pairwise_inversion,
    _wrap_angle,
)


def test_pairwise_inversion_is_zero_for_same_order_and_one_for_reverse_order():
    assert _pairwise_inversion(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0, 2.0])) == 0.0
    assert _pairwise_inversion(np.asarray([0.0, 1.0, 2.0]), np.asarray([2.0, 1.0, 0.0])) == 1.0


def test_reacher_angle_wrap_is_bounded():
    values = _wrap_angle(np.asarray([-4.0, 0.0, 4.0]))
    assert np.all(values >= -np.pi)
    assert np.all(values < np.pi)
