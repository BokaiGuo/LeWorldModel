import numpy as np
import pytest
from experiments.spectral_reachability import main_raw_fd_recompute_20261028_v1 as m


def test_full_matrix_identity_and_wrong_sign():
    j = np.arange(1920, dtype=np.float64).reshape(192,10)
    a = m.matrix_stats(j,j); b = m.matrix_stats(j,-j)
    assert a['relative_l2_to_AD'] == 0 and a['direction_cosine'] > .999999
    assert b['relative_l2_to_AD'] == 2 and b['direction_cosine'] < -.999999


def test_subset_never_promoted_to_full_matrix():
    with pytest.raises(ValueError): m.matrix_stats(np.ones((192,9)),np.ones((192,9)))


def test_incomplete_endpoint_probe_not_filled_or_dropped():
    with pytest.raises(ValueError): m.point_stats(dict(index=0),[],dict(index=0,evaluations=[]),dict(index=0))
