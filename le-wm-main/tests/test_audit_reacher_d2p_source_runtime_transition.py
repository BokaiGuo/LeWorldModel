import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_source_runtime_transition import _valid_source_rows


def test_valid_source_rows_never_crosses_an_episode_boundary():
    rows = _valid_source_rows(np.asarray([7, 10]), np.asarray([0, 7]), action_block=5)
    assert rows.tolist() == [0, 1, 7, 8, 9, 10, 11]
