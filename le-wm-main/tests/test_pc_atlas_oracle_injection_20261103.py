import numpy as np

from experiments.spectral_reachability.d2p_atlas_nullspace_v3 import task_nullspace
from experiments.spectral_reachability.pc_atlas_oracle_injection_20261103 import diagnose


def test_supplied_strong_chain_is_scored_and_replayed_without_loss():
    for index in range(8):
        row = diagnose(index)
        assert row["direct_sha256"] == row["split_sha256"]
        assert row["candidate_offered_at_matching_successor"] > 0
        assert row["branch_terminal_cost"] <= row["candidate_terminal_cost"] + 1e-12
        assert not row["selected_strictly_worse_than_candidate"]


def test_nullspace_has_all_dimensions_with_weighted_coordinates():
    for jacobian in (np.array([[1., 0., 0.]]),
                     np.array([[1., 0.], [0., 0.], [0., 0.]]),
                     np.zeros((1, 2))):
        q = np.eye(jacobian.shape[0])
        scale = np.arange(1., jacobian.shape[1] + 1.)
        null = task_nullspace(jacobian, q, scale)
        assert null.shape == (jacobian.shape[1] - np.linalg.matrix_rank(jacobian), jacobian.shape[1])
        np.testing.assert_allclose(jacobian @ null.T, 0., atol=1e-14)
