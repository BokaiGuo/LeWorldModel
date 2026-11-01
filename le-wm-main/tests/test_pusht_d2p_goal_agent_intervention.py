import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability.audit_pusht_d2p_goal_agent_intervention import costs, digest, OFFSETS


def test_all_coordinates_used_and_fixed_actions_can_change_rank():
    actions = np.zeros((2, 192))
    actions[1, -1] = 2.
    goal = np.zeros(192)
    assert np.argmin(costs(actions, goal)) == 0
    goal[-1] = 2.
    assert np.argmin(costs(actions, goal)) == 1
    assert costs(actions, goal)[0] == pytest.approx(4/192)


def test_file_digest_is_content_hash_not_array_of_path(tmp_path):
    path = tmp_path/"weights.fixture"
    path.write_bytes(b"content-only")
    assert digest(path) == hashlib.sha256(b"content-only").hexdigest()


def test_fixed_symmetric_goal_offsets():
    assert OFFSETS[0] == (0., 0.)
    np.testing.assert_array_equal(np.sum(OFFSETS, axis=0), [0., 0.])
    assert len(OFFSETS) == 5


@pytest.mark.parametrize("vectors,goal", [(np.zeros((2, 191)), np.zeros(192)),
    (np.zeros((2, 192)), np.zeros(191)), (np.full((2, 192), np.nan), np.zeros(192)),
    (np.zeros((2, 192)), np.full(192, np.inf))])
def test_invalid_goal_vectors_fail_closed(vectors, goal):
    with pytest.raises(ValueError):
        costs(vectors, goal)
