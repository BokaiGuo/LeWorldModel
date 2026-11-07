import numpy as np
import pytest

from experiments.pc_tworoom_goal_norm_diagnostic_20261107_codex import scores


def test_cosine_ignores_positive_candidate_scale():
    goal = np.ones(192)
    x = np.ones((64, 192))
    x *= np.arange(1, 65)[:, None]
    ref = np.square(x - goal).sum(axis=1)
    cosine, norms, _ = scores(x, goal, ref)
    np.testing.assert_allclose(cosine, 0, atol=1e-14)
    assert norms[-1] > norms[0]


def test_invalid_reference_and_zero_are_rejected():
    goal = np.ones(192)
    x = np.ones((64, 192))
    with pytest.raises(ValueError, match="original cost"):
        scores(x, goal, np.ones(64))
    x[0] = 0
    with pytest.raises(ValueError, match="zero norm"):
        scores(x, goal, np.square(x - goal).sum(axis=1))
