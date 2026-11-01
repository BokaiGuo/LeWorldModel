import numpy as np
import pytest

from experiments.spectral_reachability import p1_adam_checkpoint_curve_20261031 as curve


def pick(scores):
    return min(range(4), key=lambda i: (scores[i], i))


def test_pack_preserves_actions_and_breaks_score_ties_by_index():
    actions = [np.full((2, 5, 2), i / 10, dtype="<f4") for i in range(4)]
    result = curve.pack_checkpoint(actions, [2.0, 1.0, 1.0, 3.0], pick)
    assert result["winner"] == 1
    assert np.array_equal(np.asarray(result["candidate_actions"][2], dtype="<f4"), actions[2])
    assert len(set(result["candidate_sha256"])) == 4


@pytest.mark.parametrize("scores", [[1.0, float("nan"), 2.0, 3.0], [1.0]])
def test_rejects_bad_scores(scores):
    with pytest.raises(ValueError):
        curve.pack_checkpoint([np.zeros((2, 5, 2))] * 4, scores, pick)


def test_rejects_out_of_domain_actions():
    with pytest.raises(ValueError, match="invalid raw actions"):
        curve.pack_checkpoint([np.full((2, 5, 2), 1.1)] * 4, [0.0] * 4, pick)
