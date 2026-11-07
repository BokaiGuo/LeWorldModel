import numpy as np
import pytest

from experiments.pc_rgb_block_incumbent_search_20261107_codex import search


@pytest.mark.parametrize("remaining", [5, 10, 15, 20, 25])
@pytest.mark.parametrize("arm", ["baseline", "lookahead", "vector"])
def test_budget_bounds_and_determinism(remaining, arm):
    def score(x):
        return np.square(x - 0.37).sum(axis=(1, 2))

    old = np.full((remaining, 2), 0.37, dtype=np.float32)
    first = search(score, remaining=remaining, old_tail=old, seed=19, arm=arm)
    second = search(score, remaining=remaining, old_tail=old, seed=19, arm=arm)
    assert first["model_queries"] == 680
    assert sum(first["query_batches"]) == 680
    assert first["model_packet_steps"] == 680 * (remaining // 5)
    assert first["winner"] == 0
    assert first["cost"] == 0
    np.testing.assert_array_equal(first["actions"], old)
    assert first["actions_sha256"] == second["actions_sha256"]


def test_reject_wrong_tail_and_nonfinite_score():
    with pytest.raises(ValueError):
        search(lambda x: np.zeros(len(x)), remaining=10, old_tail=np.zeros((5, 2)), seed=1, arm="baseline")
    with pytest.raises(ValueError):
        search(lambda x: np.full(len(x), np.nan), remaining=10, old_tail=None, seed=1, arm="vector")
