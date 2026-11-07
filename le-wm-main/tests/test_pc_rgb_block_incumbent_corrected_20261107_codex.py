import numpy as np
import pytest

from experiments import pc_rgb_block_incumbent_search_20261107_codex as original
from experiments import pc_rgb_block_incumbent_corrected_20261107_codex as fixed


def objective(actions):
    return ((actions - 0.25) ** 2).sum(axis=(1, 2))


@pytest.mark.parametrize("arm", fixed.ARMS)
@pytest.mark.parametrize("remaining", (5, 15, 25))
def test_no_old_tail_matches_original(arm, remaining):
    old = original.search(objective, remaining=remaining, old_tail=None, seed=42, arm=arm)
    new = fixed.search(objective, remaining=remaining, old_tail=None, seed=42, arm=arm)
    assert np.array_equal(old["all_scores"], new["all_scores"])
    assert np.array_equal(old["actions"], new["actions"])
    assert old["query_batches"] == new["query_batches"]
    assert new["model_queries"] == 680


def test_representable_old_tail_is_selected_with_correct_parameters():
    batches = []
    tail = np.full((10, 2), 0.25, dtype=np.float32)

    def score(actions):
        batches.append(actions.copy())
        return objective(actions)

    result = fixed.search(score, remaining=10, old_tail=tail, seed=1, arm="baseline")
    assert np.array_equal(batches[0][0], tail)
    assert result["old_tail_representable"] and result["old_tail_elite"]
    assert result["old_tail_rank"] == 1
    assert np.array_equal(result["actions"], tail)
    assert len(batches) == 5


def test_nonrepresentable_old_tail_matches_original_and_is_retained():
    tail = np.full((10, 2), 0.25, dtype=np.float32)
    tail[1, 0] = 0.2
    old = original.search(objective, remaining=10, old_tail=tail, seed=7, arm="baseline")
    new = fixed.search(objective, remaining=10, old_tail=tail, seed=7, arm="baseline")
    assert not new["old_tail_representable"] and not new["old_tail_elite"]
    assert np.array_equal(old["all_scores"], new["all_scores"])
    assert np.array_equal(old["actions"], new["actions"])


def test_old_tail_not_elite_and_stable_tie():
    tail = np.ones((10, 2), dtype=np.float32)
    worse = fixed.search(objective, remaining=10, old_tail=tail, seed=3, arm="baseline")
    assert worse["old_tail_rank"] > 10 and not worse["old_tail_elite"]
    tie = fixed.search(lambda a: np.zeros(len(a)), remaining=10, old_tail=tail, seed=3, arm="baseline")
    assert tie["old_tail_rank"] == 1 and tie["old_tail_elite"] and tie["winner"] == 0
    assert np.array_equal(tie["actions"], tail)


def test_reject_invalid_old_tail():
    with pytest.raises(ValueError):
        fixed.search(objective, remaining=10, old_tail=np.full((10, 2), 2), seed=1, arm="baseline")
