import numpy as np
import pytest

from experiments import pc_shortlist_directed_coverage_dev_20261105_codex as run


def bank():
    return [{"chain": np.full((5, 2), v, np.float32).tolist(), "score": float(i)}
            for i, v in enumerate([0, .1, .1, .2, .3, .4, .5])]


def test_unique_shortlists_and_fixed_seed():
    items = bank()
    assert run.shortlist(items, False, 3) == [0, 1, 3, 4]
    explored = run.shortlist(items, True, 3)
    assert explored[:3] == [0, 1, 3]
    assert explored[3] in [4, 5, 6]
    assert explored == run.shortlist(items, True, 3)
    with pytest.raises(ValueError):
        run.shortlist(items[:4], True, 3)


def test_directed_opportunity_uses_absolute_improvement():
    scores = np.array([[1.] * 9, [.9] * 9, [3.] * 9, [1.1] * 9])
    scores[2, 0] = 0
    assert run.choose([0, 1, 2, 3], scores, "min") == (0, 2)
    assert run.choose([0, 1, 2, 3], scores, "directed") == (0, 2)
    scores[2, 0] = .99
    assert run.choose([0, 1, 2, 3], scores, "directed") == (0, 1)


@pytest.mark.parametrize("rule", ["min", "directed"])
def test_budget_and_all_scored_chains_can_win(rule):
    seen = []
    items = bank()

    def predictor(actions):
        a = np.asarray(actions, np.float32)
        assert a.shape[1:] == (15, 2)
        assert np.max(np.abs(a)) <= 1
        features = a.sum(axis=1).astype(np.float64)
        scores = np.linalg.norm(features, axis=1)
        seen.extend((x.copy(), float(y)) for x, y in zip(a, scores, strict=True))
        return features, scores

    output = run.evaluate_arm(predictor, items, 0, [0, 1, 3, 4], rule, 13)
    assert len(seen) == 276
    assert output["final_score"] == min(score for _, score in seen)
    assert any(np.array_equal(a, np.asarray(output["final_actions"], np.float32))
               and score == output["final_score"] for a, score in seen)
    assert len(output["retained"]) == 2 and output["retained"][0] == 0
