import numpy as np
import pytest

from experiments.spectral_reachability import p1_development_diverse_cem_20261102 as subject
from experiments.spectral_reachability import p1_dev304_diverse_saved_20261102 as saved
from experiments.spectral_reachability import p1_dev304_terminal_saved_20261102 as prior
from experiments.spectral_reachability import p1_unified_cp5_feedback_cem_20261101_codex as common


@pytest.mark.parametrize("structured", [False, True])
def test_fixed_budget_and_selection_is_global_minimum(structured):
    calls = []

    def score(variants):
        calls.append(variants.shape)
        return (variants ** 2).sum(axis=(2, 3, 4))

    result = subject.plan([1., 0.], score, structured=structured, common=common, seed=15)
    assert calls == [(64, 5, 2, 5, 2)] * 5
    assert len(result["iterations"]) == 5
    all_scores = []
    for event in result["iterations"]:
        robust = common.robust_scores(event["all_scores"])
        all_scores.extend(robust[np.asarray(event["feasible"], dtype=bool)])
        assert len(event["bank"]) == 64
        assert sum(map(len, event["elite_indices_by_population"])) == 8
    assert result["selected"]["robust_score"] == min(all_scores)


def test_reproducible_and_invalid_direction_rejected():
    score = lambda variants: (variants ** 2).sum(axis=(2, 3, 4))
    a = subject.plan([1., 0.], score, structured=True, common=common, seed=15)
    b = subject.plan([1., 0.], score, structured=True, common=common, seed=15)
    assert a == b
    with pytest.raises(ValueError):
        subject.plan([0., 0.], score, structured=True, common=common, seed=15)


def test_recorded_development_run():
    result = saved.analyze()
    assert not result["same_selected_actions"]
    assert result["frames_verified"] == 20
    assert all(not row["success"] for row in result["rows"])
    assert not result["p1_admission"] and not result["independent_confirmation"]
    old_result = prior.analyze()
    assert result["rows"][0] == old_result["rows"][0]
