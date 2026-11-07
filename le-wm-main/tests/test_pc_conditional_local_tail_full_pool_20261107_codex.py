import numpy as np
import pytest

from experiments.pc_conditional_local_tail_full_pool_20261107_codex import outcome_summary, validate_ids


def test_retained_only_and_rank():
    ids = np.arange(24)
    outcomes = []
    for i in ids:
        trace = [{"terminated": bool(i == 16 and j == 15)} for j in range(25)]
        outcomes.append({"index": int(i), "trace": trace})
    scores = np.arange(24, dtype=float)
    assert not outcome_summary(ids, scores, outcomes, [0, 1], False)["pool_success"]
    result = outcome_summary(ids, scores, outcomes, [0, 2], False)
    assert result["pool_success"] and result["pool_tail_success"]
    assert result["best_success_rank"] == 9 and not result["best_local_success"]


def test_missing_candidates_are_rejected():
    with pytest.raises(ValueError, match="missing"):
        validate_ids(np.arange(8), [0, 1])
