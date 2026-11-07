import numpy as np
import pytest

from experiments.pc_tworoom_local_b_batch_audit_20261107_codex import select


def test_select_excludes_other_arm_and_resolves_ties_stably():
    scores = np.ones(98)
    scores[66] = 0
    scores[82] = -1
    assert select(scores, 66)[0] == 66
    assert select(scores, 82)[0] == 82
    scores[:] = 0
    assert select(scores, 66) == (0, 0)
    assert select(scores, 82) == (0, 0)


def test_invalid_scores_and_arm_rejected():
    with pytest.raises(ValueError):
        select(np.zeros(98), 67)
    with pytest.raises(ValueError):
        select(np.full(98, np.nan), 66)
