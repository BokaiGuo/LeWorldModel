import numpy as np
import pytest

from experiments.pc_tworoom_local_global_reuse_20261107_codex import choose


def test_choice_uses_scores_and_retains_incumbent_on_tie():
    ids = np.arange(16)
    scores = np.arange(16, dtype=float)
    assert choose(0.0, scores, ids) is None
    assert choose(0.1, scores, ids) == 0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        choose(1, np.zeros(15), np.arange(15))
    with pytest.raises(ValueError):
        choose(1, np.full(16, np.nan), np.arange(16))
    with pytest.raises(ValueError):
        choose(1, np.zeros(16), np.zeros(16, dtype=int))
