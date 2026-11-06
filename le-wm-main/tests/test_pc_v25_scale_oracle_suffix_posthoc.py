import numpy as np
import pytest

from experiments.pc_v25_scale_oracle_suffix_posthoc_20261104_codex import summarize


def test_fixed_suffix_bank_and_retention_are_separate():
    prefixes = np.zeros((3, 5, 2), dtype=np.float32)
    scores = np.array([[.3, .2], [.04, .5], [.1, .2]])
    summary = summarize(prefixes, scores, {"response": [0, 2], "greedy": [0, 1]})
    assert summary["bank_success"]
    assert not summary["rules"]["response"]["success"]
    assert summary["rules"]["greedy"]["success"]
    assert summary["bank_winner"] == 1


def test_rejects_malformed_scores_and_selection():
    prefixes = np.zeros((2, 5, 2))
    with pytest.raises(ValueError, match="candidate scores"):
        summarize(prefixes, np.zeros((2, 1)), {"response": [0]})
    with pytest.raises(ValueError, match="selection"):
        summarize(prefixes, np.zeros((2, 2)), {"response": [0, 0]})
