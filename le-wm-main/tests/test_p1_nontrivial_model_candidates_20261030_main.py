import numpy as np
import pytest

from experiments.spectral_reachability.p1_nontrivial_model_candidates_20261030_main import analyze


def test_zero_incumbent_and_model_criteria_are_separate():
    prefixes = np.zeros((20, 5, 2), np.float32)
    prefixes[0, 0, 0] = 0.5
    prefixes[1] = prefixes[0]
    prefixes[2, 0, 0] = -0.0
    prefixes[3, 0, 0] = 0.1
    stage, scores = [1.0] * 20, [2.0] * 20
    scores[3] = 1.0
    result = analyze(prefixes, stage, 1.0, scores)
    assert result[1]["same_as_incumbent"]
    assert result[2]["zero"]
    assert not result[3]["zero"] and not result[3]["same_as_incumbent"]
    assert result[3]["model_non_improving_stage"] and result[3]["score_beats_incumbent"]


def test_rejects_missing_or_nonfinite_scores():
    with pytest.raises(ValueError):
        analyze(np.zeros((20, 5, 2)), [0.0] * 20, 0.0, [0.0] * 19)
    with pytest.raises(ValueError):
        analyze(np.zeros((20, 5, 2)), [0.0] * 20, 0.0, [float("nan")] * 20)
