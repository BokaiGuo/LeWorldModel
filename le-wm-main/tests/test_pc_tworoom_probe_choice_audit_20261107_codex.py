import numpy as np
import pytest

from experiments.pc_tworoom_probe_choice_audit_20261107_codex import choice_metrics


def test_regret_uses_selected_indices_not_score_scale():
    result = choice_metrics([5, 1, 3, 4, 2], [0, 8, 1, 3, 4], [5, 2, 1, 3, 4])
    assert result["model_index"] == 0
    assert result["encoded_index"] == 2
    assert result["model_regret"] == 4
    assert result["encoded_regret"] == 2
    assert result["zero_regret"] == 4
    assert not result["model_matches_encoded"]


def test_ties_are_deterministic_and_invalid_inputs_fail():
    result = choice_metrics([1, 1, 2, 3, 4], [0, 0, 1, 2, 3], [1, 0, 1, 2, 3])
    assert result["model_index"] == 0 and result["model_regret"] == 0
    with pytest.raises(ValueError):
        choice_metrics([0] * 4, [0] * 5, [0] * 5)
    with pytest.raises(ValueError):
        choice_metrics([np.nan] * 5, [0] * 5, [0] * 5)
