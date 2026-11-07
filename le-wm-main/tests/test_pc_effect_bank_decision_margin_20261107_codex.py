import numpy as np
import pytest

from experiments.pc_effect_bank_decision_margin_20261107_codex import analyze_root


def test_choice_and_physical_margins():
    distance = np.tile(np.arange(9, dtype=float), (8, 1))
    model = -distance
    encoded = distance.copy()
    term = np.zeros((8, 9, 25), dtype=bool)
    term[:, 0, 24] = True
    result = analyze_root(model, encoded, distance, term)
    assert result["decisions"]["model"]["regret"] == [8.0] * 8
    assert result["decisions"]["encoded_oracle"]["ever_success"] == [1] * 8
    assert result["margins"]["model"]["5.0"] == [0, 24]
    assert result["margins"]["encoded_oracle"]["5.0"] == [24, 24]
    assert sum(result["success_available_only_after_prefix"]) == 8


def test_fail_closed():
    with pytest.raises(ValueError):
        analyze_root(np.zeros((8, 8)), np.zeros((8, 9)), np.zeros((8, 9)), np.zeros((8, 9, 25), dtype=bool))
    term = np.zeros((8, 9, 25), dtype=bool)
    term[0, 1, 0] = True
    with pytest.raises(ValueError):
        analyze_root(np.zeros((8, 9)), np.zeros((8, 9)), np.zeros((8, 9)), term)
