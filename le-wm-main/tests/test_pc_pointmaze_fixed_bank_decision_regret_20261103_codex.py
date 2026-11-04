import pytest

from experiments.pc_pointmaze_fixed_bank_decision_regret_20261103_codex import analyze


def test_fixed_bank_uses_recorded_index_as_tie_break_and_physical_oracle():
    branches = [{"index": i, "open_cost": 1 if i in (1, 2) else 2,
                 "refresh_cost": 0 if i == 3 else 3,
                 "actual_encoded_cost": 0 if i == 4 else 3,
                 "physical_distance": abs(i - 3)} for i in reversed(range(9))]
    result = analyze({"root": 7, "branches": branches})
    assert result["oracle_index"] == 3
    assert result["selectors"]["open_cost"]["index"] == 1
    assert result["selectors"]["open_cost"]["physical_regret"] == 2
    assert result["selectors"]["refresh_cost"]["physical_regret"] == 0


def test_incomplete_bank_fails_closed():
    with pytest.raises(ValueError, match="nine indexed"):
        analyze({"root": 7, "branches": []})
