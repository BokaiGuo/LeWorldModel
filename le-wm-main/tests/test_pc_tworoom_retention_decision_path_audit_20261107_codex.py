import pytest

from experiments.pc_tworoom_retention_decision_path_audit_20261107_codex import (
    expected_retained, validate_program,
)


def test_retention_and_selected_score():
    d = {"parent_cost": [5., 6., 7.], "probe_cost": [[4., 8.], [3., 3.], [5., 1.]],
         "tail_cost": [8., 9., 10., 11.], "retained": [2, 1],
         "choice": {"source": "probe", "index": 5, "cost": 1.}, "strategy_seed": 10}
    assert expected_retained(d, "lookahead") == [2, 1]
    validate_program(d, "lookahead")
    d["choice"]["cost"] = 2.
    with pytest.raises(ValueError):
        validate_program(d, "lookahead")
