import pytest

from experiments.pc_matched_feedback_selected_audit_20261103_codex import audit_goal_row


def test_selected_actions_and_physical_boundary_are_bound():
    prefix = [[0., 0.]] * 5
    decisions = [{"root": 0, "step": step, "arm": "goal",
                  "initial_score": 1., "result": {
                      "selected": {"name": "zero", "prefix": prefix, "prefix_score": .9},
                      "same_bank_greedy_name": "zero", "selection": {"abstained": True},
                      "counts": {}, "seconds": {}}, "token_counts": {},
                  "encoding_calls": 3, "encoding_seconds": .1, "planning_seconds": .2}
                 for step in range(3)]
    execution = {"rows": [{"step": i, "actions": prefix} for i in range(3)]}
    physical = {"root": 0, "arm": "goal", "distances": [2., 1.8, 1.8, 1.7]}
    rows = audit_goal_row(physical, decisions, execution)
    assert [row["physical_nonimprovement"] for row in rows] == [False, True, False]
    execution["rows"][1]["actions"] = [[.1, 0.]] * 5
    with pytest.raises(ValueError):
        audit_goal_row(physical, decisions, execution)
