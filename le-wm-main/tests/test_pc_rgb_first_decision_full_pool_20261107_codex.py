import pytest

from experiments.pc_rgb_first_decision_full_pool_20261107_codex import build_scored


def trace(success):
    return {"distance": [20.] * 25, "success": [False] * 24 + [success]}


def test_score_selection_and_arm_specific_tails():
    decision = {"retained": [2, 0], "parent_cost": [5., 6., 7.],
                "probe_cost": [[8., 9., 10.]] * 3, "tail_cost": [4., 3., 2., 1.],
                "choice": {"source": "tail", "index": 3, "cost": 1.}}
    physical = [[trace(False) for _ in range(5)] for _ in range(3)]
    physical[0][4] = trace(True)
    result = build_scored(decision, None, physical, [trace(False)] * 3)
    assert result["choice_flat_index"] == 15
    assert result["ever_oracle"] and result["ever_chosen"]
    decision["choice"]["index"] = 0
    with pytest.raises(ValueError):
        build_scored(decision, None, physical, [trace(False)] * 3)


def test_reject_duplicate_retention():
    with pytest.raises(ValueError):
        build_scored({"retained": [1, 1]}, None, None, None)
