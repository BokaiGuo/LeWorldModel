import pytest

from experiments.pc_tworoom_success_contract_audit_20261107_codex import audit_trace


def arm(distance, terminated):
    return {"trace": [{"action": [0.0, 1.0], "distance": distance,
                       "terminated": terminated, "truncated": False}],
            "ever_success": terminated, "truncated": False, "terminal_distance": distance}


def test_strict_boundary():
    assert audit_trace(arm(15.999, True))
    assert not audit_trace(arm(16.0, False))
    with pytest.raises(ValueError):
        audit_trace(arm(16.0, True))


def test_rejects_illegal_action_and_summary():
    x = arm(20, False)
    x["trace"][0]["action"] = [1.1, 0]
    with pytest.raises(ValueError):
        audit_trace(x)
    x = arm(20, False)
    x["ever_success"] = True
    with pytest.raises(ValueError):
        audit_trace(x)
