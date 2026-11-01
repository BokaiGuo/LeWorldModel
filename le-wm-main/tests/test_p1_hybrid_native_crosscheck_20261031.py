import pytest

from experiments.spectral_reachability import p1_hybrid_native_crosscheck_20261031 as audit


def fixture_events():
    action = "00000000"
    operation = {"kind": "operation_returned", "index": 0,
                 "receipt": {"resource": "public_step", "branch": 0, "phase": "probe",
                             "step": 1, "raw_bits": action}}
    transition = {"kind": "returned_transition", "index": 1, "branch": 0, "phase": "probe",
                  "step": 1, "requested_bits": action, "actual_bits": action,
                  "state": [0.0] * 7, "goal_state": [1.0] * 7, "reward": -3.0,
                  "terminated": False}
    response = {"delta_native7": [0.0] * 7, "delta_angle_wrapped": 0.0, "terminated": False}
    return [operation, transition], [{"responses": [response], "start_state": [0.0] * 7}]


def test_native_crosscheck_rejects_bad_reward_and_action(monkeypatch):
    monkeypatch.setattr(audit, "metric", lambda goal, state:
                        {"native_distance": 3.0, "native_success": False})
    monkeypatch.setattr(audit, "physical_metrics", lambda state, goal: {"native_success": False})
    events, rows = fixture_events()
    anchor = {"goal": [1.0] * 7}
    assert audit.verify_transitions(events, anchor, rows) == {"plan": 0, "probe": 1}
    events[1]["reward"] = -2.0
    with pytest.raises(ValueError, match="native reward"):
        audit.verify_transitions(events, anchor, rows)
    events[1]["reward"] = -3.0
    events[1]["actual_bits"] = "bad"
    with pytest.raises(ValueError, match="action binding"):
        audit.verify_transitions(events, anchor, rows)
    events[1]["actual_bits"] = "00000000"
    rows[0]["responses"] = []
    with pytest.raises(ValueError, match="denominator"):
        audit.verify_transitions(events, anchor, rows)
