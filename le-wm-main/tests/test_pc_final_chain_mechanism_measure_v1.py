import numpy as np
import pytest

from experiments.pc_final_chain_mechanism_measure_v1 import measure


def test_final_chain_measurement_journals_before_requests_and_keeps_tail():
    chain = np.zeros((15, 2), np.float32)
    chain[5, 0], chain[-1, 1] = 1, -.5
    events = []

    def predict(actions):
        assert events[-1]["event"] == "request"
        positions = actions.astype(float).sum(axis=1)
        return positions, np.linalg.norm(positions, axis=1)

    def branches(actions):
        assert events[-1]["event"] == "branch_request"
        return actions.astype(float).sum(axis=1), np.zeros((len(actions), 15), bool)

    results = measure(chain, {"selected": chain[:5], "zero": chain[:5]}, [1, 1],
                      predict, branches, events.append, scales=(.01,))
    assert [e["event"] for e in events] == ["request", "branch_request", "result"] * 2
    assert events[0]["coordinates"] == 20
    np.testing.assert_array_equal(np.asarray(events[0]["actions"])[0], chain)
    assert len(results["selected"]["0.01"]["coordinate_class"]) == 20


def test_selected_control_must_match_final_chain_and_failures_leave_request():
    chain = np.zeros((15, 2), np.float32)
    with pytest.raises(ValueError):
        measure(chain, {"selected": np.ones((5, 2))}, [0, 0], None, None, None)
    events = []

    def fail(_):
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError):
        measure(chain, {"selected": chain[:5]}, [0, 0], fail, None, events.append)
    assert len(events) == 1 and events[0]["event"] == "request"
