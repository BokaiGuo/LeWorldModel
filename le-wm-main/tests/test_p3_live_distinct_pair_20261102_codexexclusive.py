"""Callback fixtures exercise control flow only, never neural/physical evidence."""
from contextlib import nullcontext

import numpy as np
import pytest

from experiments.spectral_reachability import p3_live_distinct_pair_20261102_codexexclusive as m


class Meter:
    def __init__(self):
        self.events = []
        self.scope = None

    def emit(self, kind, **fields):
        pass

    def call(self, resource, fn):
        event = {"resource": resource, "scope": self.scope[:], "completed": False}
        self.events.append(event)
        value = fn()
        event["completed"] = True
        return value


class Backend:
    def __init__(self, root):
        self.root = root

    def guard(self, meter):
        self.meter = meter
        return nullcontext()

    def rollout(self, task, commands, meter, phase):
        return meter.call("predict", lambda: [
            {"latent_history": [[float(self.root)]]},
            {"latent_history": [[float(self.root)]]},
        ])

    def jacobian(self, task, command, meter):
        return np.array([float(self.root)]), np.array([[1.0]])


def inputs():
    banks = [[{"index": i, "commands": [[float(i), float(r)]]}
              for i in range(30)] for r in range(2)]
    return [0, 1], banks, [Backend(0), Backend(1)]


def run(point=20, signal=lambda task, terminal, jacobian: {"w": 2.0 - task}, cost=None):
    tasks, banks, backends = inputs()
    meter = Meter()
    if cost is None:
        cost = lambda task, path: float(task)
    result = m._execute_pair(point, tasks, banks, backends, meter,
                             check_path=lambda *args: None, signal_fn=signal, cost=cost)
    return result, meter


@pytest.mark.parametrize("point,expected", [(20, [[15, 5], [5, 15]]),
                                            (30, [[22, 8], [8, 22]])])
def test_actual_callback_traversal_uses_distinct_shared_budget(point, expected):
    result, meter = run(point)
    assert [r["allocation"] for r in result["rows"]] == expected
    assert result["equal_recorded_axes"] and not result["P3_admission"]
    assert sum(e["resource"] == "metric" for e in meter.events) == 2 * point
    for row in result["rows"]:
        for root, winner in enumerate(row["winners"]):
            assert winner["commands"] == [[float(winner["index"]), float(root)]]
            assert len(winner["path"]) == 2


def test_signal_order_changes_direction_and_tie_fails():
    result, _ = run(signal=lambda task, *_: {"w": float(task + 1)})
    assert result["rows"][0]["allocation"] == [5, 15]
    with pytest.raises(ValueError, match="tied signal"):
        run(signal=lambda *_: {"w": 1.0})


def test_failure_is_not_retried_or_refunded():
    meter = Meter()
    tasks, banks, backends = inputs()
    def fail(*_):
        raise RuntimeError("failed")
    with pytest.raises(RuntimeError, match="failed"):
        m._execute_pair(20, tasks, banks, backends, meter,
                        check_path=lambda *args: None,
                        signal_fn=lambda task, *_: {"w": 2.0 - task}, cost=fail)
    assert sum(e["resource"] == "metric" for e in meter.events) == 1
    assert not meter.events[-1]["completed"]


def test_public_entry_rejects_cpu_before_neural_calls():
    class Device:
        type = "cpu"
    with pytest.raises(ValueError, match="host CUDA only"):
        m.execute_host(None, Device(), [], None, None, None, point=20)
