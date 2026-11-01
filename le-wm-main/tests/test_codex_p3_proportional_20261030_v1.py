from contextlib import contextmanager
import numpy as np
import pytest

from experiments.spectral_reachability import codex_p3_proportional_kernel_20261030_v1 as m


class Backend:
    @contextmanager
    def guard(self, meter):
        yield

    def rollout(self, task, commands, meter, mode):
        for _ in range(2):
            for axis in m.NEURAL:
                meter.call(axis, lambda: None)
        return [dict(latent_history=[[0.] * 192])]

    def jacobian(self, task, incumbent, meter):
        for _ in range(2):
            for axis in m.NEURAL:
                meter.call(axis, lambda: None)
        for _ in range(192):
            for axis in ("vjp_directions", "grad_api", "raw_backward_hook"):
                meter.call(axis, lambda: None)
        return np.zeros(192), np.zeros((192, 20))


def fake_inputs():
    bank = [[dict(index=i, commands=[[0., 0.]] * 10) for i in range(30)] for _ in range(2)]
    return [{"weight": 2.}, {"weight": 1.}], bank, [Backend(), Backend()]


def test_fixed_points_allocations_and_bank():
    assert m.POINTS == (20, 30)
    assert m.allocation(20, "uniform") == [12, 12]
    assert m.allocation(30, "uniform") == [17, 17]
    assert m.allocation(20, "signal", [2., 1.]) == [13, 7]
    assert m.allocation(20, "reverse", [2., 1.]) == [7, 13]
    assert m.allocation(30, "signal", [2., 1.]) == [20, 10]
    assert m.allocation(30, "reverse", [2., 1.]) == [10, 20]
    assert m.allocation(20, "signal", [0., 1e308]) == [1, 19]
    assert m.allocation(20, "signal", [1e308, 1e308]) == [10, 10]
    assert m.allocation(30, "signal", [0., 0.]) == [15, 15]
    with pytest.raises(ValueError):
        m.allocation(20, "uniform", [1., 2.])
    assert m.banks(m.native.BALL) == m.banks(m.native.BALL)


def test_complete_and_each_real_axis_separate():
    tasks, bank, backends = fake_inputs()
    events = []
    meter = m.Meter(events.append, lambda: None, lambda: None)
    rows = m.execute(tasks, bank, backends, meter, check_path=lambda *x: None,
                     signal_fn=lambda task, *_: {"w": task["weight"]}, cost=lambda *x: 0.)
    assert len(rows) == 6 and events[-1]["kind"] == "experiment_complete"
    assert [row["allocation"] for row in rows] == [
        [12, 12], [13, 7], [7, 13], [17, 17], [20, 10], [10, 20]]
    assert len([e for e in events if e["kind"] == "signal_complete"]) == 8
    for row in rows:
        p, arm = row["point"], row["arm"]
        assert all(row["returned"][axis] == 2 * p + 8 for axis in m.NEURAL)
        assert row["returned"]["vjp_directions"] == (0 if arm == "uniform" else 384)
        assert row["returned"]["metric"] == (p + 4 if arm == "uniform" else p)
        assert sum(row["allocation"]) == row["returned"]["metric"]


def test_caps_and_failed_dispatch_are_nonrefundable():
    events = []
    meter = m.Meter(events.append, lambda: None, lambda: None)
    meter.scope = [20, "uniform", "search", 0]
    for _ in range(m.caps(20)["metric"]):
        meter.call("metric", lambda: None)
    with pytest.raises(ValueError, match="vector cap"):
        meter.call("metric", lambda: None)
    assert meter.failed is True
    meter = m.Meter(events.append, lambda: None, lambda: None)
    meter.scope = [20, "uniform", "signal", 0]
    with pytest.raises(ValueError, match="resource phase"):
        meter.call("svd", lambda: None)
    failed = m.Meter(events.append, lambda: None, lambda: None)
    failed.scope = [20, "signal", "signal", 0]
    with pytest.raises(RuntimeError):
        failed.call("jacobian", lambda: (_ for _ in ()).throw(RuntimeError("failure")))
    assert failed.failed and failed.events[0]["completed"] is False
    with pytest.raises(ValueError):
        failed.call("jacobian", lambda: None)


def test_abort_retains_failure_and_no_later_dispatch():
    tasks, bank, backends = fake_inputs()
    events = []
    meter = m.Meter(events.append, lambda: None, lambda: None)
    with pytest.raises(RuntimeError):
        m.execute(tasks, bank, backends, meter, check_path=lambda *x: None,
                  signal_fn=lambda *x: {"w": 1.},
                  cost=lambda *x: (_ for _ in ()).throw(RuntimeError("metric failure")))
    assert events[-1]["kind"] == "experiment_failure"
    assert events[-1]["retry_authorized"] is False
    assert events[-1]["completed_arms"] == 0


def test_host_rejects_cpu():
    with pytest.raises(ValueError, match="host CUDA only"):
        m.execute_host(None, type("Device", (), {"type": "cpu"})(), [], None, None, None)


def test_incomplete_native_operation_vector_fails_closed():
    tasks, bank, backends = fake_inputs()
    class Broken(Backend):
        def rollout(self, task, commands, meter, mode):
            result = super().rollout(task, commands, meter, mode)
            meter.call("predict", lambda: None)
            return result
    backends[0] = Broken()
    events = []
    meter = m.Meter(events.append, lambda: None, lambda: None)
    with pytest.raises(ValueError, match="candidate vector"):
        m.execute(tasks, bank, backends, meter, check_path=lambda *x: None,
                  signal_fn=lambda task, *_: {"w": task["weight"]}, cost=lambda *x: 0.)
    assert events[-1]["kind"] == "experiment_failure"
    assert meter.failed is True
