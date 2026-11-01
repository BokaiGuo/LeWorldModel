import copy
import pytest

from experiments.spectral_reachability import codex_p3_proportional_kernel_20261030_v1 as kernel
from experiments.spectral_reachability import codex_p3_proportional_saved_20261030_v1 as saved
from test_codex_p3_proportional_20261030_v1 import Backend, fake_inputs


@pytest.fixture
def fixture(monkeypatch):
    class WithRows(Backend):
        def jacobian(self, task, incumbent, meter):
            terminal, matrix = super().jacobian(task, incumbent, meter)
            for i in range(192):
                meter.emit("jacobian_row", row=i, array=kernel.native.blob(matrix[i]))
            return terminal, matrix
    monkeypatch.setattr(kernel, "POINTS", (20,))
    monkeypatch.setattr(kernel, "ARMS", ("signal",))
    tasks, bank, _ = fake_inputs()
    events = []
    meter = kernel.Meter(events.append, lambda: None, lambda: None)
    kernel.execute(tasks, bank, [WithRows(), WithRows()], meter, check_path=lambda *x: None,
                   signal_fn=lambda task, *_: {"w": task["weight"]}, cost=lambda *x: 0.)
    monkeypatch.setattr(saved.native, "check_path", lambda *x: None)
    monkeypatch.setattr(saved.native, "signal", lambda task, *_: {"w": task["weight"]})
    monkeypatch.setattr(saved.native, "cost", lambda *x: 0.)
    return events, tasks, bank


def test_saved_signal_allocation_and_cost(fixture):
    events, tasks, bank = fixture
    result = saved.verify_events(events, tasks, bank, 20, "signal")
    row = result["rows"][0]
    assert row["allocation"] == [13, 7]
    assert row["attempted"]["vjp_directions"] == 384
    assert row["attempted"]["metric"] == 20
    assert not result["scientific_pass"]


@pytest.mark.parametrize("kind", ["allocation", "ledger", "jacobian", "winner", "missing", "scope"])
def test_tamper_fails(fixture, kind):
    events, tasks, bank = fixture
    events = copy.deepcopy(events)
    if kind == "allocation":
        next(e for e in events if e["kind"] == "allocation_frozen")["quota"] = [15, 5]
    elif kind == "ledger":
        next(e for e in events if e["kind"] == "resource_return")["entry"]["resource"] = "svd"
    elif kind == "jacobian":
        next(e for e in events if e["kind"] == "jacobian_row")["row"] = 3
    elif kind == "winner":
        next(e for e in events if e["kind"] == "arm_complete")["row"]["winners"] = []
    elif kind == "missing":
        events.pop()
    else:
        next(e for e in events if e["kind"] == "resource_attempt")["scope"] = [20, "signal", "search", 1]
    with pytest.raises((ValueError, KeyError)):
        saved.verify_events(events, tasks, bank, 20, "signal")
