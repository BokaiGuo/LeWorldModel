import copy
import numpy as np
import pytest
from experiments.spectral_reachability import codex_p3_online_budget_20261030_v1 as kernel
from experiments.spectral_reachability import codex_p3_online_budget_saved_20261030_v1 as saved
from test_codex_p3_online_budget_20261030_v1 import Backend, fake_inputs


@pytest.fixture
def fixture(monkeypatch):
    class WithRows(Backend):
        def jacobian(self, task, incumbent, meter):
            terminal, matrix = super().jacobian(task, incumbent, meter)
            for i in range(192):
                meter.emit("jacobian_row", row=i, array=kernel.native.blob(matrix[i]))
            return terminal, matrix
    tasks, bank, _ = fake_inputs()
    events = []
    meter = kernel.Meter(events.append, lambda: None, lambda: None)
    kernel.execute(tasks, bank, [WithRows(), WithRows()], meter, check_path=lambda *x: None,
                   signal_fn=lambda task, *_: {"w": task["weight"]}, cost=lambda *x: 0.)
    monkeypatch.setattr(saved.native, "check_path", lambda *x: None)
    monkeypatch.setattr(saved.native, "signal", lambda task, *_: {"w": task["weight"]})
    monkeypatch.setattr(saved.native, "cost", lambda *x: 0.)
    return events, tasks, bank


def test_saved_all_points(fixture):
    events, tasks, bank = fixture
    result = saved.verify_events(events, tasks, bank)
    assert len(result["rows"]) == 6
    assert result["rows"][1]["allocation"] == [15, 5]
    assert result["rows"][5]["allocation"] == [8, 22]
    assert result["scientific_pass"] is False


@pytest.mark.parametrize("kind", ["allocation", "ledger", "jacobian", "winner", "missing", "scope"])
def test_tamper_rejected(fixture, kind):
    events, tasks, bank = fixture
    events = copy.deepcopy(events)
    if kind == "allocation":
        next(e for e in events if e["kind"] == "allocation_frozen")["quota"] = [1, 23]
    elif kind == "ledger":
        next(e for e in events if e["kind"] == "resource_return")["entry"]["resource"] = "svd"
    elif kind == "jacobian":
        next(e for e in events if e["kind"] == "jacobian_row")["row"] = 3
    elif kind == "winner":
        next(e for e in events if e["kind"] == "arm_complete")["row"]["winners"] = []
    elif kind == "missing":
        events.pop()
    else:
        next(e for e in events if e["kind"] == "resource_attempt")["scope"] = [20, "signal", "signal", 0]
    with pytest.raises((ValueError, KeyError)):
        saved.verify_events(events, tasks, bank)
