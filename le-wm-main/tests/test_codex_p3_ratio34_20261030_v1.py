from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import codex_p3_ratio34_kernel_20261030_v1 as k
from experiments.spectral_reachability import codex_p3_ratio34_saved_20261030_v1 as s
from experiments.spectral_reachability import codex_p3_ratio34_delivery_20261030_v1 as delivery
from test_codex_p3_resolved_20261030_v1 import Backend, fake_inputs


@pytest.mark.parametrize("point,high,low", [(20, 15, 5), (30, 23, 7)])
def test_rank_schedule_distinct_and_ties(point, high, low):
    assert k.allocation(point, "signal", [2., 1.]) == [high, low]
    assert k.allocation(point, "reverse", [2., 1.]) == [low, high]
    assert k.allocation(point, "signal", [1., 2.]) == [low, high]
    assert k.allocation(point, "signal", [1., 1.]) == [point // 2] * 2
    assert k.allocation(point, "reverse", [0., 0.]) == [point // 2] * 2


def test_full_real_accounting_and_consumer_tamper(monkeypatch):
    class WithRows(Backend):
        def jacobian(self, task, incumbent, meter):
            terminal, matrix = super().jacobian(task, incumbent, meter)
            for i in range(192):
                meter.emit("jacobian_row", row=i, array=k.native.blob(matrix[i]))
            return terminal, matrix

    monkeypatch.setattr(k, "POINTS", (20,))
    monkeypatch.setattr(k, "ARMS", ("signal",))
    tasks, bank, _ = fake_inputs()
    events = []
    meter = k.Meter(events.append, lambda: None, lambda: None)
    k.execute(tasks, bank, [WithRows(), WithRows()], meter, check_path=lambda *a: None,
              signal_fn=lambda task, *_: {"w": task["weight"]}, cost=lambda *a: 0.)
    monkeypatch.setattr(s.native, "check_path", lambda *a: None)
    monkeypatch.setattr(s.native, "signal", lambda task, *_: {"w": task["weight"]})
    monkeypatch.setattr(s.native, "cost", lambda *a: 0.)
    row = s.verify_events(events, tasks, bank, 20, "signal")["rows"][0]
    assert row["allocation"] == [15, 5]
    assert row["attempted"]["metric"] == 20
    assert row["attempted"]["predict"] == 48
    assert row["attempted"]["vjp_directions"] == 384
    assert row["attempted"]["scheduler"] == 1
    broken = deepcopy(events)
    next(e for e in broken if e["kind"] == "allocation_frozen")["quota"] = [10, 10]
    with pytest.raises(ValueError):
        s.verify_events(broken, tasks, bank, 20, "signal")
    broken = deepcopy(events)
    broken.pop(next(i for i, e in enumerate(broken) if e["kind"] == "resource_return"))
    with pytest.raises(ValueError):
        s.verify_events(broken, tasks, bank, 20, "signal")


def test_cap_and_failure_nonrefundable():
    meter = k.Meter(lambda e: None, lambda: None, lambda: None)
    meter.scope = [20, "uniform", "search", 0]
    with pytest.raises(RuntimeError):
        meter.call("metric", lambda: (_ for _ in ()).throw(RuntimeError("failure")))
    assert not meter.events[0]["completed"]
    with pytest.raises(ValueError):
        meter.call("metric", lambda: None)
    meter = k.Meter(lambda e: None, lambda: None, lambda: None)
    meter.scope = [20, "uniform", "search", 0]
    for _ in range(24):
        meter.call("metric", lambda: None)
    with pytest.raises(ValueError):
        meter.call("metric", lambda: None)


def test_partial_batch_is_failure_and_report_kept(monkeypatch, tmp_path):
    contract = tmp_path / "contract"
    data = {"host_prepare": "unused", "host_prepare_sha256": "unused",
            "slots": [list(x) for x in delivery.SLOTS]}
    contract.write_text(json.dumps(data))
    monkeypatch.setattr(delivery, "ROOT", tmp_path)
    monkeypatch.setattr(delivery, "exclusive", lambda path, prefix: path)
    monkeypatch.setattr(delivery, "material", lambda *_: data)
    monkeypatch.setattr(delivery, "run_deadline", lambda *_: {
        "exit_code": 1, "wall_seconds": 30., "deadline_exceeded": True,
        "stdout": "", "stderr": ""})
    out = tmp_path / "batch"
    with pytest.raises(ValueError, match="batch incomplete"):
        delivery.execute(contract, delivery.digest(contract), out)
    assert len(json.loads((out / "report.json").read_text())["rows"]) == 6
