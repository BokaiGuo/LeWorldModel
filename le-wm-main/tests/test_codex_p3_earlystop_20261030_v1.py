from contextlib import contextmanager
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p3_earlystop_kernel_20261030_v1 as k
from experiments.spectral_reachability import codex_p3_earlystop_saved_20261030_v1 as s
from experiments.spectral_reachability import codex_p3_earlystop_delivery_20261030_v1 as delivery


class Backend:
    @contextmanager
    def guard(self, meter):
        yield

    def rollout(self, task, commands, meter, mode):
        for _ in range(2):
            for axis in k.NEURAL:
                meter.call(axis, lambda: None)
        return [dict(latent_history=[[0.] * 192], value=commands[0][0])]

    def jacobian(self, task, incumbent, meter):
        for _ in range(2):
            for axis in k.NEURAL:
                meter.call(axis, lambda: None)
        for i in range(192):
            for axis in ("vjp_directions", "grad_api", "raw_backward_hook"):
                meter.call(axis, lambda: None)
            meter.emit("jacobian_row", row=i, array=k.native.blob(np.zeros(20)))
        return np.zeros(192), np.zeros((192, 20))


def run_fixture(monkeypatch, values, arm="uniform"):
    monkeypatch.setattr(k, "POINTS", (20,))
    monkeypatch.setattr(k, "ARMS", (arm,))
    tasks = [{"weight": 2.}, {"weight": 1.}]
    bank = [[dict(index=i, commands=[[float(values[root][i]), 0.]] * 10)
             for i in range(30)] for root in range(2)]
    events = []
    meter = k.Meter(events.append, lambda: None, lambda: None)
    k.execute(tasks, bank, [Backend(), Backend()], meter, check_path=lambda *a: None,
              signal_fn=lambda task, *_: {"w": task["weight"]},
              cost=lambda task, path: path[0]["value"])
    monkeypatch.setattr(s.native, "check_path", lambda *a: None)
    monkeypatch.setattr(s.native, "signal", lambda task, *_: {"w": task["weight"]})
    monkeypatch.setattr(s.native, "cost", lambda task, path: path[0]["value"])
    return events, tasks, bank


def test_earlystop_tie_and_reset_and_quota(monkeypatch):
    values = [[0.] * 30, [10., 10., 10., 9.] + list(range(8, -18, -1))]
    events, tasks, bank = run_fixture(monkeypatch, values)
    row = s.verify_events(events, tasks, bank, 20, "uniform")["rows"][0]
    assert row["allocation"] == [12, 12]
    assert row["actual_allocation"] == [6, 12]
    assert row["unused_allocation"] == [6, 0]
    assert row["stop_reasons"] == ["patience_exhausted", "quota_exhausted"]
    assert row["attempted"]["metric"] == 18
    assert row["attempted"]["predict"] == 36
    assert row["attempted"]["jacobian"] == 0
    assert row["attempted"]["scheduler"] == 1


def test_signal_cost_is_not_padded(monkeypatch):
    events, tasks, bank = run_fixture(monkeypatch, [[0.] * 30] * 2, "signal")
    row = s.verify_events(events, tasks, bank, 20, "signal")["rows"][0]
    assert row["actual_allocation"] == [6, 6]
    assert row["attempted"]["predict"] == 32
    assert row["attempted"]["metric"] == 12
    assert row["attempted"]["vjp_directions"] == 384


@pytest.mark.parametrize("kind", ["stop", "missing", "reorder", "ledger", "extra"])
def test_saved_rejects_tamper(monkeypatch, kind):
    events, tasks, bank = run_fixture(monkeypatch, [[0.] * 30] * 2)
    events = deepcopy(events)
    i = next(i for i, e in enumerate(events) if e["kind"] == "root_search_complete")
    if kind == "stop":
        events[i]["reason"] = "quota_exhausted"
    elif kind == "missing":
        events.pop(i)
    elif kind == "reorder":
        events[i], events[i+1] = events[i+1], events[i]
    elif kind == "ledger":
        events.pop(next(i for i, e in enumerate(events) if e["kind"] == "resource_return"))
    else:
        events.insert(i, deepcopy(next(e for e in events if e["kind"] == "candidate_attempt")))
    with pytest.raises((ValueError, KeyError)):
        s.verify_events(events, tasks, bank, 20, "uniform")


def test_nonrefundable_failure():
    meter = k.Meter(lambda e: None, lambda: None, lambda: None)
    meter.scope = [20, "uniform", "search", 0]
    with pytest.raises(RuntimeError):
        meter.call("metric", lambda: (_ for _ in ()).throw(RuntimeError("failed")))
    assert len(meter.events) == 1 and not meter.events[0]["completed"]
    with pytest.raises(ValueError):
        meter.call("metric", lambda: None)


def test_delivery_fixed_slots():
    assert len(delivery.SLOTS) == 6 and delivery.SECONDS == 30


def test_partial_batch_keeps_report_and_fails(monkeypatch, tmp_path):
    contract = tmp_path / "contract.json"
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
    report = json.loads((out / "report.json").read_text())
    assert len(report["rows"]) == 6
    assert all(not row["delivered_and_saved"] for row in report["rows"])
