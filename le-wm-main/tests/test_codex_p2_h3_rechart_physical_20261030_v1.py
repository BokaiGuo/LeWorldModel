import numpy as np
import pytest
from experiments.spectral_reachability import codex_p2_h3_rechart_physical_20261030_v1 as m


def test_actual_bound_eight_h3_plans():
    b = m.build()
    assert len(b["plans"]) == 8
    assert all(p["planned_commands"] == 15 and not p["original_raw_bits_proved"] for p in b["plans"])
    assert b["budgets"]["selected"] == 120
    p = b["plans"][0].copy()
    p["commands"] = [[2.,0.]]*15
    with pytest.raises(ValueError): m.physical.validate_plan(p)


def test_h3_meter_per_slot_cap_and_failure():
    class Env:
        def step(self, action): return None
    events = []
    meter = m.Meter(Env(), events.append)
    meter.bind(dict(root_sha256="r", method="joint", selection_sha256="x", planned_commands=15))
    meter.phase = "selected"
    for _ in range(15): meter.step(np.zeros(2))
    with pytest.raises(ValueError): meter.step(np.zeros(2))
    assert meter.steps["selected"] == dict(attempts=15, returns=15)
    assert len(events) == 30


def test_constructor_failure_keeps_eight_slots(tmp_path, monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    def fail(*args, **kwargs): raise RuntimeError("constructor failure")
    monkeypatch.setattr(base.gym, "make", fail)
    out = tmp_path/(m.NAME+"_failure")
    report = m.execute(out)
    assert report["status"] == "partial_or_failed"
    assert len(report["rows"]) == 8 and all(r["status"] == "blocked" for r in report["rows"])
    assert m.consume(out, m.chart.prior.checks.run.digest_file(out/"report.json"))["rows"] == 8
