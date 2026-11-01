import copy
import json

import pytest

from experiments.spectral_reachability import p1_uphill_imagined_midpoint_physical_20261031_main as p


def test_full_roster(monkeypatch):
    monkeypatch.setattr(p.prior, "PLAN", p.PLAN)
    monkeypatch.setattr(p.prior, "PLAN_SHA", p.PLAN_SHA)
    monkeypatch.setattr(p.prior, "NAME", p.NAME)
    monkeypatch.setattr(p.prior.planner, "NAME", p.planner.NAME)
    result = p.prior.build()
    assert len(result["roster"]) == 24
    assert not result["P1_admission"]


def test_rejects_first_packet_tamper(monkeypatch):
    old = json.loads(p.prior.OLD.read_bytes())
    plan = json.loads(p.PLAN.read_bytes())
    inputs, _ = p.prior.planner.inputs()
    monkeypatch.setattr(p.prior.planner, "NAME", p.planner.NAME)
    plan = copy.deepcopy(plan)
    plan["rows"][0]["first_packet_hex"] = "bad"
    with pytest.raises(Exception):
        p.prior.assemble(old, plan, inputs)
