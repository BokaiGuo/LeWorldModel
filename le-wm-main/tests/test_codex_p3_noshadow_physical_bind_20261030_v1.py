import copy
import pytest
from experiments.spectral_reachability import codex_p3_noshadow_physical_bind_20261030_v1 as b
from test_codex_p3_noshadow_20261030_v1 import fixture


def test_binding_denominator_and_cost(monkeypatch):
    events, _, _ = fixture(monkeypatch)
    slots = [{"root_sha256": str(i), "task_id": str(i)} for i in (0, 1)]
    refs = [{"root_sha256": str(i)} for i in (0, 1)]
    rows = b.summarize(events, slots, refs)
    assert len(rows) == 6
    assert [r["actual_resource_attempts"]["predict"] for r in rows] == [20, 30, 10, 20, 10, 30]
    assert all(r["physical_authorized"] is False for r in rows)
    bad = copy.deepcopy(events)
    next(e for e in bad if e["kind"] == "arm_complete")["winners"][0]["commands"][0][0] = 2.
    with pytest.raises(ValueError):
        b.summarize(bad, slots, refs)
    refs[0]["root_sha256"] = "wrong"
    with pytest.raises(ValueError):
        b.summarize(events, slots, refs)
