import json

import pytest

from experiments.spectral_reachability import codex_p1_cp5_durable_budget_20261030_v1 as m


def test_cap_block_is_persisted_without_dispatch(tmp_path):
    path = tmp_path / "events.jsonl"
    meter = m.Budget(path, dict.fromkeys(m.AXES, 1), 10)
    assert meter.call("encoder", 1, lambda: 7) == 7
    with pytest.raises(ValueError, match="cap"):
        meter.call("encoder", 1, lambda: pytest.fail("must not execute"))
    meter.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["kind"] for e in events] == ["header", "attempt", "return", "blocked", "terminal"]
    assert events[-1]["payload"]["status"] == "failed"
    assert events[-1]["payload"]["attempted"]["projector"] == 0
    assert all(events[i]["previous"] == events[i-1]["sha256"] for i in range(1, len(events)))
    with pytest.raises(FileExistsError):
        m.Budget(path, dict.fromkeys(m.AXES, 1), 10)


def test_failure_and_wall_block_remain_distinct(tmp_path):
    meter = m.Budget(tmp_path/"failure", dict.fromkeys(m.AXES, 1), 10)
    with pytest.raises(RuntimeError):
        meter.call("projector", 1, lambda: (_ for _ in ()).throw(RuntimeError("fault")))
    meter.close()
    assert json.loads((tmp_path/"failure").read_text().splitlines()[-2])["kind"] == "failure"
    times = iter([0., 11.])
    meter = m.Budget(tmp_path/"wall", dict.fromkeys(m.AXES, 1), 10, clock=lambda: next(times))
    with pytest.raises(ValueError, match="cap"):
        meter.call("encoder", 1, lambda: None)
    meter.close()
    assert json.loads((tmp_path/"wall").read_text().splitlines()[-2])["kind"] == "blocked"
