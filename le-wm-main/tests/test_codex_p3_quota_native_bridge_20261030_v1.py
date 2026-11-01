from contextlib import contextmanager
import pytest

from experiments.spectral_reachability import codex_p3_quota_native_bridge_20261030_v1 as b


class Backend:
    def __init__(self, fail=None, omit=False):
        self.fail, self.omit = fail, omit

    @contextmanager
    def guard(self, meter):
        yield

    def rollout(self, task, raw, meter, mode):
        for i in range(2):
            for axis in b.AXES:
                if self.omit and axis == "pred_proj":
                    continue
                meter.call(axis, lambda: None)
            if self.fail == i:
                raise RuntimeError("failed rollout")
        return [task["id"], raw]


def inputs():
    tasks = [{"id": 0}, {"id": 1}]
    banks = [[{"index": i, "commands": [[float(i) / 32, 0.] for _ in range(10)]}
              for i in range(15)] for _ in range(2)]
    return tasks, banks


def execute(backends, events):
    tasks, banks = inputs()
    meter = b.Meter(events.append, lambda: None, lambda: None)
    result = b._execute(tasks, banks, [2., 1.], backends, meter,
                        check_path=lambda task, raw, path: None,
                        cost=lambda task, path: -path[1][0][0])
    return result, meter


def test_native_meter_and_separate_root_winners():
    events = []
    result, meter = execute([Backend(), Backend()], events)
    assert [r["quotas"] for r in result] == [[10, 10], [15, 5], [5, 15]]
    assert [[w["index"] for w in r["winners"]] for r in result] == [[9, 9], [14, 4], [4, 14]]
    assert len(meter.events) == 720
    assert all(e["completed"] for e in meter.events)
    assert sum(e["kind"] == "resource_attempt" for e in events) == 720
    assert sum(e["kind"] == "resource_return" for e in events) == 720


def test_missing_actual_call_refused():
    events = []
    with pytest.raises(ValueError, match="operation vector"):
        execute([Backend(omit=True), Backend()], events)
    assert events[-1]["kind"] == "panel_failure"
    assert events[-1]["entry_count"] == 6


def test_failure_preserves_charged_prefix_no_retry():
    events = []
    with pytest.raises(RuntimeError):
        execute([Backend(fail=0), Backend()], events)
    assert events[-1]["kind"] == "panel_failure"
    assert events[-1]["entry_count"] == 4
    assert not any(e["kind"] == "candidate_complete" for e in events)


def test_cpu_host_rejected_without_model():
    with pytest.raises(ValueError, match="CUDA"):
        b.execute_host(None, type("D", (), {"type": "cpu"})(), [], [], [], None, None, None)


def test_predispatch_cap_and_failed_call_accounting():
    events = []
    meter = b.Meter(events.append, lambda: None, lambda: None)
    meter.scope = ["signal", 0, 0]
    meter.call("predict", lambda: None)
    meter.call("predict", lambda: None)
    with pytest.raises(ValueError, match="candidate axis cap"):
        meter.call("predict", lambda: None)
    assert len(meter.events) == 2
    meter.scope = ["signal", 0, 1]
    with pytest.raises(RuntimeError):
        meter.call("predict", lambda: (_ for _ in ()).throw(RuntimeError("failed")))
    assert meter.events[-1]["completed"] is False
    with pytest.raises(ValueError, match="meter resource"):
        meter.call("predict", lambda: None)
