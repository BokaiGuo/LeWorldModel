from contextlib import nullcontext
from types import SimpleNamespace

from experiments.spectral_reachability import p1_backtrack_physical_runner_20261030 as runner


class FakeStore:
    def __init__(self, path):
        self.path, self.previous = path, None
        self.events = []
        (path / "journal.jsonl").write_text("fixture")

    def put(self, name, value, final=False):
        self.events.append(("put", name, value))

    def emit(self, kind, **fields):
        self.events.append((kind, fields))
        self.previous = "a" * 64

    def check(self, *args):
        return None


def fixture(monkeypatch, tmp_path, fail_at):
    plans = [dict(index=i, root_slot=i // 8, condition="off", arm="arm", role="role",
                  task_id="a" * 64) for i in range(16)]
    p = dict(contract=runner.contract.contract(), plans=plans, bindings={})
    data = {"roots": [object(), object()], "source_plan": {}}
    env = lambda: SimpleNamespace(unwrapped=SimpleNamespace(render_action=False, render_size=224,
                                                            render_mode="rgb_array"), close=lambda: None)
    monkeypatch.setattr(runner.native, "validate_env", lambda *_: None)
    monkeypatch.setattr(runner.native, "native_hooks", lambda *_: nullcontext())
    monkeypatch.setattr(runner, "census", lambda *_: None)
    def run_episode(_env, _root, _source, plan, row, meter, _store):
        if plan["index"] == fail_at:
            raise ValueError("fixture failure")
        meter.call("reset", lambda: None, role="placement", seed=0)
        meter.call("public_step", lambda: None, phase="plan", step=1)
        row["status"] = "horizon_completed"
    monkeypatch.setattr(runner.episode, "episode", run_episode)
    return p, data, env, FakeStore(tmp_path)


def test_fixed_denominator_and_receipt_indices(monkeypatch, tmp_path):
    p, data, factory, store = fixture(monkeypatch, tmp_path, None)
    report = runner.execute(p, data, store, factory, "/prepare", "a" * 64, authorization=runner.AUTH)
    assert report["status"] == "physical_recorded_not_independently_consumed"
    assert len(report["rows"]) == 16
    assert [r["index"] for r in report["receipts"]] == list(range(len(report["receipts"])))
    assert all(r["own_returned"] == 1 for r in report["rows"])
    assert all(r["role"] == "placement" and r["candidate_role"] == "role"
               for r in report["receipts"] if r["resource"] == "reset")
    assert report["P1_admission"] is False


def test_failure_stops_without_retry_or_dropping_remaining_rows(monkeypatch, tmp_path):
    p, data, factory, store = fixture(monkeypatch, tmp_path, 2)
    report = runner.execute(p, data, store, factory, "/prepare", "a" * 64, authorization=runner.AUTH)
    assert report["status"] == "partial_or_failed"
    assert report["rows"][2]["status"] == "engineering_failed"
    assert all(row["status"] == "not_started" for row in report["rows"][3:])
    assert report["counts"]["constructor"]["attempted"] == 3
    assert report["counts"]["close"]["completed"] == 3
