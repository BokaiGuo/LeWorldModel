from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import codex_e1_learned_durable_runner_20261101_exclusive as m
from experiments.spectral_reachability.codex_e1_proposal_durable_events_20261101 import read_events


def events(path):
    return [x["payload"]["event"] for x in read_events(path)]


def fake_runner(path, fail=False):
    class Module:
        def __init__(self):
            self.pre = []
            self.post = []

        def register_forward_pre_hook(self, hook):
            self.pre.append(hook)
            return SimpleNamespace(remove=lambda: self.pre.remove(hook))

        def register_forward_hook(self, hook):
            self.post.append(hook)
            return SimpleNamespace(remove=lambda: self.post.remove(hook))

        def __call__(self, fail=False):
            for hook in self.pre:
                hook(self, ())
            if fail:
                raise ValueError("model failure")
            for hook in self.post:
                hook(self, (), 7)
            return 7

    class Trace:
        def call(self, kind, args, fn):
            return fn()

    def search(t, cem, box, root, z, goal, trace, row, index):
        def operation():
            assert [e["operation"] for e in events(path) if e["type"] == "attempt"] == [
                "run_cuda", "high_search", "trace.high_cost"]
            if fail:
                raise ValueError("model failure")
            return 7
        return trace.call("high_cost", [], operation)

    def run_cuda(manifest, sha, output, *, main_authorized):
        root = SimpleNamespace(**{name: Module() for name in m.HIGH_MODELS})
        return runner.high_search(None, None, None, root, None, None, Trace(), {}, 2)

    runner = SimpleNamespace(Trace=Trace, high_search=search, run_cuda=run_cuda)
    return runner


def invoke(runner, path, **kwargs):
    return m.run(runner, manifest="manifest", manifest_sha256="sha", output="output",
                 journal_path=path, run_id="new", arm="learned",
                 main_authorized=kwargs.get("authorized", True))


def test_real_call_path_records_nested_attempts_and_returns(tmp_path):
    path = tmp_path / "events.jsonl"
    runner = fake_runner(path)
    original_call, original_search = runner.Trace.call, runner.high_search
    assert invoke(runner, path) == 7
    rows = events(path)
    assert [e["type"] for e in rows] == ["cost_scope"] + ["attempt"] * 3 + ["return"] * 3
    assert [e["attempt_index"] for e in rows[4:]] == [3, 2, 1]
    assert rows[2]["details"]["row_index"] == 2
    assert rows[0]["scope"] == "shared_unallocated"
    assert rows[0]["amount"] is None and rows[0]["historical_costs"] == "unknown"
    assert runner.Trace.call is original_call and runner.high_search is original_search


def test_failure_durable_and_restored(tmp_path):
    path = tmp_path / "events.jsonl"
    runner = fake_runner(path, fail=True)
    original = runner.Trace.call
    with pytest.raises(ValueError, match="model failure"):
        invoke(runner, path)
    rows = events(path)
    assert [e["type"] for e in rows] == ["cost_scope"] + ["attempt"] * 3 + ["failure"] * 3
    assert [e["attempt_index"] for e in rows[4:]] == [3, 2, 1]
    assert runner.Trace.call is original


def test_existing_journal_and_authorization_fail_before_run(tmp_path):
    path = tmp_path / "events.jsonl"
    runner = fake_runner(path)
    with pytest.raises(ValueError, match="authorization"):
        invoke(runner, path, authorized=False)
    assert not path.exists()
    path.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        invoke(runner, path)
    assert path.read_bytes() == b"old"


def test_write_failure_prevents_actual_work(tmp_path, monkeypatch):
    from experiments.spectral_reachability import codex_e1_proposal_durable_events_20261101 as durable
    path = tmp_path / "events.jsonl"
    runner = fake_runner(path)
    monkeypatch.setattr(durable.os, "write", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        invoke(runner, path)
    assert path.read_bytes() == b""


def test_existing_runner_trace_is_instrumented(tmp_path, monkeypatch):
    from experiments.spectral_reachability import agent_hilewm_rgb_high_low_20261028_v1 as actual
    path = tmp_path / "events.jsonl"
    store = SimpleNamespace(guard=lambda: None, array=lambda value: "array")
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(actual, "precision", lambda t: None)
    monkeypatch.setattr(actual, "paid", SimpleNamespace(diag=SimpleNamespace(layout=lambda x: {})))
    trace = actual.Trace(torch, store, [])
    with m.ProposalJournal(path, run_id="new", arm="learned", proposal_kind="test") as journal:
        with m.instrument(actual, journal):
            def work():
                assert events(path)[0]["operation"] == "trace.high_cost"
                return 42
            assert trace.call("high_cost", [], work) == 42
    assert [e["type"] for e in events(path)] == ["attempt", "return"]
    assert trace.events[0]["returned"]


def test_model_calls_and_failed_forward_are_not_invented(tmp_path):
    import torch

    path = tmp_path / "events.jsonl"
    class Trace:
        def call(self, kind, args, fn):
            return fn()

    class Failing(torch.nn.Module):
        def forward(self, x):
            raise ValueError("forward failed")

    root = SimpleNamespace(macro_to_condition=torch.nn.Identity(),
                           high_predictor=Failing(),
                           high_pred_proj=torch.nn.Identity())
    def search(t, cem, box, root, z, goal, trace, row, index):
        trace.call("encoder", [], lambda: 1)
        root.macro_to_condition(torch.tensor([1.0]))
        root.high_predictor(torch.tensor([1.0]))

    runner = SimpleNamespace(Trace=Trace, high_search=search)
    with m.ProposalJournal(path, run_id="new", arm="learned", proposal_kind="test") as journal:
        with m.instrument(runner, journal):
            with pytest.raises(ValueError, match="forward failed"):
                runner.high_search(None, None, None, root, None, None, Trace(), {}, 3)
    rows = events(path)
    calls = [(i, e) for i, e in enumerate(rows) if e.get("operation") == "model_call"]
    assert [e["details"]["model"] for _, e in calls] == [
        "encoder", "macro_to_condition", "high_predictor"]
    returned = {e["attempt_index"] for e in rows if e["type"] == "return"}
    assert calls[0][0] in returned and calls[1][0] in returned
    assert calls[2][0] not in returned
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in (root.macro_to_condition, root.high_predictor, root.high_pred_proj))
