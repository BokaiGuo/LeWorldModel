"""CPU-only saved native transcript replay; no real model, GPU or physics."""
import copy
import hashlib
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_budgeted_plan_20261030_codex as plan
from experiments.spectral_reachability import p2_h15_budgeted_saved_consumer_20261030_codex as consumer
from experiments.spectral_reachability import p2_h15_cuda_ledger_runner_20261030_codex as runner
from experiments.spectral_reachability import p2_h15_observable_budget_bridge_20261030_codex as bridge
from experiments.spectral_reachability import p2_h15_strict_saved_consumer_20261030_codex as old
from experiments.spectral_reachability import p2_hook_contract_bridge_20261030_codex as prior
from test_p2_h15_neural_ledger_adapter_20261030_codex import saved_signal, contract
from test_p2_h15_strict_saved_consumer_20261030_codex import rewrite_chain


class Clock:
    def __init__(self):
        self.now = 100
    def __call__(self):
        return self.now


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("budgeted_plan")
    report, task, _, _ = saved_signal()
    raw = (prior.HOST / "root0" / "events.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == report["journal_sha256"]
    events = [row["payload"] for row in durable.read_chain(raw)]
    calls, active = [], None
    for event in events:
        if event["kind"] == "native_call_attempt":
            active = [event]
        elif active is not None:
            active.append(event)
            if event["kind"] == "native_call_return":
                calls.append(active)
                active = None
    assert len(calls) == 43
    model = object()
    hashes = {"original_state_dict": task["binding"]["model_sha256"], "all_buffers": "b" * 64}
    c = contract(task)
    sha = core.digest(c)
    manifest = old.source_manifest()

    class SavedBackend:
        def __init__(self, supplied_model, device, supplied_task, store):
            assert supplied_model is model and supplied_task == task and device.type == "cuda"
            self.store, self.index = store, 0
        def invoke(self, method, request, ledger):
            call = calls[self.index]
            self.index += 1
            assert call[0]["method"] == method and call[0]["request"] == request
            body = call[1:-1]
            starts = [x["entry"]["index"] for x in body if x["kind"] == "resource_attempt"]
            first = len(ledger.events)
            native.replay_resources(body, ledger)
            mapping = {old_index: first + i for i, old_index in enumerate(starts)}
            for old_event in body:
                if old_event["kind"] in native.RESOURCE:
                    continue
                event = copy.deepcopy(old_event)
                event["scope"] = ledger.scope
                if event.get("operation_index") is not None:
                    event["operation_index"] = mapping[event["operation_index"]]
                self.store.event(event)
            return call[-1]["result"]
        def signal(self, request, ledger):
            return self.invoke("signal", request, ledger)
        def replay(self, context, actions, segmented, ledger):
            return self.invoke("replay", dict(parent_context=context, actions=actions, segmented=segmented), ledger)

    clock = Clock()
    b = bridge.Bridge(root / "budget", c, sha, "joint", "saved-fixture", clock=clock)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner.signal, "model_hash", lambda _: hashes.copy())
        patch.setattr(runner.tree, "NativeTree", SavedBackend)
        r = runner.Runner(root / "run", c, sha, dict.fromkeys(core.AXES, 10000), "joint",
                          "saved-fixture", task, model, SimpleNamespace(type="cuda"))
        p = plan.Plan(r, b, expected_source_sha256=plan.source_hash())
        result = p.execute(lambda: None)
        output = r.finish()
        bout = b.finish(output)
        pout = p.finish(output, bout)
    args = dict(native_sha256=output["native_io_sha256"], ledger_sha256=output["sha256"],
                budget_sha256=bout["sha256"], plan_sha256=pout["sha256"], contract_sha256=sha,
                expected_task=task, expected_task_sha256=core.digest(task), expected_model_hashes=hashes,
                expected_manifest=manifest, expected_manifest_sha256=core.digest(manifest),
                expected_plan_source_sha256=plan.source_hash(), expected_bridge_source_sha256=bridge.source_hash(),
                arm="joint", run_id="saved-fixture")
    return root / "run", root / "budget", args, result


def test_complete_budgeted_saved_transcript(completed):
    run_dir, budget_dir, args, plan_result = completed
    out = consumer.consume(run_dir, budget_dir, **args)
    assert out["winner"] == plan_result["winner"] and out["native_calls"] == 43
    assert out["budget"]["counts"]["replay_calls"] == dict(attempted=42, completed=42)
    assert out["budget"]["counts"]["physical_steps"] == dict(attempted=0, completed=0)
    assert out["resources"]["completed"]["predict_samples"] == 132
    assert out["resources"]["completed"]["VJP_directions"] == 192
    assert out["resources"]["completed"]["metric_calls"] == 40
    assert out["resources"]["completed"]["replay_calls"] == 42
    assert not out["budget_fairness"] and not out["coverage_verified"] and not out["P2_admission"]


@pytest.mark.parametrize("key", ["native_sha256", "ledger_sha256", "budget_sha256", "plan_sha256",
                                 "contract_sha256", "expected_task_sha256", "expected_manifest_sha256",
                                 "expected_plan_source_sha256", "expected_bridge_source_sha256"])
def test_external_anchor_mismatch(completed, key):
    run_dir, budget_dir, args, _ = completed
    with pytest.raises(ValueError):
        consumer.consume(run_dir, budget_dir, **dict(args, **{key: "0" * 64}))


def test_rechained_plan_winner_tamper(completed, tmp_path):
    run_dir, budget_dir, args, _ = completed
    target = tmp_path / "run"
    target.mkdir()
    for name in ("native_io.jsonl", "events.jsonl", "budgeted_plan.jsonl"):
        (target / name).write_bytes((run_dir / name).read_bytes())
    def change(rows):
        rows[1]["payload"]["winner"] = (rows[1]["payload"]["winner"] + 1) % 40
    sha = rewrite_chain(target / "budgeted_plan.jsonl", change)
    with pytest.raises(ValueError, match="winner"):
        consumer.consume(target, budget_dir, **dict(args, plan_sha256=sha))


def test_rechained_budget_count_tamper(completed, tmp_path):
    run_dir, budget_dir, args, _ = completed
    target = tmp_path / "budget"
    target.mkdir()
    path = target / "budget_events.jsonl"
    path.write_bytes((budget_dir / "budget_events.jsonl").read_bytes())
    def change(rows):
        rows[-1]["payload"]["counts"]["replay_calls"]["completed"] = 41
    sha = rewrite_chain(path, change)
    with pytest.raises(ValueError, match="totals"):
        consumer.consume(run_dir, target, **dict(args, budget_sha256=sha))


def test_rechained_native_buffer_tamper(completed, tmp_path):
    run_dir, budget_dir, args, _ = completed
    target = tmp_path / "run"
    target.mkdir()
    for name in ("native_io.jsonl", "events.jsonl", "budgeted_plan.jsonl"):
        (target / name).write_bytes((run_dir / name).read_bytes())
    def change(rows):
        event = next(x["payload"] for x in rows if x["kind"] == "native"
                     and x["payload"].get("kind") == "io_input")
        next(iter(event["arrays"].values()))["sha256"] = "0" * 64
    sha = rewrite_chain(target / "native_io.jsonl", change)
    def relink(rows):
        rows[-1]["payload"]["native_sha256"] = sha
    plan_sha = rewrite_chain(target / "budgeted_plan.jsonl", relink)
    with pytest.raises(ValueError):
        consumer.consume(target, budget_dir, **dict(args, native_sha256=sha, plan_sha256=plan_sha))


def test_replay_cap_stops_plan_before_third_callback(tmp_path, monkeypatch):
    report, task, _, signal = saved_signal()
    c = contract(task)
    c["axes"]["replay_calls"]["cap"] = 2
    sha, model = core.digest(c), object()
    hashes = {"original_state_dict": task["binding"]["model_sha256"], "all_buffers": "b" * 64}
    class Stub:
        def __init__(self, *_):
            self.calls = 0
        def signal(self, request, ledger):
            return ledger.call("predict", lambda: signal, shape=[1, 3, 192])
        def replay(self, context, actions, segmented, ledger):
            index = self.calls
            self.calls += 1
            return ledger.call("predict", lambda: report["result"]["candidates"][index]["path"],
                               shape=[1, 3, 192])
    monkeypatch.setattr(runner.signal, "model_hash", lambda _: hashes.copy())
    monkeypatch.setattr(runner.tree, "NativeTree", Stub)
    b = bridge.Bridge(tmp_path / "budget", c, sha, "joint", "cap", clock=Clock())
    r = runner.Runner(tmp_path / "run", c, sha, dict.fromkeys(core.AXES, 10000),
                      "joint", "cap", task, model, SimpleNamespace(type="cuda"))
    p = plan.Plan(r, b, expected_source_sha256=plan.source_hash())
    with pytest.raises(core.BudgetExceeded):
        p.execute(lambda: None)
    assert r.backend.calls == 2 and r.cpu.next_candidate == 2
    assert b.counts["replay_calls"] == dict(attempted=2, completed=2)
    output = r.finish()
    bout = b.finish(output)
    assert output["status"] == bout["status"] == "failed_or_blocked"
    with pytest.raises(ValueError, match="completed plan"):
        p.finish(output, bout)


def test_rechained_replay_outside_its_outer_call_is_rejected(completed, tmp_path):
    run_dir, budget_dir, args, _ = completed
    target, target_budget = tmp_path / "run", tmp_path / "budget"
    target.mkdir()
    target_budget.mkdir()
    for name in ("native_io.jsonl", "events.jsonl", "budgeted_plan.jsonl"):
        (target / name).write_bytes((run_dir / name).read_bytes())
    (target_budget / "budget_events.jsonl").write_bytes((budget_dir / "budget_events.jsonl").read_bytes())
    def move_return(rows):
        start = next(i for i, x in enumerate(rows) if x["kind"] == "resource"
                     and x["payload"]["kind"] == "resource_attempt"
                     and x["payload"]["entry"]["resource"] == "replay")
        op_index = rows[start]["payload"]["entry"]["index"]
        end = next(i for i in range(start + 1, len(rows)) if rows[i]["kind"] == "resource"
                   and rows[i]["payload"]["kind"] == "resource_return"
                   and rows[i]["payload"]["entry"]["index"] == op_index)
        rows.insert(start + 1, rows.pop(end))
    ledger_sha = rewrite_chain(target / "events.jsonl", move_return)
    def update_io(rows):
        rows[-1]["payload"]["ledger_sha256"] = ledger_sha
    io_sha = rewrite_chain(target / "native_io.jsonl", update_io)
    def update_budget(rows):
        rows[-1]["payload"]["ledger_sha256"] = ledger_sha
    budget_sha = rewrite_chain(target_budget / "budget_events.jsonl", update_budget)
    def update_plan(rows):
        rows[-1]["payload"].update(native_sha256=io_sha, ledger_sha256=ledger_sha, budget_sha256=budget_sha)
    plan_sha = rewrite_chain(target / "budgeted_plan.jsonl", update_plan)
    with pytest.raises(ValueError, match="nested in matching outer replay"):
        consumer.consume(target, target_budget, **dict(args, ledger_sha256=ledger_sha, native_sha256=io_sha,
                                                       budget_sha256=budget_sha, plan_sha256=plan_sha))
