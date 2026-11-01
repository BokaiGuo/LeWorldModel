"""Joint attestation + bridge integration on saved data and CPU-only fakes."""
import copy
import hashlib
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_budgeted_manifest_consumer_20261030_codex as joint
from experiments.spectral_reachability import p2_h15_budgeted_plan_20261030_codex as plan
from experiments.spectral_reachability import p2_h15_cuda_ledger_runner_20261030_codex as runner
from experiments.spectral_reachability import p2_h15_observable_budget_bridge_20261030_codex as bridge
from experiments.spectral_reachability import p2_h15_production_manifest_20261030_codex as production
from experiments.spectral_reachability import p2_hook_contract_bridge_20261030_codex as prior
from test_p2_h15_budgeted_plan_20261030_codex import Clock
from test_p2_h15_neural_ledger_adapter_20261030_codex import saved_signal, contract
from test_p2_h15_production_manifest_20261030_codex import FakeModel, fake_torch
from test_p2_h15_strict_saved_consumer_20261030_codex import rewrite_chain


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    root = tmp_path_factory.mktemp("joint_h15")
    report, task, _, _ = saved_signal()
    raw = (prior.HOST / "root0" / "events.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == report["journal_sha256"]
    calls, active = [], None
    for row in durable.read_chain(raw):
        event = row["payload"]
        if event["kind"] == "native_call_attempt":
            active = [event]
        elif active is not None:
            active.append(event)
            if event["kind"] == "native_call_return":
                calls.append(active)
                active = None
    assert len(calls) == 43
    model, torch = FakeModel(), fake_torch()
    hashes = {"original_state_dict": task["binding"]["model_sha256"], "all_buffers": "b" * 64}
    c = contract(task)
    base = production.manifest(c, task, hashes, production.capture_classes(model))
    envelope = joint.manifest(base, core.digest(base), c, task)

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
            mapping = {old: first + i for i, old in enumerate(starts)}
            for old in body:
                if old["kind"] in native.RESOURCE:
                    continue
                event = copy.deepcopy(old)
                event["scope"] = ledger.scope
                if event.get("operation_index") is not None:
                    event["operation_index"] = mapping[event["operation_index"]]
                self.store.event(event)
            return call[-1]["result"]
        def signal(self, request, ledger):
            return self.invoke("signal", request, ledger)
        def replay(self, context, actions, segmented, ledger):
            return self.invoke("replay", dict(parent_context=context, actions=actions, segmented=segmented), ledger)

    b = bridge.Bridge(root / "budget", c, core.digest(c), "joint", "joint-fixture", clock=Clock())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner.signal, "model_hash", lambda _: hashes.copy())
        patch.setattr(runner.tree, "NativeTree", SavedBackend)
        r = runner.Runner(root / "run", c, core.digest(c), dict.fromkeys(core.AXES, 10000),
                          "joint", "joint-fixture", task, model, SimpleNamespace(type="cuda"))
        att = production.Attestation(r, base, core.digest(base), torch)
        p = plan.Plan(r, b, expected_source_sha256=envelope["extension_sources"]["plan_sha256"])
        result = p.execute(lambda: None)
        ro = r.finish()
        bo = b.finish(ro)
        po = p.finish(ro, bo)
        ao = att.finish(ro)
    options = dict(manifest_value=envelope, manifest_sha256=core.digest(envelope),
                   contract=c, expected_task=task, expected_task_sha256=core.digest(task),
                   attestation_sha256=ao["sha256"], native_sha256=ro["native_io_sha256"],
                   ledger_sha256=ro["sha256"], budget_sha256=bo["sha256"], plan_sha256=po["sha256"],
                   arm="joint", run_id="joint-fixture")
    return root / "run", root / "budget", options, result


def test_end_to_end_attestation_and_budgeted_saved_consumer(integrated):
    run, budget_dir, options, result = integrated
    check = joint.consume(run, budget_dir, **options)
    assert check["budgeted_saved"]["winner"] == result["winner"]
    assert check["budgeted_saved"]["budget"]["counts"]["replay_calls"] == dict(attempted=42, completed=42)
    assert check["model_class_source_snapshots_recorded"] and check["precision_snapshots_recorded"]
    assert not check["continuous_precision_verified"] and not check["runtime_dependency_closure_verified"]
    assert not check["budget_fairness"] and not check["ready_to_execute"] and not check["P2_admission"]


@pytest.mark.parametrize("key", ["manifest_sha256", "attestation_sha256", "native_sha256",
                                 "ledger_sha256", "budget_sha256", "plan_sha256", "expected_task_sha256"])
def test_external_anchor_mismatch_fails(integrated, key):
    run, budget_dir, options, _ = integrated
    with pytest.raises(ValueError):
        joint.consume(run, budget_dir, **dict(options, **{key: "0" * 64}))


@pytest.mark.parametrize("change", ["precision", "class_source", "plan_source", "bridge_source", "consumer_source", "axis"])
def test_recommitted_manifest_tamper_rejected(integrated, change):
    run, budget_dir, options, _ = integrated
    envelope = copy.deepcopy(options["manifest_value"])
    base = envelope["base"]
    if change == "precision":
        base["precision"]["matmul_TF32"] = True
    elif change == "class_source":
        base["classes"][0]["sha256"] = "0" * 64
    elif change == "axis":
        base["axes"]["replay_calls"]["cap"] += 1
    else:
        key = {"plan_source": "plan_sha256", "bridge_source": "bridge_sha256",
               "consumer_source": "budget_consumer_sha256"}[change]
        envelope["extension_sources"][key] = "0" * 64
    envelope["base_sha256"] = core.digest(base)
    with pytest.raises(ValueError):
        joint.consume(run, budget_dir, **dict(options, manifest_value=envelope,
                                              manifest_sha256=core.digest(envelope)))


@pytest.mark.parametrize("change", ["before_precision", "after_class", "native_link", "missing_after"])
def test_rechained_attestation_tamper_fails(integrated, tmp_path, change):
    run, budget_dir, options, _ = integrated
    target = tmp_path / "run"
    target.mkdir()
    for name in ("production_attestation.jsonl", "budgeted_plan.jsonl", "events.jsonl", "native_io.jsonl"):
        (target / name).write_bytes((run / name).read_bytes())
    def mutate(rows):
        if change == "before_precision":
            rows[0]["payload"]["snapshot"]["precision"]["autocast_cuda"] = True
        elif change == "after_class":
            rows[1]["payload"]["snapshot"]["classes"][0]["class_name"] = "forged.Class"
        elif change == "native_link":
            rows[1]["payload"]["native_sha256"] = "0" * 64
        else:
            rows.pop()
    changed = rewrite_chain(target / "production_attestation.jsonl", mutate)
    with pytest.raises(ValueError):
        joint.consume(target, budget_dir, **dict(options, attestation_sha256=changed))


def test_rechained_budget_and_plan_rejected(integrated, tmp_path):
    run, budget_dir, options, _ = integrated
    target, target_budget = tmp_path / "run", tmp_path / "budget"
    target.mkdir()
    target_budget.mkdir()
    for name in ("production_attestation.jsonl", "budgeted_plan.jsonl", "events.jsonl", "native_io.jsonl"):
        (target / name).write_bytes((run / name).read_bytes())
    path = target_budget / "budget_events.jsonl"
    path.write_bytes((budget_dir / "budget_events.jsonl").read_bytes())
    def alter_budget(rows):
        rows[-1]["payload"]["counts"]["replay_calls"]["completed"] = 41
    budget_sha = rewrite_chain(path, alter_budget)
    def relink(rows):
        rows[-1]["payload"]["budget_sha256"] = budget_sha
    plan_sha = rewrite_chain(target / "budgeted_plan.jsonl", relink)
    with pytest.raises(ValueError, match="totals"):
        joint.consume(target, target_budget, **dict(options, budget_sha256=budget_sha, plan_sha256=plan_sha))
