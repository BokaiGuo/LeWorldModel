"""Historical saved transcript replay only; no neural, CUDA, or physics work."""
import copy
import hashlib
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_cuda_ledger_runner_20261030_codex as runner
from experiments.spectral_reachability import p2_h15_strict_saved_consumer_20261030_codex as strict
from experiments.spectral_reachability import p2_hook_contract_bridge_20261030_codex as prior
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec
from test_p2_h15_neural_ledger_adapter_20261030_codex import saved_signal, contract
from test_p2_h15_cuda_ledger_runner_20261030_codex import setup as stub_setup


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("h15_strict") / "run"
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
    assert len(calls) == 43 and active is None
    model = object()
    hashes = {"original_state_dict": task["binding"]["model_sha256"], "all_buffers": "b" * 64}
    c = contract(task)
    contract_sha = spec.sha(spec.canonical(c))
    manifest = strict.source_manifest()

    class SavedOnlyBackend:
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
            native.replay_resources(body, ledger)  # inert callbacks; not a model
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

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner.signal, "model_hash", lambda _: hashes.copy())
        patch.setattr(runner.tree, "NativeTree", SavedOnlyBackend)
        r = runner.Runner(directory, c, contract_sha, dict.fromkeys(core.AXES, 10000), "joint",
                          "saved-fixture", task, model, SimpleNamespace(type="cuda"))
        r.plan()
        result = r.finish()
    args = dict(native_sha256=result["native_io_sha256"], ledger_sha256=result["sha256"],
                contract_sha256=contract_sha, expected_task=task, expected_task_sha256=core.digest(task),
                expected_model_hashes=hashes, expected_manifest=manifest,
                expected_manifest_sha256=core.digest(manifest), arm="joint", run_id="saved-fixture")
    return directory, args


def test_full_saved_transcript_and_resource_closure(fixture):
    directory, args = fixture
    result = strict.consume(directory, **args)
    assert result["native_calls"] == 43
    assert result["resources"]["completed"]["predict_samples"] == 132
    assert result["resources"]["completed"]["VJP_directions"] == 192
    assert result["resources"]["completed"]["metric_calls"] == 40
    assert not result["coverage_verified"] and not result["budget_fairness"]
    assert not result["runtime_model_source_closure_verified"] and not result["P2_admission"]


@pytest.mark.parametrize("key,bad", [
    ("native_sha256", "0" * 64), ("ledger_sha256", "0" * 64),
    ("contract_sha256", "0" * 64), ("expected_task_sha256", "0" * 64),
    ("expected_manifest_sha256", "0" * 64), ("arm", "composition"), ("run_id", "other"),
])
def test_external_anchor_mismatch_refused(fixture, key, bad):
    directory, args = fixture
    args = dict(args, **{key: bad})
    with pytest.raises(ValueError):
        strict.consume(directory, **args)


def test_source_and_model_anchor_mismatch_refused(fixture):
    directory, args = fixture
    manifest = dict(args["expected_manifest"], tree="0" * 64)
    with pytest.raises(ValueError, match="source manifest"):
        strict.consume(directory, **dict(args, expected_manifest=manifest,
                                         expected_manifest_sha256=core.digest(manifest)))
    model = dict(args["expected_model_hashes"], all_buffers="0" * 64)
    with pytest.raises(ValueError, match="binding"):
        strict.consume(directory, **dict(args, expected_model_hashes=model))


def test_runner_complete_stub_without_native_witness_is_refused(tmp_path, monkeypatch):
    make, _, task, contract_sha, hashes = stub_setup(tmp_path, monkeypatch)
    r = make()
    r.plan()
    out = r.finish()
    manifest = strict.source_manifest()
    with pytest.raises(ValueError):
        strict.consume(tmp_path / "run", native_sha256=out["native_io_sha256"], ledger_sha256=out["sha256"],
                       contract_sha256=contract_sha, expected_task=task, expected_task_sha256=core.digest(task),
                       expected_model_hashes=hashes, expected_manifest=manifest,
                       expected_manifest_sha256=core.digest(manifest), arm="joint", run_id="stub")


def rewrite_chain(path, mutate):
    rows = durable.read_chain(path.read_bytes())
    mutate(rows)
    previous, chunks = None, []
    for i, row in enumerate(rows):
        body = dict(index=i, previous=previous, kind=row["kind"], payload=row["payload"])
        previous = transport.digest(body)
        chunks.append(transport.canonical(dict(body, sha256=previous)) + b"\n")
    data = b"".join(chunks)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("tamper", ["request", "score", "missing_call", "io_index", "buffer", "task"])
def test_rechained_semantic_tampering_refused(fixture, tmp_path, tamper):
    directory, args = fixture
    target = tmp_path / "copy"
    target.mkdir()
    (target / "events.jsonl").write_bytes((directory / "events.jsonl").read_bytes())
    path = target / "native_io.jsonl"
    path.write_bytes((directory / "native_io.jsonl").read_bytes())
    def mutate(rows):
        if tamper == "request":
            event = next(r["payload"] for r in rows if r["kind"] == "native"
                         and r["payload"].get("kind") == "native_call_attempt")
            event["request"]["horizon"] = 1
        elif tamper == "score":
            event = next(r["payload"] for r in rows if r["kind"] == "native"
                         and r["payload"].get("kind") == "plan_return")
            event["result"]["candidates"][0]["score"] += 1
        elif tamper == "missing_call":
            i = next(i for i, r in enumerate(rows) if r["kind"] == "native"
                     and r["payload"].get("kind") == "native_call_return")
            rows.pop(i)
        elif tamper == "io_index":
            event = next(r["payload"] for r in rows if r["kind"] == "native"
                         and r["payload"].get("operation_index") is not None)
            event["operation_index"] = 999999
        elif tamper == "buffer":
            event = next(r["payload"] for r in rows if r["kind"] == "native"
                         and r["payload"].get("kind") == "io_input")
            next(iter(event["arrays"].values()))["sha256"] = "0" * 64
        else:
            event = next(r["payload"] for r in rows if r["kind"] == "native"
                         and r["payload"].get("kind") == "native_call_attempt")
            event["task"]["goal"][0] += 1
    new_sha = rewrite_chain(path, mutate)
    with pytest.raises((ValueError, KeyError, IndexError)):
        strict.consume(target, **dict(args, native_sha256=new_sha))
