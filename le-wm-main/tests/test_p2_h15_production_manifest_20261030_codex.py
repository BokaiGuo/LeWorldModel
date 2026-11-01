"""CPU-only fake model/precision snapshots and historical saved transcript tests."""
import copy
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_h15_cuda_ledger_runner_20261030_codex as runner
from experiments.spectral_reachability import p2_h15_production_manifest_20261030_codex as prod
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec
from test_p2_h15_neural_ledger_adapter_20261030_codex import saved_signal, contract
from test_p2_h15_strict_saved_consumer_20261030_codex import fixture as saved_fixture, rewrite_chain


class FakeModel:
    training = False
    def named_modules(self, remove_duplicate=False):
        assert remove_duplicate is False
        return [("", self)]
    def modules(self):
        return [self]
    def parameters(self):
        return [SimpleNamespace(requires_grad=False, device=SimpleNamespace(type="cuda"), dtype="float32")]
    def buffers(self):
        return []


def fake_torch():
    cuda = SimpleNamespace(math_sdp_enabled=lambda: True, flash_sdp_enabled=lambda: False,
                           mem_efficient_sdp_enabled=lambda: False, cudnn_sdp_enabled=lambda: False,
                           matmul=SimpleNamespace(allow_tf32=False))
    return SimpleNamespace(float32="float32",
                           backends=SimpleNamespace(cuda=cuda, cudnn=SimpleNamespace(allow_tf32=False)),
                           is_autocast_enabled=lambda _: False)


def values(monkeypatch):
    _, task, _, _ = saved_signal()
    model, torch = FakeModel(), fake_torch()
    hashes = {"original_state_dict": task["binding"]["model_sha256"], "all_buffers": "b" * 64}
    monkeypatch.setattr(runner.signal, "model_hash", lambda _: hashes.copy())
    c = contract(task)
    classes = prod.capture_classes(model)
    manifest = prod.manifest(c, task, hashes, classes)
    return task, model, torch, hashes, c, manifest


def test_all_axes_bound_and_gate_refuses(tmp_path, monkeypatch):
    task, model, torch, _, c, value = values(monkeypatch)
    assert prod.validate_manifest(value, core.digest(value), c, task) == value
    assert prod.snapshot(model, torch, value)["classes"] == value["classes"]
    assert set(value["axes"]) == set(spec.AXES)
    for axis in spec.AXES:
        assert {k: value["axes"][axis][k] for k in ("unit", "cap", "aggregation")} == c["axes"][axis]
    assert not prod.production_gate(value)["ready_to_execute"]
    assert set(prod.production_gate(value)["missing_full_axis_coverage"]) == set(spec.AXES)


@pytest.mark.parametrize("change", ["class_hash", "class_name", "precision", "cap", "coverage", "sources", "contract"])
def test_recommitted_manifest_tampering_rejected(monkeypatch, change):
    task, _, _, _, c, value = values(monkeypatch)
    value = copy.deepcopy(value)
    if change == "class_hash":
        value["classes"][0]["sha256"] = "0" * 64
    elif change == "class_name":
        # The static validator accepts an externally committed name; live snapshot must reject it.
        value["classes"][0]["class_name"] = "invented.Class"
        prod.validate_manifest(value, core.digest(value), c, task)
        with pytest.raises(ValueError, match="class graph"):
            prod.snapshot(FakeModel(), fake_torch(), value)
        return
    elif change == "precision":
        value["precision"]["matmul_TF32"] = True
    elif change == "cap":
        value["axes"]["flops"]["cap"] += 1
    elif change == "coverage":
        value["axes"]["replay_calls"]["instrumentation"] = "complete"
    elif change == "sources":
        value["named_sources"]["runner"] = "0" * 64
    else:
        value["contract_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        prod.validate_manifest(value, core.digest(value), c, task)


def test_precision_model_and_class_drift_rejected(monkeypatch):
    _, model, torch, hashes, _, value = values(monkeypatch)
    torch.backends.cuda.matmul.allow_tf32 = True
    with pytest.raises(ValueError, match="TF32"):
        prod.snapshot(model, torch, value)
    torch.backends.cuda.matmul.allow_tf32 = False
    hashes["all_buffers"] = "c" * 64
    with pytest.raises(ValueError, match="buffers"):
        prod.snapshot(model, torch, value)
    hashes["all_buffers"] = "b" * 64
    model.forward = lambda x: x
    with pytest.raises(ValueError, match="forward override"):
        prod.snapshot(model, torch, value)


def test_class_file_byte_drift_rejected(tmp_path, monkeypatch):
    task, model, _, hashes, c, _ = values(monkeypatch)
    source = tmp_path / "model_class.py"
    source.write_text("class Model: pass\n")
    monkeypatch.setattr(prod.inspect, "getfile", lambda cls: str(source))
    value = prod.manifest(c, task, hashes, prod.capture_classes(model))
    prod.validate_manifest(value, core.digest(value), c, task)
    source.write_text("class Model: changed = True\n")
    with pytest.raises(ValueError, match="class source commitment"):
        prod.validate_manifest(value, core.digest(value), c, task)


def test_attestation_begin_rejects_bad_precision_fail_stops(tmp_path, monkeypatch):
    task, model, torch, _, c, value = values(monkeypatch)
    class Stub:
        def __init__(self, *_):
            pass
    monkeypatch.setattr(runner.tree, "NativeTree", Stub)
    r = runner.Runner(tmp_path / "run", c, core.digest(c), dict.fromkeys(core.AXES, 10000),
                      "joint", "fake", task, model, SimpleNamespace(type="cuda"))
    torch.backends.cuda.matmul.allow_tf32 = True
    with pytest.raises(ValueError, match="TF32"):
        prod.Attestation(r, value, core.digest(value), torch)
    assert r.driver.failed and not (tmp_path / "run" / "production_attestation.jsonl").exists()
    assert r.finish()["status"] == "failed_or_blocked"


@pytest.mark.parametrize("drift", [False, True])
def test_attestation_lifecycle_stub_only(tmp_path, monkeypatch, drift):
    report, task, _, response = saved_signal()
    task2, model, torch, hashes, c, value = values(monkeypatch)
    assert task == task2
    class Stub:
        def __init__(self, *_):
            self.index = 0
        def signal(self, request, ledger):
            return ledger.call("predict", lambda: response, shape=[1, 3, 192])
        def replay(self, context, actions, segmented, ledger):
            index = self.index if self.index < 40 else report["result"]["winner"]
            self.index += 1
            return ledger.call("predict", lambda: report["result"]["candidates"][index]["path"],
                               shape=[1, 3, 192])
    monkeypatch.setattr(runner.tree, "NativeTree", Stub)
    r = runner.Runner(tmp_path / "run", c, core.digest(c), dict.fromkeys(core.AXES, 10000),
                      "joint", "stub", task, model, SimpleNamespace(type="cuda"))
    attestation = prod.Attestation(r, value, core.digest(value), torch)
    r.plan()
    output = r.finish()
    if drift:
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(ValueError, match="TF32"):
            attestation.finish(output)
        rows = prod.strict._read(attestation.journal.path,
                                 spec.sha(attestation.journal.path.read_bytes()), prod.LIMIT)
        assert [row["kind"] for row in rows] == ["before"]
    else:
        sidecar = attestation.finish(output)
        assert not sidecar["ready_to_execute"]
        with pytest.raises(ValueError):
            prod.consume(tmp_path / "run", attestation_sha256=sidecar["sha256"], manifest_value=value,
                         manifest_sha256=core.digest(value), expected_task=task,
                         expected_task_sha256=core.digest(task), contract=c,
                         native_sha256=output["native_io_sha256"], ledger_sha256=output["sha256"],
                         arm="joint", run_id="stub")


def test_saved_strict_transcript_and_sidecar_binding(saved_fixture, tmp_path, monkeypatch):
    directory, args = saved_fixture
    _, model, torch, _, c, _ = values(monkeypatch)
    task = args["expected_task"]
    hashes = args["expected_model_hashes"]
    monkeypatch.setattr(runner.signal, "model_hash", lambda _: hashes.copy())
    value = prod.manifest(c, task, hashes, prod.capture_classes(model))
    # Synthetic sidecar verifies transport and cross-binding only; it is not a live attestation.
    target = tmp_path / "copy"
    target.mkdir()
    for name in ("events.jsonl", "native_io.jsonl"):
        (target / name).write_bytes((directory / name).read_bytes())
    snap = prod.snapshot(model, torch, value)
    journal = transport.Journal(target / "production_attestation.jsonl", prod.LIMIT)
    journal.emit("before", dict(schema=prod.NAME, manifest_sha256=core.digest(value), snapshot=snap))
    journal.emit("after", dict(snapshot=snap, native_sha256=args["native_sha256"],
                               ledger_sha256=args["ledger_sha256"], status="complete"))
    journal.close()
    sidecar_sha = spec.sha(journal.path.read_bytes())
    options = dict(attestation_sha256=sidecar_sha, manifest_value=value, manifest_sha256=core.digest(value),
                   expected_task=task, expected_task_sha256=core.digest(task), contract=c,
                   native_sha256=args["native_sha256"], ledger_sha256=args["ledger_sha256"],
                   arm=args["arm"], run_id=args["run_id"])
    result = prod.consume(target, **options)
    assert result["class_source_snapshots_recorded"] and result["precision_snapshots_recorded"]
    assert not result["continuous_precision_verified"] and not result["ready_to_execute"]
    with pytest.raises(ValueError):
        prod.consume(target, **dict(options, attestation_sha256="0" * 64))
    with pytest.raises(ValueError):
        prod.consume(target, **dict(options, ledger_sha256="0" * 64))
    def alter(rows):
        rows[-1]["payload"]["snapshot"]["precision"]["autocast_cuda"] = True
    changed = rewrite_chain(journal.path, alter)
    with pytest.raises(ValueError, match="sidecar cross-binding"):
        prod.consume(target, **dict(options, attestation_sha256=changed))
