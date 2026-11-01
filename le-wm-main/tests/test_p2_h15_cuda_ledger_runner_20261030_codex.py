"""CPU-only seam tests: fake native calls, real saved H15 task and CPU kernels."""
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_h15_cuda_ledger_runner_20261030_codex as runner
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec
from test_p2_h15_neural_ledger_adapter_20261030_codex import saved_signal, contract


def setup(tmp_path, monkeypatch, *, predict=1000, model_hash=None):
    report, task, _, saved = saved_signal()
    c = contract(task, predict=predict)
    digest = spec.sha(spec.canonical(c))
    model = object()
    hashes = {"original_state_dict": model_hash or task["binding"]["model_sha256"], "all_buffers": "b" * 64}
    monkeypatch.setattr(runner.signal, "model_hash", lambda _: hashes.copy())

    class Stub:
        def __init__(self, model_arg, device, task_arg, store):
            assert model_arg is model and task_arg == task and device.type == "cuda"
            self.store = store
            self.replays = 0

        def signal(self, request, ledger):
            return ledger.call("predict", lambda: saved, shape=[1, 3, 192])

        def replay(self, context, actions, segmented, ledger):
            index = self.replays if self.replays < 40 else report["result"]["winner"]
            self.replays += 1
            return ledger.call("predict", lambda: report["result"]["candidates"][index]["path"], shape=[1, 3, 192])

    monkeypatch.setattr(runner.tree, "NativeTree", Stub)
    def make():
        return runner.Runner(tmp_path / "run", c, digest, dict.fromkeys(core.AXES, 10000),
                             "joint", "stub", task, model, SimpleNamespace(type="cuda"))
    return make, report, task, digest, hashes


def test_shared_ledger_saved_task_cpu_compilation_and_native_seam(tmp_path, monkeypatch):
    make, report, task, digest, _ = setup(tmp_path, monkeypatch)
    r = make()
    response = r.signal()
    bank = r.cpu.compile(response["jacobian"], response["nominal"], task["goal"])
    assert len(bank["raw15"]) == 40
    path = r.replay(bank["raw15"][0], index=0)
    assert path == report["result"]["candidates"][0]["path"]
    for i, candidate in enumerate(report["result"]["candidates"]):
        r.cpu.score_candidate(i, candidate["path"][-1]["latent_history"][-1], task["goal"])
    out = r.finish()
    check = runner.verify(out, digest)
    assert check["status"] == "complete" and check["model_unchanged"]
    assert check["resources"]["completed"]["predict_samples"] == 2
    assert check["resources"]["completed"]["metric_calls"] == 40
    assert not check["coverage_verified"] and not check["budget_fairness"] and not check["P2_admission"]


def test_cap_refuses_before_native_callback_and_fails_closed(tmp_path, monkeypatch):
    make, _, _, digest, _ = setup(tmp_path, monkeypatch, predict=0)
    r = make()
    with pytest.raises(core.BudgetExceeded):
        r.signal()
    out = r.finish()
    check = runner.verify(out, digest)
    assert check["status"] == "failed_or_blocked"
    assert check["resources"]["completed"]["predict_samples"] == 0


def test_model_identity_rejected_before_artifact_creation(tmp_path, monkeypatch):
    make, _, _, _, _ = setup(tmp_path, monkeypatch, model_hash="c" * 64)
    with pytest.raises(ValueError, match="model identity"):
        make()
    assert not (tmp_path / "run").exists()


def test_model_drift_fail_stops_before_native_dispatch(tmp_path, monkeypatch):
    make, _, _, digest, hashes = setup(tmp_path, monkeypatch)
    r = make()
    hashes["all_buffers"] = "c" * 64
    with pytest.raises(ValueError, match="model drift"):
        r.signal()
    out = r.finish()
    check = runner.verify(out, digest)
    assert check["status"] == "failed_or_blocked" and not check["model_unchanged"]
    assert check["resources"]["completed"]["predict_samples"] == 0


def test_native_journal_tamper_detected(tmp_path, monkeypatch):
    make, _, _, digest, _ = setup(tmp_path, monkeypatch)
    r = make()
    r.signal()
    out = r.finish()
    with open(out["native_io"], "ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner.verify(out, digest)


def test_full_plan_on_saved_paths_stub(tmp_path, monkeypatch):
    make, report, _, digest, _ = setup(tmp_path, monkeypatch)
    r = make()
    result = r.plan()
    assert result["winner"] == report["result"]["winner"]
    out = r.finish()
    check = runner.verify(out, digest)
    assert check["status"] == "complete"
    assert check["resources"]["completed"]["predict_samples"] == 43
    assert check["resources"]["completed"]["metric_calls"] == 40


def test_bad_replay_scope_is_fail_stopped(tmp_path, monkeypatch):
    make, _, _, digest, _ = setup(tmp_path, monkeypatch)
    r = make()
    with pytest.raises(ValueError, match="replay scope"):
        r.replay([], index=-1)
    with pytest.raises(ValueError, match="fail-stopped"):
        r.signal()
    assert runner.verify(r.finish(), digest)["status"] == "failed_or_blocked"


def test_native_io_failure_prevents_dispatch(tmp_path, monkeypatch):
    make, _, _, digest, _ = setup(tmp_path, monkeypatch)
    r = make()
    original = r.io.emit
    def fail_native(kind, payload):
        if kind == "native":
            raise OSError("simulated journal failure")
        return original(kind, payload)
    monkeypatch.setattr(r.io, "emit", fail_native)
    with pytest.raises(OSError, match="journal failure"):
        r.signal()
    out = r.finish()
    check = runner.verify(out, digest)
    assert check["status"] == "failed_or_blocked"
    assert check["resources"]["completed"]["predict_samples"] == 0
