import copy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as native
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as saved_backend
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_neural_ledger_adapter_20261030_codex as adapter
from experiments.spectral_reachability import p2_hook_contract_bridge_20261030_codex as prior
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec


def saved_signal():
    host = spec.read_bound(prior.HOST / "report.json", prior.HOST_SHA)
    root = prior.HOST / "root0"
    report = spec.read_bound(root / "report.json", host["slots"][0]["report_sha256"])
    raw = (root / "events.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == report["journal_sha256"]
    events = [x["payload"] for x in durable.read_chain(raw)]
    start = next(i for i, x in enumerate(events) if x.get("kind") == "native_call_attempt"
                 and x.get("method") == "signal")
    end = next(i for i in range(start + 1, len(events)) if events[i].get("kind") == "native_call_return")
    return report, events[start]["task"], events[start + 1:end], events[end]["result"]


def contract(task, *, predict=1000, vjp=1000):
    ident = {k: "a" * 64 for k in spec.IDENTITY}
    for target, src in adapter.IDENTITY_MAP.items():
        ident[target] = task["binding"][src]
    axes = {k: {"unit": unit, "aggregation": "max" if k == "peak_device_bytes" else "sum",
                "cap": 1000} for k, unit in spec.AXES.items()}
    axes["predict_samples"]["cap"] = predict
    axes["vjp_directions"]["cap"] = vjp
    axes["selection_evaluations"]["cap"] = 40
    return {"schema": spec.NAME + ".contract.v1", "arms": ["joint", "composition"],
            "scope": "per_root_per_arm_standalone_planning_and_selected_physical_execution",
            "identity": ident, "axes": axes}


def create(tmp_path, task, *, predict=1000, vjp=1000):
    c = contract(task, predict=predict, vjp=vjp)
    digest = spec.sha(spec.canonical(c))
    cpu, ledger, d = adapter.create(tmp_path / "run", c, digest, dict.fromkeys(core.AXES, 10000),
                                    "joint", "cpu-fixture", task)
    ledger.scope = {"phase": "signal"}
    return cpu, ledger, d, digest


def test_saved_real_neural_hook_tree_and_real_cpu_bank_share_ledger(tmp_path):
    report, task, events, signal = saved_signal()
    cpu, ledger, d, digest = create(tmp_path, task)
    saved_backend.replay_resources(events, ledger)  # inert metadata only, no model
    assert ledger.completed["predict_samples"] == 6
    assert ledger.completed["VJP_directions"] == 192
    bank = cpu.compile(signal["jacobian"], signal["nominal"], task["goal"])
    np.testing.assert_allclose(bank["logits"], report["result"]["bank"]["logits"], rtol=1e-11, atol=1e-11)
    scores = [cpu.score_candidate(i, row["path"][-1]["latent_history"][-1], task["goal"])
              for i, row in enumerate(report["result"]["candidates"])]
    assert int(np.argmin(scores)) == report["result"]["winner"]
    out = cpu.finish()
    verified = durable.verify(out["journal"], out["sha256"], digest)
    assert verified["completed"]["predict_samples"] == 6
    assert verified["completed"]["VJP_directions"] == 192
    assert verified["completed"]["array_SVD_calls"] == 2
    assert verified["completed"]["solve_calls"] == 1
    assert verified["completed"]["metric_calls"] == 40
    assert not verified["coverage_verified"] and not verified["contract_receipt_ready"]


def test_actual_native_call_method_reads_event_index_and_scope(tmp_path):
    _, task, _, _ = saved_signal()
    _, ledger, d, digest = create(tmp_path, task)
    captured = []
    class CpuHarness:
        def __init__(self):
            self.ledger = ledger
        def arrays(self, kind, operation_index=None, **values):
            captured.append((kind, operation_index, copy.deepcopy(self.ledger.scope)))
    harness = CpuHarness()
    value = native.NativeSignal.call(harness, "predict", lambda: np.array([7.]), {"latent": np.zeros(1)},
                                     shape=[1, 3, 192])
    assert value.tolist() == [7.]
    assert [(x[0], x[1]) for x in captured] == [("io_input", 0), ("io_return", 0)]
    assert captured[0][2]["phase"] == "signal"
    assert ledger.events[0]["index"] == 0
    out = d.finish()
    assert durable.verify(out["journal"], out["sha256"], digest)["completed"]["predict_samples"] == 1


def test_actual_native_instance_hooks_cpu_stub_and_restoration(tmp_path):
    _, task, _, _ = saved_signal()
    _, ledger, d, digest = create(tmp_path, task)
    called = []
    class Module:
        def __init__(self, name):
            self.name = name
        def forward(self, *args):
            called.append(self.name)
            return args[0]
        def __call__(self, *args):
            return self.forward(*args)
    class Model:
        def __init__(self):
            for name in ("action_encoder", "predictor", "pred_proj", "encoder", "projector"):
                setattr(self, name, Module(name))
    class Harness:
        def __init__(self):
            self.model, self.ledger = Model(), ledger
        def arrays(self, *args, **kwargs):
            return None
        call = native.NativeSignal.call
    h = Harness()
    with native.NativeSignal.observe(h):
        h.model.action_encoder(np.zeros((1, 3, 10)))
        h.model.predictor(np.zeros((1, 3, 192)), np.zeros((1, 3, 192)))
        h.model.pred_proj(np.zeros((3, 192)))
        with pytest.raises(ValueError, match="forbids visual"):
            h.model.encoder(np.zeros((1, 3, 192)))
    assert called == ["action_encoder", "predictor", "pred_proj"]
    assert all("forward" not in getattr(h.model, name).__dict__ for name in
               ("action_encoder", "predictor", "pred_proj", "encoder", "projector"))
    out = d.finish()
    check = durable.verify(out["journal"], out["sha256"], digest)
    assert check["status"] == "failed_or_blocked"
    assert check["completed"]["action_encoder_calls"] == 1
    assert check["completed"]["predictor_calls"] == 1
    assert check["completed"]["pred_proj_calls"] == 1
    assert check["attempted"]["encoder_calls"] == 1 and check["completed"]["encoder_calls"] == 0


@pytest.mark.parametrize("resource,kwargs,limits", [
    ("predict", {"shape": [1, 3, 192]}, {"predict": 0}),
    ("vjp", {"directions": 1}, {"vjp": 0}),
])
def test_neural_caps_refuse_before_call(tmp_path, resource, kwargs, limits):
    _, task, _, _ = saved_signal()
    _, ledger, d, digest = create(tmp_path, task, **limits)
    called = []
    with pytest.raises(core.BudgetExceeded):
        ledger.call(resource, lambda: called.append(True), **kwargs)
    assert called == []
    out = d.finish()
    assert durable.verify(out["journal"], out["sha256"], digest)["status"] == "failed_or_blocked"


def test_reject_identity_mismatch_before_directory_creation(tmp_path):
    _, task, _, _ = saved_signal()
    c = contract(task)
    c["identity"]["model_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identity"):
        adapter.create(tmp_path / "run", c, spec.sha(spec.canonical(c)),
                       dict.fromkeys(core.AXES, 100), "joint", "bad", task)
    assert not (tmp_path / "run").exists()


def test_neural_failure_is_nonrefundable_and_fail_stop(tmp_path):
    _, task, _, _ = saved_signal()
    _, ledger, d, digest = create(tmp_path, task)
    def fail():
        raise RuntimeError("native callback failure")
    with pytest.raises(RuntimeError):
        ledger.call("predict", fail, shape=[2, 3, 192])
    with pytest.raises(ValueError, match="fail-stopped"):
        ledger.call("predict", lambda: None, shape=[1, 3, 192])
    out = d.finish()
    check = durable.verify(out["journal"], out["sha256"], digest)
    assert check["attempted"]["predict_samples"] == 2 and check["completed"]["predict_samples"] == 0


def test_ledger_exposes_copies_and_rejects_source_or_task_drift(tmp_path):
    _, task, _, _ = saved_signal()
    _, ledger, d, digest = create(tmp_path, task)
    ledger.scope["phase"] = "injected"
    assert ledger.scope["phase"] == "signal"
    ledger.task["goal"][0] += 1
    with pytest.raises(ValueError, match="drift"):
        ledger.call("predict", lambda: None, shape=[1, 3, 192])
    out = d.finish()
    assert durable.verify(out["journal"], out["sha256"], digest)["status"] == "failed_or_blocked"
